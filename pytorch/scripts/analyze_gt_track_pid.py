#!/usr/bin/env python3
"""Analyze exiD GT trajectory vs PID control expectations for one track."""
import argparse
import copy
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_online_dreamer_v2 import build_map_features, build_scenario_dict  # noqa: E402

MERGE_ZONE = 30  # same as train_online MERGE_ZONE_FRAMES if defined there


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=int, default=79)
    ap.add_argument("--tid", type=int, default=97)
    ap.add_argument("--loc", type=int, default=6)
    ap.add_argument("--merge-idx", type=int, default=59)
    ap.add_argument("--fail-step", type=int, default=85,
                    help="Sim steps when GT episode ended (from log)")
    ap.add_argument("--data-dir", default="/share/home/u23516/data/exiD-dataset-v2.1/data")
    args = ap.parse_args()

    features, off_x, off_y = build_map_features(args.loc)
    _, t_len, (f0, f1) = build_scenario_dict(
        args.rec, args.tid, args.loc, copy.deepcopy(features), off_x, off_y, args.data_dir)

    csv_path = os.path.join(args.data_dir, f"{args.rec:02d}_tracks.csv")
    ego = pd.read_csv(csv_path, low_memory=False)
    ego = ego[ego["trackId"] == args.tid].sort_values("frame")
    seg = ego[(ego["frame"] >= f0) & (ego["frame"] <= f1)].copy()
    seg["fi"] = seg["frame"] - f0

    vx = seg["xVelocity"].values.astype(float)
    vy = seg["yVelocity"].values.astype(float)
    speed = np.sqrt(vx**2 + vy**2)
    heading_deg = seg["heading"].values.astype(float)
    x = seg["xCenter"].values.astype(float)
    y = seg["yCenter"].values.astype(float)

    # Kinematics from positions (verify velocity)
    dt = 0.04  # 25fps, decision_repeat=2 in sim but exiD frame = 25Hz
    pos_speed = np.zeros(len(seg))
    for i in range(1, len(seg)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        pos_speed[i] = np.hypot(dx, dy) / dt

    # Heading rate
    hd_rad = np.radians(heading_deg)
    yaw_rate = np.zeros(len(seg))
    for i in range(1, len(seg)):
        d = (hd_rad[i] - hd_rad[i - 1] + math.pi) % (2 * math.pi) - math.pi
        yaw_rate[i] = d / dt

    mi = args.merge_idx
    zones = {
        "pre_merge (phase0)": (0, mi),
        "merge_zone (phase1)": (mi, min(mi + MERGE_ZONE, t_len)),
        "post_merge (phase2)": (min(mi + MERGE_ZONE, t_len), t_len),
    }

    print("=" * 72)
    print(f"Track loc{args.loc}_rec{args.rec}_t{args.tid}  merge_idx={mi}  t_len={t_len}  frames [{f0},{f1}]")
    print(f"Sim GT episode ended at step={args.fail_step} ({100*args.fail_step/t_len:.1f}% of trajectory)")
    print("=" * 72)

    print("\n## 1. exiD original trajectory (scalar stats)")
    for name, (a, b) in zones.items():
        sl = slice(a, min(b, len(speed)))
        if a >= len(speed):
            continue
        sp = speed[sl]
        print(f"\n  [{name}] frames {a}:{b}")
        print(f"    speed (CSV):  min={sp.min():.2f} mean={sp.mean():.2f} max={sp.max():.2f} m/s")
        print(f"    speed (pos diff): mean={pos_speed[sl].mean():.2f} max={pos_speed[sl].max():.2f}")

    print("\n## 2. Speed profile around sim failure step")
    s = args.fail_step
    w = 15
    i0 = max(0, s - w)
    i1 = min(len(seg), s + w + 1)
    print(f"  frame index {i0}..{i1-1} (center={s}):")
    print("    fi   speed   d_speed   pos_spd   yaw_rate°/s")
    for i in range(i0, i1):
        fi = int(seg.iloc[i]["fi"]) if i < len(seg) else i
        ds = speed[i] - speed[i - 1] if i > 0 else 0.0
        print(f"    {fi:3d}  {speed[i]:6.2f}  {ds:+7.2f}  {pos_speed[i]:6.2f}  {math.degrees(yaw_rate[i]):+8.1f}")

    print("\n## 3. PID throttle sign analysis (open-loop, no sim)")
    # Same gains as world_model_policy.py
    KP, KI, KD, DT = 0.6, 0.15, 0.1, 1.0 / 25.0
    gt_speeds = speed.copy()
    if len(gt_speeds) < t_len:
        gt_speeds = np.pad(gt_speeds, (0, t_len - len(gt_speeds)), mode="edge")
    elif len(gt_speeds) > t_len:
        gt_speeds = gt_speeds[:t_len]

    # Assume MetaDrive starts at exiD frame-0 speed; then simulate speed with first-order lag
    current = float(speed[0])
    err_int = 0.0
    prev_err = 0.0
    sim_speed = []
    throttles = []
    for step in range(min(t_len, args.fail_step + 1)):
        desired = float(gt_speeds[step])
        err = desired - current
        err_int += err * DT
        err_int = float(np.clip(err_int, -3.0, 3.0))
        deriv = (err - prev_err) / DT
        thr = KP * err + KI * err_int + KD * deriv
        thr = float(np.clip(thr, -1.0, 1.0))
        throttles.append(thr)
        prev_err = err
        # crude plant: throttle scales accel, drag pulls to 0
        accel = thr * 3.0 - 0.15 * current
        current = max(0.0, current + accel * DT)
        sim_speed.append(current)

    th = np.array(throttles)
    print(f"  Open-loop PID plant (crude) over steps 0..{len(th)-1}:")
    print(f"    throttle: min={th.min():.3f} mean={th.mean():.3f} max={th.max():.3f}")
    print(f"    frac throttle<0: {(th < 0).mean()*100:.1f}%  frac throttle<-0.5: {(th < -0.5).mean()*100:.1f}%")
    print(f"    at fail_step {s}: desired={gt_speeds[s]:.2f} sim_speed={sim_speed[s]:.2f} throttle={th[s]:.3f}")

    # If MetaDrive overshoots exiD speed by +3 m/s at step 40
    print("\n  Sensitivity: constant +3 m/s speed overshoot from step 20:")
    current = float(speed[0])
    err_int = prev_err = 0.0
    neg_count = 0
    for step in range(args.fail_step + 1):
        desired = float(gt_speeds[step])
        current_sim = current + (3.0 if step >= 20 else 0.0)
        err = desired - current_sim
        err_int = float(np.clip(err_int + err * DT, -3.0, 3.0))
        deriv = (err - prev_err) / DT
        thr = float(np.clip(KP * err + KI * err_int + KD * deriv, -1.0, 1.0))
        if thr < 0:
            neg_count += 1
        prev_err = err
        accel = thr * 3.0 - 0.15 * current
        current = max(0.0, current + accel * DT)
    print(f"    steps with throttle<0 up to {args.fail_step}: {neg_count}/{args.fail_step+1}")

    print("\n## 4. Lateral / heading (exiD — PID has no lateral target from exiD)")
    lat_disp = np.zeros(len(seg))
    for i in range(1, len(seg)):
        dx, dy = x[i] - x[i - 1], y[i] - y[i - 1]
        h = hd_rad[i]
        # lateral increment in ego frame
        lat_disp[i] = -dx * math.sin(h) + dy * math.cos(h)
    cum_lat = np.cumsum(lat_disp)
    print(f"  cumulative lateral drift (ego frame, exiD): at fail_step {s}: {cum_lat[min(s, len(cum_lat)-1)]:.2f} m")
    print(f"  heading change over episode: {heading_deg[0]:.1f}° -> {heading_deg[min(s,len(seg)-1)]:.1f}°")
    print(f"  max |yaw_rate| in 0..{s}: {np.degrees(np.abs(yaw_rate[:s+1])).max():.1f} °/s")

    print("\n## 5. Merge-phase exiD behavior (what PID should do vs RL reward)")
    m0, m1 = mi, min(mi + MERGE_ZONE, len(speed))
    sp_merge = speed[m0:m1]
    print(f"  merge zone frames {m0}:{m1}: speed min={sp_merge.min():.2f} mean={sp_merge.mean():.2f} max={sp_merge.max():.2f}")
    if m0 > 0:
        print(f"  speed at merge-10 / merge / merge+10: {speed[m0-10]:.2f} / {speed[m0]:.2f} / {speed[min(m0+10,len(speed)-1)]:.2f}")
    if args.fail_step < mi:
        print(f"  >> Sim failed at step {args.fail_step} BEFORE merge_idx {mi} (still ramp/pre-merge in reward)")
    elif args.fail_step < mi + MERGE_ZONE:
        print(f"  >> Sim failed during merge zone window")
    else:
        print(f"  >> Sim failed after merge zone")

    print("\n## 6. Likely failure causes (checklist)")
    causes = []
    if args.fail_step < mi:
        causes.append("失败发生在汇入前：PID 尚未进入 merge 奖励窗口，属匝道/对准阶段")
    if speed[max(0, s - 5):s + 1].mean() < speed[max(0, s - 20):max(0, s - 10)].mean() - 1.0:
        causes.append(f"失败前 exiD 速度在下降（frame {s} 附近），PID 会跟刹")
    if th[s] < -0.1:
        causes.append(f"开环估计在 step {s} 油门为负 ({th[s]:.2f})，MetaDrive 会制动")
    if np.degrees(np.abs(yaw_rate[max(0, s - 10):s + 1])).max() > 8:
        causes.append("失败前航向变化率大，仅车道 PID 可能跟不上")
    causes.append("MetaDrive out_of_road = |navigation.current_lateral|>4m，BEV 可能仍看起来在路面上")
    causes.append("PID 不跟踪 exiD (x,y)，只跟踪标量速度 + 当前 lane，开环横向必然累积误差")
    for i, c in enumerate(causes, 1):
        print(f"  {i}. {c}")
    print()


if __name__ == "__main__":
    main()
