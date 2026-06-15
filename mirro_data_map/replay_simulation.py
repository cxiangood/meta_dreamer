"""
Replay highway-merge-in trajectories on the SUMO-based map in MetaDrive.

Usage:
    # Headless (BEV top-down images saved every N frames)
    python3 mirro_data_map/replay_simulation.py --steps 50

    # 3D viewer (needs display)
    python3 mirro_data_map/replay_simulation.py --render

    # Both: 3D + periodic BEV screenshots
    python3 mirro_data_map/replay_simulation.py --render --bev --bev-every 20

    # Choose ego vehicle
    python3 mirro_data_map/replay_simulation.py --render --track-id 42
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# ── ensure imports work from repo root ──
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.scenario_description import ScenarioDescription as SD

import math
import pandas as pd
from metadrive.type import MetaDriveType

LANE_WIDTH_MAIN = 3.75
LANE_WIDTH_RAMP = 3.0

DATASET_DIR = "/Users/jiojio/Documents/课题组/毕设/mirro_dataset_on_ramp/Highway-merge-in"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def to_sumo(localY, localX):
    return localY, -localX


def build_scenario(dataset_dir: str = DATASET_DIR, sdc_track_id=None, max_traffic=48):
    """Build the full ScenarioDescription from CSV data in localXY space."""
    traj = pd.read_csv(os.path.join(dataset_dir, "Trajectory.csv"))
    traj.columns = [c.strip() for c in traj.columns]
    meta = pd.read_csv(os.path.join(dataset_dir, "TrackIDstate.csv"))

    if sdc_track_id is None:
        if meta["RampVehicle"].any():
            sdc_track_id = int(meta.loc[meta["RampVehicle"], "trackId"].iloc[0])
        else:
            sdc_track_id = int(meta["trackId"].iloc[0])

    ego_row = meta.loc[meta["trackId"] == sdc_track_id].iloc[0]
    f0 = int(ego_row["InitialFrame"])
    f1 = f0 + int(ego_row["TotalFrame"]) - 1
    ego_class = str(ego_row["VehicleClass"])
    window = traj[(traj["frameId"] >= f0) & (traj["frameId"] <= f1)].copy()

    # Select nearby vehicles
    candidates = [int(t) for t in window["trackId"].unique() if int(t) != sdc_track_id][:max_traffic]
    selected = [sdc_track_id] + candidates
    meta_by_id = meta.set_index("trackId")
    t_len = f1 - f0 + 1
    ts = np.arange(t_len, dtype=np.float32) * 0.1

    # ── Build map features (same as _build_and_render.py) ──
    lane1_sy = -(LANE_WIDTH_MAIN / 2)
    lane2_sy = -(LANE_WIDTH_MAIN + LANE_WIDTH_MAIN / 2)
    road_x_start, road_x_end = 220.0, -10.0

    features = {}
    for lid, sy, in [("1", lane1_sy), ("2", lane2_sy)]:
        features[lid] = {
            "type": MetaDriveType.LANE_SURFACE_STREET,
            "polyline": np.array([[road_x_start, sy], [road_x_end, sy]], dtype=np.float32),
        }

    # Ramp centerline from data
    ramp_data = traj[traj["laneId"] == 3].dropna(subset=["localX", "localY"])
    lane2_data = traj[traj["laneId"] == 2].dropna(subset=["localX", "localY"])

    slab_step = 10.0
    ly_vals = ramp_data["localY"].values
    ly_min, ly_max = np.percentile(ly_vals, 1), np.percentile(ly_vals, 99)
    edges = np.arange(math.floor(ly_min / slab_step) * slab_step, ly_max + slab_step, slab_step)

    ramp_slabs = []
    for i in range(len(edges) - 1):
        m = (ramp_data["localY"] >= edges[i]) & (ramp_data["localY"] < edges[i + 1])
        if m.sum() < 5:
            continue
        ramp_slabs.append((0.5 * (edges[i] + edges[i + 1]),
                           float(ramp_data.loc[m, "localX"].median())))

    lane2_slabs = {}
    for i in range(len(edges) - 1):
        m = (lane2_data["localY"] >= edges[i]) & (lane2_data["localY"] < edges[i + 1])
        if m.sum() < 5:
            continue
        mid = 0.5 * (edges[i] + edges[i + 1])
        lane2_slabs[mid] = float(lane2_data.loc[m, "localX"].median())

    ramp_slabs = np.array(ramp_slabs)
    ramp_shape_pts = []
    for ly, lx in ramp_slabs[np.argsort(-ramp_slabs[:, 0])]:
        sx, sy = to_sumo(ly, lx)
        ramp_shape_pts.append((sx, sy))

    # Convergence curve to merge point
    last_sx, last_sy = ramp_shape_pts[-1]
    merge_end_ly = 10.0
    merge_target_lx = lane2_slabs.get(merge_end_ly, -5.625)
    merge_sx, merge_sy = to_sumo(merge_end_ly, merge_target_lx)
    for i in range(1, 9):
        t = i / 8
        ts_val = 3 * t ** 2 - 2 * t ** 3
        ramp_shape_pts.append((last_sx + ts_val * (merge_sx - last_sx),
                               last_sy + ts_val * (merge_sy - last_sy)))

    features["3"] = {
        "type": MetaDriveType.LANE_SURFACE_STREET,
        "polyline": np.array(ramp_shape_pts, dtype=np.float32),
    }

    # Boundaries
    b12_y = (lane1_sy + lane2_sy) / 2
    features["boundary_1_2"] = {
        "type": MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
        "polyline": np.array([[road_x_start, b12_y], [road_x_end, b12_y]], dtype=np.float32),
    }
    features["edge_inner"] = {
        "type": MetaDriveType.BOUNDARY_LINE,
        "polyline": np.array([[road_x_start, lane1_sy - LANE_WIDTH_MAIN / 2],
                              [road_x_end, lane1_sy - LANE_WIDTH_MAIN / 2]], dtype=np.float32),
    }
    features["edge_outer"] = {
        "type": MetaDriveType.BOUNDARY_LINE,
        "polyline": np.array([[road_x_start, lane2_sy + LANE_WIDTH_MAIN / 2],
                              [road_x_end, lane2_sy + LANE_WIDTH_MAIN / 2]], dtype=np.float32),
    }
    ramp_edge = np.array([(sx, sy - LANE_WIDTH_RAMP / 2) for sx, sy in ramp_shape_pts], dtype=np.float32)
    features["edge_ramp"] = {"type": MetaDriveType.BOUNDARY_LINE, "polyline": ramp_edge}

    b23_pts = [(sx, (sy + lane2_sy) / 2) for sx, sy in ramp_shape_pts if sx >= 80]
    if len(b23_pts) >= 2:
        features["boundary_2_3"] = {
            "type": MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
            "polyline": np.array(b23_pts, dtype=np.float32),
        }

    # ── Build vehicle tracks ──
    tracks = {}
    for tid in selected:
        try:
            vclass = str(meta_by_id.loc[tid, "VehicleClass"])
        except KeyError:
            vclass = "car"
        sub = window[window["trackId"] == tid].sort_values("frameId")
        pos = np.zeros((t_len, 3), dtype=np.float32)
        vel = np.zeros((t_len, 2), dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)
        length = np.zeros(t_len, dtype=np.float32)
        width = np.zeros(t_len, dtype=np.float32)
        height = np.ones(t_len, dtype=np.float32) * 1.5

        for _, r in sub.iterrows():
            fi = int(float(r["frameId"])) - f0
            if fi < 0 or fi >= t_len:
                continue
            pos[fi, 0] = float(r["localY"])
            pos[fi, 1] = -float(r["localX"])
            pos[fi, 2] = 0.0
            vel[fi, 0] = float(r["xVelocity"])
            vel[fi, 1] = float(r["yVelocity"])
            valid[fi] = True
            w, h = float(r["width"]), float(r["height"])
            ref = 4.5 if "truck" not in vclass.lower() else 8.0
            sc = ref / max(w, h, 1.0)
            length[fi] = max(w, h) * sc
            width[fi] = min(w, h) * sc

        if not valid.any():
            continue
        heading = np.zeros(t_len, dtype=np.float32)
        idx = np.flatnonzero(valid)
        for k, i in enumerate(idx):
            i = int(i)
            if k == 0 and idx.size == 1:
                h = 0.0
            elif k == 0:
                d = pos[int(idx[k + 1])] - pos[i]
            elif k == idx.size - 1:
                d = pos[i] - pos[int(idx[k - 1])]
            else:
                d = pos[int(idx[k + 1])] - pos[int(idx[k - 1])]
            dn = float(np.hypot(d[0], d[1]))
            heading[i] = math.atan2(float(d[1]), float(d[0])) if dn > 0.05 else (
                float(heading[int(idx[k - 1])]) if k > 0 else 0.0)

        vel_out = np.zeros((t_len, 2), dtype=np.float32)
        for i in idx:
            i = int(i)
            spd = math.hypot(float(vel[i, 0]), float(vel[i, 1]))
            h = float(heading[i])
            vel_out[i, 0] = spd * math.cos(h)
            vel_out[i, 1] = spd * math.sin(h)

        tid_str = str(tid)
        tracks[tid_str] = {
            "type": MetaDriveType.VEHICLE,
            "state": {"position": pos, "velocity": vel_out, "heading": heading,
                      "valid": valid, "length": length, "width": width, "height": height},
            "metadata": {"type": MetaDriveType.VEHICLE, "object_id": tid_str,
                         "dataset": "highway_merge_in"},
        }

    sdc_str = str(sdc_track_id)
    scenario_dict = {
        "id": f"highway-merge-in-sumo-{sdc_track_id}",
        "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": MetaDriveType.COORDINATE_METADRIVE,
            "ts": ts, "sdc_id": sdc_str,
            "scenario_id": f"hmi_sumo_{sdc_track_id}",
            "dataset": "highway_merge_in",
            "source_file": os.path.basename(dataset_dir),
            "ego_vehicle_class": ego_class,
            "frame_range": (f0, f1),
        },
        "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": features,
    }
    SD.sanity_check(scenario_dict, check_self_type=True)
    return SD(scenario_dict)


def main():
    parser = argparse.ArgumentParser(description="Replay highway-merge-in on SUMO-based map")
    parser.add_argument("--render", action="store_true", help="Open 3D viewer")
    parser.add_argument("--bev", action="store_true", help="Save BEV screenshots during replay")
    parser.add_argument("--bev-every", type=int, default=30, help="Save BEV every N steps")
    parser.add_argument("--steps", type=int, default=None, help="Max steps (default: full scenario)")
    parser.add_argument("--track-id", type=int, default=None, help="Ego track id")
    parser.add_argument("--max-traffic", type=int, default=48)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    scenario = build_scenario(sdc_track_id=args.track_id, max_traffic=args.max_traffic)
    t_len = int(scenario["length"])
    n_steps = args.steps or t_len

    render = args.render
    if not render:
        os.environ["METADRIVE_HEADLESS"] = "1"

    env = ScenarioOnlineEnv(config=dict(
        use_render=render,
        agent_policy=ReplayEgoCarPolicy,
        horizon=n_steps + 50,
        set_static=True,
        camera_smooth=False,
        vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False),
    ))
    env.set_scenario(scenario)
    env.reset()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    bev_dir = os.path.join(out_dir, "bev_frames")
    if args.bev:
        os.makedirs(bev_dir, exist_ok=True)

    print(f"Scenario: {scenario['id']}, {t_len} frames, {len(scenario['tracks'])} vehicles")
    print(f"Running {n_steps} steps (render={render}, bev={args.bev})")

    for i in range(n_steps):
        o, r, tm, tc, info = env.step([0.0, 0.0])

        if args.bev and i % args.bev_every == 0:
            img = env.render(mode="topdown", window=False, screen_size=(2048, 1024),
                             film_size=(12000, 12000), semantic_map=False)
            if img is not None:
                path = os.path.join(bev_dir, f"frame_{i:04d}.png")
                cv2.imwrite(path, img)

        if tm or tc:
            print(f"  Terminated at step {i}")
            break

    # Final BEV with full map view (no vehicle following)
    map_img = env.render(mode="topdown", window=False, screen_size=(2048, 1024),
                         film_size=(12000, 12000), semantic_map=False)
    if map_img is not None:
        cv2.imwrite(os.path.join(out_dir, "bev_final.png"), map_img)
        print(f"  Final BEV → {out_dir}/bev_final.png")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
