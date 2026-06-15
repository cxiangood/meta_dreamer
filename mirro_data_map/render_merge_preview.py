"""
渲染 exiD Location 0 的匝道汇入场景 BEV 预览。

使用 analyze_ramp_merge.py 的结果，选取有代表性的场景渲染。
"""

import os, sys, math
import numpy as np
import pandas as pd
import cv2
import imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bev_renderer import SUMONetwork, BEVRenderer

DATA_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1/data"
MAP_DIR = "/Users/jiojio/metadrive/mirro_data_map"
OUTPUT_DIR = os.path.join(MAP_DIR, "exid_merge_preview")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载地图 ──
print("加载 Location 0 SUMO 地图...")
net_xml = os.path.join(MAP_DIR, "exid_loc0_orig.net.xml")
network = SUMONetwork(net_xml)
renderer = BEVRenderer(network, resolution=256, view_range=50.0)
print(f"  中心: ({network.cx:.1f}, {network.cy:.1f})")

# ── 匝道 lanelet 集合（来自 analyze_ramp_merge.py 分析结果）──
# 速度 < 25 m/s 的 lanelet
lanelet_speeds = {}
for rec_id in range(19):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv"), low_memory=False)
    for ll_id in df["laneletId"].unique():
        ll_data = df[df["laneletId"] == ll_id]
        if ll_id not in lanelet_speeds:
            lanelet_speeds[ll_id] = []
        lanelet_speeds[ll_id].extend(ll_data["lonVelocity"].values.tolist())

ramp_lanelets = {ll for ll, spds in lanelet_speeds.items() if np.mean(spds) < 25.0}
main_lanelets = {ll for ll, spds in lanelet_speeds.items() if np.mean(spds) >= 25.0}

# ── 选取典型场景 ──
# 选取不同匝道路径、有邻居、速度变化明显的
selected = [
    (0, 59),    # ramp 1760→main 1673, 21→26 m/s
    (0, 87),    # ramp 1760→main 1673, 20→23 m/s
    (0, 65),    # ramp 1920→main 1664, 28→26 m/s (反方向匝道)
    (0, 133),   # ramp 1665→main 1764, 31→33 m/s (高速匝道)
    (0, 283),   # ramp 1760→main 1673, 18→22 m/s (低速汇入)
    (0, 188),   # ramp 1760→main 1672, 22→29 m/s (大幅加速)
]

print(f"\n渲染 {len(selected)} 个匝道汇入场景...")

for idx, (rec_id, tid) in enumerate(selected):
    print(f"\n[{idx+1}/{len(selected)}] rec{rec_id:02d}_track{tid}")

    # 加载轨迹
    tracks_csv = pd.read_csv(os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv"), low_memory=False)
    meta_csv = pd.read_csv(os.path.join(DATA_DIR, f"{rec_id:02d}_tracksMeta.csv"))

    ego_meta = meta_csv[meta_csv["trackId"] == tid]
    if len(ego_meta) == 0:
        print(f"  跳过：无 meta")
        continue

    f0 = int(ego_meta["initialFrame"].iloc[0])
    f1 = int(ego_meta["finalFrame"].iloc[0])
    n_frames = f1 - f0 + 1

    # 获取帧范围内的车辆
    rec_tracks = tracks_csv[
        (tracks_csv["frame"] >= f0) & (tracks_csv["frame"] <= f1)
    ]
    veh_counts = rec_tracks.groupby("trackId").size().sort_values(ascending=False)
    selected_ids = veh_counts.head(60).index.tolist()
    if tid not in selected_ids:
        selected_ids.append(tid)

    tracks = {}
    for vid in selected_ids:
        veh = rec_tracks[rec_tracks["trackId"] == vid]
        v_meta = meta_csv[meta_csv["trackId"] == vid]
        if len(v_meta) == 0:
            continue
        w = float(v_meta["width"].iloc[0])
        l = float(v_meta["length"].iloc[0])

        frames = veh["frame"].values - f0
        pos = np.zeros((n_frames, 2))
        heading = np.zeros(n_frames)
        valid = np.zeros(n_frames, dtype=bool)

        for fi, row in zip(frames, veh.itertuples()):
            if 0 <= fi < n_frames:
                pos[fi] = [row.xCenter, row.yCenter]
                heading[fi] = math.radians(row.heading)
                valid[fi] = True

        tracks[str(vid)] = {"pos": pos, "heading": heading, "valid": valid, "length": l, "width": w}

    # 居中坐标
    for tdata in tracks.values():
        tdata["pos"][:, 0] -= network.cx
        tdata["pos"][:, 1] -= network.cy

    # 找汇入帧（lanelet 从 ramp → main 的转折点）
    ego_df = tracks_csv[tracks_csv["trackId"] == tid].sort_values("frame")
    merge_local_frame = None
    prev_ll = None
    for _, row in ego_df.iterrows():
        cur_ll = row["laneletId"]
        if prev_ll is not None and prev_ll in ramp_lanelets and cur_ll in main_lanelets:
            merge_local_frame = int(row["frame"]) - f0
            break
        prev_ll = cur_ll

    if merge_local_frame is None:
        merge_local_frame = n_frames // 2

    # 渲染关键帧：汇入前20帧 → 汇入后15帧
    start = max(0, merge_local_frame - 20)
    end = min(n_frames, merge_local_frame + 15)
    key_frames = list(range(start, end))

    scenario_dir = os.path.join(OUTPUT_DIR, f"rec{rec_id:02d}_track{tid}")
    os.makedirs(scenario_dir, exist_ok=True)

    # 保存场景信息
    ego_before = ego_df[ego_df["frame"] < f0 + merge_local_frame]
    ego_after = ego_df[ego_df["frame"] >= f0 + merge_local_frame]
    sp_before = ego_before["lonVelocity"].mean() if len(ego_before) > 0 else 0
    sp_after = ego_after["lonVelocity"].mean() if len(ego_after) > 0 else 0
    ramp_lls = [ll for ll in ego_df["laneletId"].unique() if ll in ramp_lanelets]
    main_lls = [ll for ll in ego_df["laneletId"].unique() if ll in main_lanelets]

    with open(os.path.join(scenario_dir, "info.txt"), "w") as f:
        f.write(f"recording: {rec_id}\ntrackId: {tid}\n")
        f.write(f"total_frames: {n_frames}\nmerge_frame: {merge_local_frame}\n")
        f.write(f"speed: {sp_before:.1f} -> {sp_after:.1f} m/s\n")
        f.write(f"ramp_lanelets: {ramp_lls}\nmain_lanelets: {main_lls}\n")
        f.write(f"nearby_vehicles: {len(tracks)}\n")

    # 渲染帧
    frames_rgb = []
    for fi in key_frames:
        rgb, semantic = renderer.render_frame(tracks, tid, fi)

        # 在 RGB 上标注帧号和汇入状态
        status = "RAMP" if fi < merge_local_frame else ("MERGING" if fi == merge_local_frame else "MERGED")
        cv2.putText(rgb, f"F{fi:03d} {status}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(rgb, f"v={sp_before if fi < merge_local_frame else sp_after:.0f}m/s", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

        cv2.imwrite(os.path.join(scenario_dir, f"frame_{fi:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        frames_rgb.append(rgb)

    # 保存 GIF
    if frames_rgb:
        # 下采样到合理帧率
        step = max(1, len(frames_rgb) // 30)
        gif_frames = frames_rgb[::step]
        gif_path = os.path.join(scenario_dir, "preview.gif")
        imageio.mimsave(gif_path, gif_frames, fps=8, loop=0)
        print(f"  ✓ {len(key_frames)} 帧, GIF {len(gif_frames)} 帧 | "
              f"速度 {sp_before:.1f}→{sp_after:.1f} m/s | 汇入帧={merge_local_frame}")

print(f"\n{'='*60}")
print(f"预览已保存到: {OUTPUT_DIR}")
print(f"每个场景有: preview.gif, frame_XXXX.png, info.txt")
print(f"{'='*60}")
