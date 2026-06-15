"""
探索 exiD 数据：筛选匝道汇入场景。

策略：
1. 查看每个 location 的地图结构（哪些有匝道）
2. 筛选在匝道附近进行换道的车辆
3. 统计各 location 的可用场景数量
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1/data"
MAP_DIR = "/Users/jiojio/metadrive/mirro_data_map"

print("=" * 70)
print("exiD 匝道汇入场景探索")
print("=" * 70)

# 1. 加载所有 recording 元信息
rec_metas = []
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith("_recordingMeta.csv"):
        rec_metas.append(pd.read_csv(os.path.join(DATA_DIR, f)))
all_meta = pd.concat(rec_metas, ignore_index=True)

print(f"\n总 recordings: {len(all_meta)}")
print(f"Locations: {sorted(all_meta['locationId'].unique())}")

# 每个location的recording数量
loc_counts = all_meta.groupby("locationId").agg(
    recordings=("recordingId", "count"),
    total_tracks=("numTracks", "sum"),
    avg_duration=("duration", "mean"),
).reset_index()
print("\n各 Location 统计:")
print(loc_counts.to_string(index=False))

# 2. 检查哪些 location 有 SUMO map
print("\n\n--- SUMO 地图可用性 ---")
available_locs = []
for loc in sorted(all_meta['locationId'].unique()):
    orig_map = os.path.join(MAP_DIR, f"exid_loc{loc}_orig.net.xml")
    plain_map = os.path.join(MAP_DIR, f"exid_loc{loc}.net.xml")
    if os.path.exists(orig_map):
        print(f"Location {loc}: exid_loc{loc}_orig.net.xml ✓ (with internal edges)")
        available_locs.append(loc)
    elif os.path.exists(plain_map):
        print(f"Location {loc}: exid_loc{loc}.net.xml ✓ (no internal edges)")
        available_locs.append(loc)
    else:
        # 检查 OpenDRIVE 源文件
        odr_dir = f"/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1/maps/opendrive"
        odr_files = []
        if os.path.exists(odr_dir):
            for d in os.listdir(odr_dir):
                if d.startswith(f"{loc}_"):
                    odr_path = os.path.join(odr_dir, d)
                    if os.path.isdir(odr_path):
                        for ff in os.listdir(odr_path):
                            if ff.endswith(".xodr"):
                                odr_files.append(os.path.join(odr_path, ff))
        if odr_files:
            print(f"Location {loc}: 无 SUMO map, 但有 OpenDRIVE: {odr_files[0]}")
        else:
            print(f"Location {loc}: 无 map")

# 3. 筛选换道车辆（匝道汇入候选）
print(f"\n\n{'='*70}")
print("换道车辆筛选 (仅 locations with SUMO map)")
print(f"{'='*70}")

for loc in available_locs:
    rec_ids = all_meta[all_meta["locationId"] == loc]["recordingId"].values
    total_lc_vehicles = 0
    total_lc_frames = 0

    for rec_id in rec_ids:
        tracks_csv = os.path.join(DATA_DIR, f"{int(rec_id):02d}_tracks.csv")
        if not os.path.exists(tracks_csv):
            continue

        df = pd.read_csv(tracks_csv, low_memory=False)
        # 筛选换道帧
        lc_frames = df[df["laneChange"] != 0]
        lc_vehicles = lc_frames["trackId"].unique()
        total_lc_vehicles += len(lc_vehicles)
        total_lc_frames += len(lc_frames)

    n_recs = len(rec_ids)
    print(f"\nLocation {loc}: {n_recs} recordings, {total_lc_vehicles} 换道车辆, {total_lc_frames} 换道帧")

# 4. 对 location 0 做详细分析：找出匝道附近的换道
print(f"\n\n{'='*70}")
print("Location 0 详细分析：匝道附近换道场景")
print(f"{'='*70}")

loc0_recs = all_meta[all_meta["locationId"] == 0]["recordingId"].values
ramp_scenarios = []

for rec_id in loc0_recs[:5]:  # 先看前5个recording
    tracks_csv = os.path.join(DATA_DIR, f"{int(rec_id):02d}_tracks.csv")
    tracks_meta_csv = os.path.join(DATA_DIR, f"{int(rec_id):02d}_tracksMeta.csv")

    df = pd.read_csv(tracks_csv, low_memory=False)
    meta = pd.read_csv(tracks_meta_csv)

    # 找换道车辆
    lc_ids = df[df["laneChange"] != 0]["trackId"].unique()

    for tid in lc_ids[:10]:  # 每个recording看前10个换道车
        veh = df[df["trackId"] == tid]
        lc_frames = veh[veh["laneChange"] != 0]

        if len(lc_frames) == 0:
            continue

        # 计算换道期间的位置范围
        f0, f1 = veh["frame"].min(), veh["frame"].max()
        x_range = veh["xCenter"].max() - veh["xCenter"].min()
        y_range = veh["yCenter"].max() - veh["yCenter"].min()
        avg_speed = veh["lonVelocity"].mean()

        # lanelet 变化 = 换道
        lanelets = veh["laneletId"].unique()

        # 检查是否有前后左右的邻居（交互场景）
        has_left_lead = (veh["leftLeadId"] != -1).any()
        has_left_rear = (veh["leftRearId"] != -1).any()
        has_right_lead = (veh["rightLeadId"] != -1).any()

        ramp_scenarios.append({
            "recording": int(rec_id),
            "trackId": tid,
            "numFrames": len(veh),
            "lc_frames": len(lc_frames),
            "x_range": f"{x_range:.0f}m",
            "y_range": f"{y_range:.0f}m",
            "avg_speed": f"{avg_speed:.1f}m/s",
            "lanelets": list(lanelets[:5]),
            "has_neighbors": has_left_lead or has_left_rear,
        })

    print(f"  Recording {int(rec_id):02d}: {len(lc_ids)} 换道车辆")

if ramp_scenarios:
    print(f"\n  找到 {len(ramp_scenarios)} 个候选场景（前5个recording）:")
    for s in ramp_scenarios[:20]:
        print(f"    rec{s['recording']:02d} track{s['trackId']:4d} | "
              f"{s['numFrames']:4d}帧 换道{s['lc_frames']:3d}帧 | "
              f"速度{s['avg_speed']} | Δx={s['x_range']} Δy={s['y_range']} | "
              f"lanelets={s['lanelets'][:3]} | 邻居={'Y' if s['has_neighbors'] else 'N'}")

# 5. 检查 location 0 的地图匝道结构
print(f"\n\n--- Location 0 地图匝道检查 ---")
try:
    import sumolib
    net = sumolib.net.readNet(os.path.join(MAP_DIR, "exid_loc0_orig.net.xml"), withInternal=True)
    edges = net.getEdges()
    # 查找包含 "ramp" 或 "merge" 的边
    ramp_edges = [e for e in edges if any(k in e.getID().lower() for k in ['ramp', 'merge', ':'])]
    print(f"总 edges: {len(edges)}")
    print(f"Junction/internal edges: {len([e for e in edges if e.getID().startswith(':')])}")

    # 查找速度限制不同的边（匝道通常速度限制较低）
    speed_limits = set()
    for e in edges:
        for lane in e.getLanes():
            speed_limits.add(round(lane.getSpeed(), 1))
    print(f"速度限制: {sorted(speed_limits)} m/s")

    # 低速边 = 可能是匝道
    low_speed_edges = []
    for e in edges:
        if not e.getID().startswith(':'):
            speeds = [round(l.getSpeed(), 1) for l in e.getLanes()]
            if any(s < 20.0 for s in speeds):
                low_speed_edges.append((e.getID(), speeds, e.getLaneNumber()))

    if low_speed_edges:
        print(f"\n低速边 (< 20 m/s, 可能匝道): {len(low_speed_edges)} 条")
        for eid, spds, nl in low_speed_edges[:10]:
            print(f"  {eid}: speed={spds}, lanes={nl}")
except Exception as e:
    print(f"SUMO 加载失败: {e}")

print("\n" + "=" * 70)
print("建议：先用 location 0 渲染几个换道场景看效果")
print("=" * 70)
