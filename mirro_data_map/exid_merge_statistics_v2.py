"""
更新后的 exiD 匝道汇入统计 - 标注 MetaDrive 导入状态。

地图文件 → MetaDrive 导入流程:
  OpenDRIVE .xodr → netconvert → SUMO .net.xml (with internal edges)
  → RoadLaneJunctionGraph → extract_map_features → MetaDrive 场景
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1/data"
MAP_DIR = "/Users/jiojio/metadrive/mirro_data_map"

LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}

LOC_MAPS = {
    0: "exid_loc0_orig.net.xml", 1: "exid_loc1_orig.net.xml",
    2: "exid_loc2_orig.net.xml", 3: "exid_loc3_orig.net.xml",
    4: "exid_loc4.net.xml",      5: "exid_loc5_orig.net.xml",
    6: "exid_loc6_orig.net.xml",
}

# 加载元信息
rec_metas = []
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith("_recordingMeta.csv"):
        rec_metas.append(pd.read_csv(os.path.join(DATA_DIR, f)))
all_meta = pd.concat(rec_metas, ignore_index=True)

# MetaDrive 导入信息（从刚才的测试获取）
LOC_METAINFO = {
    0: {"roads": 213, "lanes": 657, "junctions": 76},
    1: {"roads": 123, "lanes": 273, "junctions": 57},
    2: {"roads": 66,  "lanes": 179, "junctions": 22},
    3: {"roads": 132, "lanes": 315, "junctions": 41},
    4: {"roads": 130, "lanes": 391, "junctions": 45},
    5: {"roads": 135, "lanes": 315, "junctions": 41},
    6: {"roads": 70,  "lanes": 283, "junctions": 20},
}

# ── 按 location 逐个分析 ──
all_results = []
loc_summaries = []

for loc_id in sorted(all_meta["locationId"].unique()):
    rec_rows = all_meta[all_meta["locationId"] == loc_id].sort_values("recordingId")
    map_file = os.path.join(MAP_DIR, LOC_MAPS[loc_id])
    metainfo = LOC_METAINFO[loc_id]

    # 收集该 location 所有 lanelet 速度
    lanelet_speeds = {}
    for _, rm in rec_rows.iterrows():
        rec_id = int(rm["recordingId"])
        csv_path = os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        for ll_id in df["laneletId"].unique():
            if ll_id not in lanelet_speeds:
                lanelet_speeds[ll_id] = []
            lanelet_speeds[ll_id].extend(df[df["laneletId"] == ll_id]["lonVelocity"].values.tolist())

    loc_speed_limit = rec_rows["speedLimit"].iloc[0]
    threshold = loc_speed_limit * 0.85 if loc_speed_limit > 0 else 25.0

    ramp_lanelets = {ll for ll, spd in lanelet_speeds.items() if np.mean(spds := lanelet_speeds[ll]) < threshold}
    main_lanelets = {ll for ll, spd in lanelet_speeds.items() if np.mean(lanelet_speeds[ll]) >= threshold}

    # 逐 recording 统计
    loc_merge_total = 0
    traffic_counts = {"低": 0, "中": 0, "高": 0}
    traffic_merges = {"低": 0, "中": 0, "高": 0}
    traffic_tracks = {"低": 0, "中": 0, "高": 0}

    for _, rm in rec_rows.iterrows():
        rec_id = int(rm["recordingId"])
        csv_path = os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path, low_memory=False)
        n_tracks = int(rm["numTracks"])
        duration_s = rm["duration"] / rm["frameRate"]

        # 找匝道汇入车辆
        merge_count = 0
        for tid in df["trackId"].unique():
            veh = df[df["trackId"] == tid]
            if len(veh) < 20:
                continue
            lanelet_seq = veh.sort_values("frame")["laneletId"].values
            for i in range(1, len(lanelet_seq)):
                if lanelet_seq[i-1] in ramp_lanelets and lanelet_seq[i] in main_lanelets:
                    merge_count += 1
                    break

        traffic = "低" if n_tracks < 300 else ("中" if n_tracks < 800 else "高")
        traffic_counts[traffic] += 1
        traffic_merges[traffic] += merge_count
        traffic_tracks[traffic] += n_tracks
        loc_merge_total += merge_count

        all_results.append({
            "location": loc_id, "loc_name": LOC_NAMES[loc_id],
            "recording": rec_id, "n_tracks": n_tracks,
            "duration_s": round(duration_s, 1),
            "merge_count": merge_count, "traffic_level": traffic,
        })

    loc_summaries.append({
        "loc": loc_id, "name": LOC_NAMES[loc_id],
        "map_file": LOC_MAPS[loc_id],
        "roads": metainfo["roads"], "lanes": metainfo["lanes"], "junctions": metainfo["junctions"],
        "n_recs": len(rec_rows), "total_tracks": int(rec_rows["numTracks"].sum()),
        "total_merge": loc_merge_total,
        "traffic_low_recs": traffic_counts["低"], "traffic_mid_recs": traffic_counts["中"], "traffic_high_recs": traffic_counts["高"],
        "merge_low": traffic_merges["低"], "merge_mid": traffic_merges["中"], "merge_high": traffic_merges["高"],
    })

# ── 输出 ──
print(f"\n{'='*100}")
print(f"{'exiD 匝道汇入场景统计（MetaDrive 可导入）':^100}")
print(f"{'='*100}")
print()
print(f"{'Loc':>3} | {'名称':<22} | {'Recordings':>10} | {'总车辆':>7} | {'汇入总数':>8} | "
      f"{'低流量':>14} | {'中流量':>14} | {'高流量':>14} | {'地图文件':<25} | {'MetaDrive':>9}")
print(f"{'':>3} | {'':>22} | {'':>10} | {'':>7} | {'':>8} | "
      f"{'rec/汇入':>14} | {'rec/汇入':>14} | {'rec/汇入':>14} | {'':>25} | {'roads/lanes':>9}")
print("─" * 130)

grand_total = 0
for s in loc_summaries:
    grand_total += s["total_merge"]
    low_str = f"{s['traffic_low_recs']}rec/{s['merge_low']:>4}" if s['traffic_low_recs'] > 0 else "  -"
    mid_str = f"{s['traffic_mid_recs']}rec/{s['merge_mid']:>4}" if s['traffic_mid_recs'] > 0 else "  -"
    high_str = f"{s['traffic_high_recs']}rec/{s['merge_high']:>4}" if s['traffic_high_recs'] > 0 else "  -"

    print(f"{s['loc']:>3} | {s['name']:<22} | {s['n_recs']:>10} | {s['total_tracks']:>7} | {s['total_merge']:>8} | "
          f"{low_str:>14} | {mid_str:>14} | {high_str:>14} | {s['map_file']:<25} | "
          f"{s['roads']}r/{s['lanes']}l/{s['junctions']}j")

print("─" * 130)
print(f"{'':>3} | {'总计':<22} | {sum(s['n_recs'] for s in loc_summaries):>10} | "
      f"{sum(s['total_tracks'] for s in loc_summaries):>7} | {grand_total:>8} | "
      f"{sum(s['merge_low'] for s in loc_summaries):>14} | "
      f"{sum(s['merge_mid'] for s in loc_summaries):>14} | "
      f"{sum(s['merge_high'] for s in loc_summaries):>14}")

print(f"\n{'='*100}")
print(f"总计: {grand_total} 条匝道汇入, 全部 7 个 Location 地图已导入 MetaDrive")
print(f"{'='*100}")

# 保存
results_df = pd.DataFrame(all_results)
csv_path = os.path.join(MAP_DIR, "exid_merge_statistics.csv")
results_df.to_csv(csv_path, index=False)

summary_df = pd.DataFrame(loc_summaries)
summary_path = os.path.join(MAP_DIR, "exid_merge_summary.csv")
summary_df.to_csv(summary_path, index=False)

print(f"\n详细数据: {csv_path}")
print(f"汇总数据: {summary_path}")
