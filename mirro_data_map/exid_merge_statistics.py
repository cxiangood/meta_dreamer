"""
统计 exiD 全部 7 个 location 的匝道汇入场景。

对每个 recording:
  1. 按 lanelet 平均车速划分 主路/匝道 lanelet
  2. 找出 从匝道 lanelet → 主路 lanelet 的汇入车辆
  3. 按 总车辆数 划分车流量 低/中/高
  4. 汇总每个 location 的可用汇入数据量 + 对应地图文件
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1/data"
MAP_DIR = "/Users/jiojio/metadrive/mirro_data_map"
ODR_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1/maps/opendrive"

# ── 1. 加载所有 recording 元信息 ──
print("加载 recording 元信息...")
rec_metas = []
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith("_recordingMeta.csv"):
        rec_metas.append(pd.read_csv(os.path.join(DATA_DIR, f)))
all_meta = pd.concat(rec_metas, ignore_index=True)

# location 名称映射
LOC_NAMES = {
    0: "cologne_butzweiler",
    1: "cologne_fortiib",
    2: "aachen_brand",
    3: "bergheim_roemer",
    4: "cologne_klettenberg",
    5: "aachen_laurensberg",
    6: "merzenich_rather",
}

# 地图文件状态
def get_map_status(loc_id):
    orig = os.path.join(MAP_DIR, f"exid_loc{loc_id}_orig.net.xml")
    plain = os.path.join(MAP_DIR, f"exid_loc{loc_id}.net.xml")
    odr_dir = os.path.join(ODR_DIR, f"{loc_id}_{LOC_NAMES[loc_id]}")
    odr_file = os.path.join(odr_dir, f"{LOC_NAMES[loc_id]}.xodr")

    if os.path.exists(orig):
        return "SUMO ✓", orig
    elif os.path.exists(plain):
        return "SUMO ✓ (无 internal)", plain
    elif os.path.exists(odr_file):
        return "OpenDRIVE (待转换)", odr_file
    else:
        return "无地图", None

# ── 2. 逐 location 分析 ──
print(f"\n{'='*90}")
print(f"{'exiD 匝道汇入场景统计':^90}")
print(f"{'='*90}")

all_results = []  # 每条记录: (loc, rec, n_tracks, duration_s, merge_count, traffic_level)

for loc_id in sorted(all_meta["locationId"].unique()):
    rec_rows = all_meta[all_meta["locationId"] == loc_id].sort_values("recordingId")
    map_status, map_file = get_map_status(loc_id)

    # 先收集该 location 所有 lanelet 的速度（用于分类）
    print(f"\n{'─'*90}")
    print(f"Location {loc_id} ({LOC_NAMES[loc_id]}) | 地图: {map_status}")
    print(f"  Recordings: {len(rec_rows)} 条 (rec {rec_rows['recordingId'].min():.0f}-{rec_rows['recordingId'].max():.0f})")
    print(f"{'─'*90}")

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
            lanelet_speeds[ll_id].extend(
                df[df["laneletId"] == ll_id]["lonVelocity"].values.tolist()
            )

    # 按 speedLimit 或数据速度分类匝道/主路
    # 策略：该 location 所有 lanelet 平均速度的双峰分布
    ll_avg_speeds = {ll: np.mean(spds) for ll, spds in lanelet_speeds.items()}
    all_avg = sorted(ll_avg_speeds.values())

    # 用 speedLimit 作为阈值，如果没有明确的，用数据驱动
    loc_speed_limit = rec_rows["speedLimit"].iloc[0]
    if loc_speed_limit > 0:
        threshold = loc_speed_limit * 0.85  # 速度限制的 85% 作为分界
    else:
        # 数据驱动：找速度分布的自然间隙
        threshold = 25.0  # 默认

    ramp_lanelets = {ll for ll, spd in ll_avg_speeds.items() if spd < threshold}
    main_lanelets = {ll for ll, spd in ll_avg_speeds.items() if spd >= threshold}

    print(f"  速度阈值: {threshold:.1f} m/s (speedLimit={loc_speed_limit:.1f})")
    print(f"  匝道 lanelets: {len(ramp_lanelets)} 个 (低速)")
    print(f"  主路 lanelets: {len(main_lanelets)} 个 (高速)")

    # 如果匝道或主路为空，说明这个 location 可能没有匝道结构
    if not ramp_lanelets or not main_lanelets:
        print(f"  ⚠ 缺少匝道或主路 lanelet，跳过")
        for _, rm in rec_rows.iterrows():
            rec_id = int(rm["recordingId"])
            all_results.append({
                "location": loc_id, "loc_name": LOC_NAMES[loc_id],
                "recording": rec_id, "n_tracks": int(rm["numTracks"]),
                "duration_s": rm["duration"] / rm["frameRate"],
                "merge_count": 0, "traffic_level": "-",
                "map_status": map_status,
            })
        continue

    # 逐 recording 统计
    header_printed = False
    loc_merge_total = 0

    for _, rm in rec_rows.iterrows():
        rec_id = int(rm["recordingId"])
        csv_path = os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path, low_memory=False)
        n_tracks = int(rm["numTracks"])
        duration_s = rm["duration"] / rm["frameRate"]
        vehicles_per_min = n_tracks / (duration_s / 60) if duration_s > 0 else 0

        # 找匝道汇入车辆
        merge_count = 0
        merge_details = []

        for tid in df["trackId"].unique():
            veh = df[df["trackId"] == tid]
            if len(veh) < 20:
                continue

            lanelet_seq = veh.sort_values("frame")["laneletId"].values

            # 检查是否有 ramp → main 的 lanelet 转换
            for i in range(1, len(lanelet_seq)):
                if lanelet_seq[i-1] in ramp_lanelets and lanelet_seq[i] in main_lanelets:
                    speed_before = veh.iloc[:i]["lonVelocity"].mean()
                    speed_after = veh.iloc[i:]["lonVelocity"].mean()
                    merge_count += 1
                    merge_details.append({
                        "trackId": tid,
                        "merge_frame": i,
                        "speed_before": speed_before,
                        "speed_after": speed_after,
                    })
                    break

        loc_merge_total += merge_count

        # 车流量分级 (基于 recording 中总车辆数)
        if n_tracks < 300:
            traffic = "低"
        elif n_tracks < 800:
            traffic = "中"
        else:
            traffic = "高"

        all_results.append({
            "location": loc_id, "loc_name": LOC_NAMES[loc_id],
            "recording": rec_id, "n_tracks": n_tracks,
            "duration_s": duration_s,
            "merge_count": merge_count, "traffic_level": traffic,
            "map_status": map_status,
        })

        if not header_printed:
            print(f"\n  {'rec':>4s} | {'总车辆':>6s} | {'时长(s)':>7s} | {'车/min':>6s} | {'流量':>4s} | {'汇入数':>6s}")
            print(f"  {'─'*4}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*4}─┼─{'─'*6}")
            header_printed = True

        print(f"  {rec_id:4d} | {n_tracks:6d} | {duration_s:7.1f} | {vehicles_per_min:6.1f} | {traffic:>4s} | {merge_count:6d}")

    print(f"\n  ★ Location {loc_id} 汇入总数: {loc_merge_total} 条")

# ── 3. 汇总表 ──
results_df = pd.DataFrame(all_results)

print(f"\n{'='*90}")
print(f"{'汇 总':^90}")
print(f"{'='*90}")

# 按 location 汇总
for loc_id in sorted(results_df["location"].unique()):
    loc_df = results_df[results_df["location"] == loc_id]
    total_merge = loc_df["merge_count"].sum()
    total_tracks = loc_df["n_tracks"].sum()
    n_recs = len(loc_df)
    map_status = loc_df["map_status"].iloc[0]

    # 按流量分级统计
    traffic_summary = {}
    for level in ["低", "中", "高"]:
        sub = loc_df[loc_df["traffic_level"] == level]
        if len(sub) > 0:
            traffic_summary[level] = {
                "recs": len(sub),
                "tracks": sub["n_tracks"].sum(),
                "merges": sub["merge_count"].sum(),
            }

    # 地图文件
    _, map_file = get_map_status(loc_id)

    print(f"\nLocation {loc_id} ({LOC_NAMES[loc_id]})")
    print(f"  地图文件: {map_file if map_file else 'N/A'}")
    print(f"  Recordings: {n_recs} 条, 总车辆: {total_tracks}, 汇入总数: {total_merge}")
    for level, info in traffic_summary.items():
        print(f"    {level}流量: {info['recs']} 条 recording, {info['tracks']} 辆车, {info['merges']} 条汇入")

# ── 4. 总计 ──
total_all = results_df["merge_count"].sum()
print(f"\n{'='*90}")
print(f"总计: {len(results_df)} 条 recording, {total_all} 条匝道汇入数据")
print(f"{'='*90}")

# 保存 CSV
csv_path = os.path.join(MAP_DIR, "exid_merge_statistics.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n详细数据已保存到: {csv_path}")
