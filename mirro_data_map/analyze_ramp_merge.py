"""
从 exiD Location 0 筛选真正的匝道汇入场景。

匝道汇入特征（不是普通换道）：
1. 车辆起始在匝道车道（低 laneletId 或特定 lanelet），汇入到主路车道
2. 速度从低到高（匝道加速→主路速度）
3. 位置从匝道区域移向主路（大 lateral 位移）
4. lonVelocity 显著增加

先分析地图结构 → 找出匝道 lane → 找出从匝道汇入主路的车辆
"""

import os, sys, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1/data"
MAP_DIR = "/Users/jiojio/metadrive/mirro_data_map"

# ── 1. 分析 Location 0 地图：找匝道 ──
print("=" * 60)
print("Step 1: 分析 Location 0 地图结构，找出匝道车道")
print("=" * 60)

import sumolib
net = sumolib.net.readNet(
    os.path.join(MAP_DIR, "exid_loc0_orig.net.xml"), withInternal=True
)

# 按 laneletId 收集车道信息
edges_info = {}
for edge in net.getEdges():
    eid = edge.getID()
    if eid.startswith(":"):  # 跳过 junction internal
        continue
    lanes = edge.getLanes()
    speeds = [l.getSpeed() for l in lanes]
    from_node = edge.getFromNode().getID()
    to_node = edge.getToNode().getID()
    n_lanes = len(lanes)
    edge_type = edge.getFunction() if hasattr(edge, 'getFunction') else ""

    # 收集车道形状
    shapes = []
    for l in lanes:
        shape = l.getShape()
        # 计算车道的平均 Y
        ys = [p[1] for p in shape]
        shapes.append(np.mean(ys))

    edges_info[eid] = {
        "speeds": speeds,
        "avg_speed": np.mean(speeds),
        "n_lanes": n_lanes,
        "from": from_node,
        "to": to_node,
        "lane_ys": shapes,
    }

# 分类：主路 vs 匝道
# 主路：高速（>25 m/s）、多车道（>=3）
# 匝道：低速（<25 m/s）、少车道（1-2），或连接主路与低速区域
main_edges = {}
ramp_edges = {}

for eid, info in edges_info.items():
    avg_spd = info["avg_speed"]
    n_lanes = info["n_lanes"]

    if avg_spd >= 25.0 and n_lanes >= 3:
        main_edges[eid] = info
    elif avg_spd < 25.0 or n_lanes <= 2:
        ramp_edges[eid] = info

print(f"\n主路边: {len(main_edges)} 条")
for eid, info in sorted(main_edges.items())[:5]:
    print(f"  {eid}: {info['n_lanes']} lanes, {info['avg_speed']:.1f} m/s")

print(f"\n匝道/连接边: {len(ramp_edges)} 条")
for eid, info in sorted(ramp_edges.items()):
    print(f"  {eid}: {info['n_lanes']} lanes, {info['avg_speed']:.1f} m/s, "
          f"Y={info['lane_ys'][:3]}")

# ── 2. 找出 laneletId 到边/车道的映射 ──
# exiD 的 laneletId 对应 Lanelet2 格式，需要找映射关系
print(f"\n{'='*60}")
print("Step 2: 找 laneletId → 车道类型映射")
print("=" * 60)

# 从 tracks.csv 中提取所有 laneletId
all_lanelets = set()
for rec_id in range(19):
    tracks_csv = pd.read_csv(
        os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv"), low_memory=False
    )
    all_lanelets.update(tracks_csv["laneletId"].unique())

print(f"总 lanelet IDs: {len(all_lanelets)}")
print(f"ID 范围: {min(all_lanelets)} - {max(all_lanelets)}")

# 对每个 lanelet，看车辆的典型速度来判断是主路还是匝道
lanelet_speeds = {}
lanelet_y = {}
for rec_id in range(19):
    tracks_csv = pd.read_csv(
        os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv"), low_memory=False
    )
    for ll_id in all_lanelets:
        ll_data = tracks_csv[tracks_csv["laneletId"] == ll_id]
        if len(ll_data) > 0:
            if ll_id not in lanelet_speeds:
                lanelet_speeds[ll_id] = []
                lanelet_y[ll_id] = []
            lanelet_speeds[ll_id].extend(ll_data["lonVelocity"].values.tolist())
            lanelet_y[ll_id].extend(ll_data["yCenter"].values.tolist())

# 分类 lanelet
main_lanelets = set()
ramp_lanelets = set()

for ll_id, speeds in lanelet_speeds.items():
    avg_spd = np.mean(speeds)
    if avg_spd >= 25.0:
        main_lanelets.add(ll_id)
    else:
        ramp_lanelets.add(ll_id)

print(f"\n主路 lanelets (avg speed >= 25 m/s): {len(main_lanelets)}")
print(f"匝道/低速 lanelets (avg speed < 25 m/s): {len(ramp_lanelets)}")

print("\n匝道 lanelets 详情:")
for ll_id in sorted(ramp_lanelets):
    spds = lanelet_speeds[ll_id]
    ys = lanelet_y[ll_id]
    print(f"  lanelet {ll_id}: avg_speed={np.mean(spds):.1f} m/s, "
          f"avg_y={np.mean(ys):.0f}, n_pts={len(spds)}")

# ── 3. 找匝道汇入车辆 ──
print(f"\n{'='*60}")
print("Step 3: 筛选匝道汇入车辆")
print("=" * 60)

merge_scenarios = []
for rec_id in range(19):
    tracks_csv = pd.read_csv(
        os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv"), low_memory=False
    )
    meta_csv = pd.read_csv(os.path.join(DATA_DIR, f"{rec_id:02d}_tracksMeta.csv"))

    for tid in tracks_csv["trackId"].unique():
        veh = tracks_csv[tracks_csv["trackId"] == tid]
        v_meta = meta_csv[meta_csv["trackId"] == tid]
        if len(v_meta) == 0 or len(veh) < 30:
            continue

        # 检查 lanelet 变化：从匝道 lanelet → 主路 lanelet
        lanelet_seq = veh["laneletId"].values
        has_ramp_to_main = False
        merge_frame = -1

        for i in range(1, len(lanelet_seq)):
            if lanelet_seq[i-1] in ramp_lanelets and lanelet_seq[i] in main_lanelets:
                has_ramp_to_main = True
                merge_frame = i
                break

        if not has_ramp_to_main:
            continue

        # 速度变化：汇入前低速，汇入后高速
        speed_before = veh.iloc[:merge_frame]["lonVelocity"].mean() if merge_frame > 0 else 0
        speed_after = veh.iloc[merge_frame:]["lonVelocity"].mean()

        # 有无邻居
        has_neighbor = (
            (veh["leftLeadId"] != -1).any()
            or (veh["leftRearId"] != -1).any()
            or (veh["leadId"].apply(lambda x: x != -1 and x != '-1')).any()
        )

        # laneChange 标记
        has_lc = (veh["laneChange"] != 0).any()

        n_frames = len(veh)
        ramp_ll = [ll for ll in lanelet_seq if ll in ramp_lanelets]
        main_ll = [ll for ll in lanelet_seq if ll in main_lanelets]

        merge_scenarios.append({
            "recording": rec_id,
            "trackId": tid,
            "numFrames": n_frames,
            "merge_frame": merge_frame,
            "speed_before": speed_before,
            "speed_after": speed_after,
            "has_neighbor": has_neighbor,
            "has_laneChange": has_lc,
            "ramp_lanelets": list(set(ramp_ll))[:5],
            "main_lanelets": list(set(main_ll))[:5],
            "lanelet_seq": lanelet_seq.tolist(),
        })

print(f"\n找到 {len(merge_scenarios)} 个匝道汇入场景!")

for s in merge_scenarios[:30]:
    print(f"  rec{s['recording']:02d} track{s['trackId']:4d} | "
          f"{s['numFrames']:4d}帧 | 汇入帧={s['merge_frame']:3d} | "
          f"速度 {s['speed_before']:.1f}→{s['speed_after']:.1f} m/s | "
          f"邻居={'Y' if s['has_neighbor'] else 'N'} | "
          f"LC={'Y' if s['has_laneChange'] else 'N'} | "
          f"ramp_ll={s['ramp_lanelets'][:2]} → main_ll={s['main_lanelets'][:2]}")
