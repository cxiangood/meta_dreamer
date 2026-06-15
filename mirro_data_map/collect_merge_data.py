"""
收集 exiD 匝道汇入场景的 DreamerV3 训练数据。

对每条匝道汇入轨迹:
  1. 在 MetaDrive 中回放，用 RGBCamera 捕获 BEV 图像
  2. 从 CSV 帧差计算 action (steering, throttle)
  3. 从 laneletId/velocity 计算启发式 reward
  4. 保存为 .npz 文件

用法:
    # 单条轨迹
    python3 mirro_data_map/collect_merge_data.py --recording 0 --track-id 59

    # 整个 recording 的所有汇入
    python3 mirro_data_map/collect_merge_data.py --recording 0

    # 整个 location (所有 recording)
    python3 mirro_data_map/collect_merge_data.py --location 0

    # 干跑 (只统计和计算 reward，不启动 MetaDrive)
    python3 mirro_data_map/collect_merge_data.py --recording 0 --dry-run
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── 配置 (HPC 和本地通用) ──
SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo",
)
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
os.environ["SUMO_HOME"] = SUMO_HOME
os.environ["METADRIVE_HEADLESS"] = "1"

import sumolib
from metadrive.type import MetaDriveType as MT
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features

DATASET_DIR = os.environ.get(
    "EXID_DATASET_DIR",
    "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1",
)
DATA_DIR = os.path.join(DATASET_DIR, "data")
MAP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(MAP_DIR, "exid_dreamer_data")

LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}

# BEV 相机参数
BEV_HEIGHT = 50.0
BEV_PITCH = -89.0
BEV_W, BEV_H = 400, 300  # 2x 分辨率
FPS = 25

# ── Reward 权重 (v2: 速度进度 + 加减速) ──
W_BASE_COST = -0.2   # 每步基础成本
W_SPEED_FRAC = 0.5   # 速度进度（占主路速度比例）
W_ACCEL = 0.2        # 加速奖励
W_DECEL = -0.3       # 减速惩罚
W_SPEED_POST = 0.3   # 汇入后速度保持
W_PROGRESS = 5.0     # 汇入完成奖励
W_SUCCESS = 10.0     # 成功完成奖励


def get_map_file(loc_id):
    orig = os.path.join(MAP_DIR, f"exid_loc{loc_id}_orig.net.xml")
    plain = os.path.join(MAP_DIR, f"exid_loc{loc_id}.net.xml")
    if os.path.exists(orig):
        return orig
    if os.path.exists(plain):
        return plain
    raise FileNotFoundError(f"No SUMO map for location {loc_id}")


def build_map_features(loc_id):
    """加载 SUMO 地图并提取 map features（带重采样）。"""
    net_xml = get_map_file(loc_id)
    raw_net = sumolib.net.readNet(net_xml, withInternal=True)
    xmin, ymin, xmax, ymax = raw_net.getBoundary()
    off_x = -(xmax + xmin) / 2
    off_y = -(ymax + ymin) / 2

    graph = RoadLaneJunctionGraph(net_xml)
    features = extract_map_features(graph)

    # 重采样短线段
    SPACING = 2.0
    for k, v in features.items():
        if v.get("type") in (MT.LINE_BROKEN_SINGLE_WHITE, MT.LINE_BROKEN_SINGLE_YELLOW):
            pl = v.get("polyline", [])
            if len(pl) < 4:
                pl = np.array(pl, dtype=np.float32)
                d = pl[-1] - pl[0]
                length = float(np.hypot(d[0], d[1]))
                n = max(int(length / SPACING), 4)
                pts = [pl[0]]
                for i in range(1, n):
                    pts.append(pl[0] + (i / n) * d)
                pts.append(pl[-1])
                v["polyline"] = np.array(pts, dtype=np.float32)

    return features, off_x, off_y


def load_lanelet2_onramp_ids(loc_id):
    """从 Lanelet2 .osm 地图读取 onramp 标注。"""
    import xml.etree.ElementTree as ET
    osm_path = os.path.join(
        DATASET_DIR, "maps", "lanelet2",
        f"{loc_id}_{LOC_NAMES[loc_id]}", f"location{loc_id}.osm",
    )
    if not os.path.exists(osm_path):
        return None, None
    tree = ET.parse(osm_path)
    root = tree.getroot()
    onramp_ids = set()
    highway_ids = set()
    for rel in root.findall("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        lid = int(rel.get("id"))
        if tags.get("onramp") == "yes":
            onramp_ids.add(lid)
        elif tags.get("subtype") == "highway":
            highway_ids.add(lid)
    return onramp_ids, highway_ids


def classify_lanelets(rec_ids, speed_limit):
    """按 Lanelet2 onramp 标注划分匝道/主路 lanelet。

    优先使用 Lanelet2 的 ground truth 标注（onramp=yes）。
    如 Lanelet2 地图不可用，回退到速度分类。
    """
    # 从 recording ID 获取 location ID
    meta = pd.read_csv(os.path.join(DATA_DIR, f"{rec_ids[0]:02d}_recordingMeta.csv"))
    loc_id = int(meta["locationId"].iloc[0])

    onramp_ids, highway_ids = load_lanelet2_onramp_ids(loc_id)
    if onramp_ids and highway_ids:
        # 用 Lanelet2 ground truth
        # 计算主路平均速度
        dfs = []
        for rec_id in rec_ids:
            csv_path = os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv")
            if not os.path.exists(csv_path):
                continue
            dfs.append(pd.read_csv(csv_path, usecols=["laneletId", "lonVelocity"]))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            ll_avg = df.groupby("laneletId")["lonVelocity"].mean()
            main_speed = float(ll_avg[ll_avg.index.isin(highway_ids)].mean()) if highway_ids else speed_limit
        else:
            main_speed = speed_limit
        print(f"  [Lanelet2 GT] onramp={len(onramp_ids)}, highway={len(highway_ids)}, main_speed={main_speed:.1f}", file=sys.stderr)
        return onramp_ids, highway_ids, main_speed

    # 回退：速度分类
    dfs = []
    for rec_id in rec_ids:
        csv_path = os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv")
        if not os.path.exists(csv_path):
            continue
        dfs.append(pd.read_csv(csv_path, usecols=["laneletId", "lonVelocity"]))
    if not dfs:
        return set(), set(), speed_limit
    df = pd.concat(dfs, ignore_index=True)
    ll_avg = df.groupby("laneletId")["lonVelocity"].mean()
    if speed_limit > 0:
        threshold = speed_limit * 0.85
    else:
        threshold = 25.0
        print(f"  [speedLimit 无效, 使用默认阈值 {threshold} m/s]")
    ramp_ll = set(ll_avg[ll_avg < threshold].index)
    main_ll = set(ll_avg[ll_avg >= threshold].index)
    main_speed = float(ll_avg[ll_avg >= threshold].mean()) if main_ll else speed_limit
    return ramp_ll, main_ll, main_speed


def _get_starting_edge_and_lane(net, x, y):
    """查找 (x, y) 最近的 SUMO edge/lane，返回 (edge_id, lane_index, distance)。"""
    min_dist = float("inf")
    best_edge = ""
    best_idx = 0
    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        for lane in edge.getLanes():
            shape = lane.getShape()
            for i in range(len(shape) - 1):
                px1, py1 = shape[i]
                px2, py2 = shape[i + 1]
                dx, dy = px2 - px1, py2 - py1
                l2 = dx * dx + dy * dy
                if l2 == 0:
                    d = math.hypot(x - px1, y - py1)
                else:
                    t = max(0.0, min(1.0, ((x - px1) * dx + (y - py1) * dy) / l2))
                    d = math.hypot(x - (px1 + t * dx), y - (py1 + t * dy))
                if d < min_dist:
                    min_dist = d
                    best_edge = edge.getID()
                    best_idx = lane.getIndex()
    return best_edge, best_idx, min_dist


# Location-specific starting edge filter.
# Format: {loc_id: {edge_id: min_lane_to_filter}}
# lane >= min_lane_to_filter 的车辆会被排除。
MAIN_ROAD_FILTER_EDGES = {
    0: {
        "-9#0": 0,   # 主路: 全部排除
        "-9#1": 0,
        "-9#2": 0,
        "-9#3": 0,
        "-9#4": 1,   # 加速车道: lane >= 1 排除
    },
}


def find_merge_tracks(rec_id, ramp_ll, main_ll, sumo_net=None, loc_id=None):
    """找出 recording 中所有发生 ramp→main 转换的车辆。

    Location 0 特殊过滤: 主路 -9#4 上只有 _0 可以，_1/_2 排除。
    其他 Location 不过滤。
    """
    csv_path = os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path, usecols=[
        "trackId", "frame", "laneletId", "heading",
        "lonVelocity", "lonAcceleration", "latAcceleration",
        "xCenter", "yCenter", "xVelocity", "yVelocity",
        "latVelocity",
    ])

    filter_edges = MAIN_ROAD_FILTER_EDGES.get(loc_id, set()) if loc_id is not None else set()

    merges = []
    for tid, veh in df.groupby("trackId"):
        if len(veh) < 20:
            continue
        veh = veh.sort_values("frame")
        ll_seq = veh["laneletId"].values
        merge_frame_idx = None
        for i in range(1, len(ll_seq)):
            if ll_seq[i - 1] in ramp_ll and ll_seq[i] in main_ll:
                merge_frame_idx = i
                break
        if merge_frame_idx is None:
            continue

        # 最短段过滤: 匝道段和汇入后段各至少 MIN_SEG 帧
        MIN_SEG = 20  # 0.8s @ 25fps
        n = len(veh)
        if merge_frame_idx < MIN_SEG or (n - merge_frame_idx) < MIN_SEG:
            print(f"    track {tid}: too short (ramp={merge_frame_idx}, post={n - merge_frame_idx}), skip",
                  file=sys.stderr)
            continue

        # Location-specific: 过滤主路内侧车道
        if sumo_net is not None and filter_edges:
            start_row = veh.iloc[0]
            edge_id, lane_idx, dist = _get_starting_edge_and_lane(
                sumo_net, float(start_row["xCenter"]), float(start_row["yCenter"]),
            )
            if edge_id in filter_edges and lane_idx >= filter_edges[edge_id]:
                print(f"    track {tid}: starts in {edge_id}_{lane_idx}, skip", file=sys.stderr)
                continue

        frames = veh["frame"].values
        merges.append({
            "track_id": int(tid),
            "f0": int(frames[0]),
            "f1": int(frames[-1]),
            "merge_frame_idx": merge_frame_idx,
        })
    return merges


def extract_actions_and_rewards(ego_df, ramp_ll, main_ll, main_speed, dt=1.0 / 25.0):
    """从 CSV 数据计算 action 和 heuristic reward。

    Returns:
        actions: (T, 2) [steering, throttle]
        rewards: (T,)
        dones: (T,)
        lanelet_ids: (T,)
        lon_velocities: (T,)
        positions: (T, 2) [x, y] 原始坐标
    """
    T = len(ego_df)
    actions = np.zeros((T, 2), dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    dones = np.zeros(T, dtype=bool)
    ll_ids = ego_df["laneletId"].values.astype(np.int32)
    lon_vel = ego_df["lonVelocity"].values.astype(np.float32)
    lat_vel = ego_df["latVelocity"].values.astype(np.float32)
    headings = ego_df["heading"].values.astype(np.float32)
    lon_accel = ego_df["lonAcceleration"].values.astype(np.float32)
    lat_accel = ego_df["latAcceleration"].values.astype(np.float32)
    positions = ego_df[["xCenter", "yCenter"]].values.astype(np.float32)

    merged = False
    merge_step = T  # 默认最后一步

    for i in range(T):
        # ── 检测是否已经汇入 ──
        if i > 0 and not merged:
            if ll_ids[i - 1] in ramp_ll and ll_ids[i] in main_ll:
                merged = True
                merge_step = i

        # ── Action: steering, throttle ──
        if i < T - 1:
            # steering = heading 变化率 / dt, 归一化到 [-1, 1]
            d_heading = math.radians(headings[i + 1] - headings[i])
            # 处理 0-360 跨越
            if d_heading > math.pi:
                d_heading -= 2 * math.pi
            elif d_heading < -math.pi:
                d_heading += 2 * math.pi
            actions[i, 0] = np.clip(d_heading / dt / 3.0, -1.0, 1.0)  # 归一化

            # throttle = 速度变化率 / dt, 归一化到 [-1, 1]
            d_speed = lon_vel[i + 1] - lon_vel[i]
            actions[i, 1] = np.clip(d_speed / dt / 5.0, -1.0, 1.0)  # 归一化

        # ── Heuristic reward (v2) ──
        r = 0.0

        # 1) 基础成本
        r += W_BASE_COST

        if not merged:
            # 2) 速度进度: 达到主路速度的比例
            r += W_SPEED_FRAC * min(lon_vel[i] / main_speed, 1.0)
            # 3) 加速奖励 / 减速惩罚
            if i > 0:
                dv = lon_vel[i] - lon_vel[i - 1]
                if dv > 0.05:
                    r += W_ACCEL
                elif dv < -0.05:
                    r += W_DECEL
        else:
            # 4) 汇入后: 保持主路速度
            speed_diff = abs(lon_vel[i] - main_speed) / main_speed
            r += W_SPEED_POST * (1.0 - min(speed_diff, 1.0))

        # 5) 汇入完成奖励
        if i == merge_step:
            r += W_PROGRESS

        # 6) 成功完成奖励
        if i == T - 1 and merged:
            r += W_SUCCESS
            dones[i] = True

        rewards[i] = r

    return actions, rewards, dones, ll_ids, lon_vel, positions


def build_scenario_dict(rec_id, track_id, map_features, off_x, off_y, max_vehicles=1000):
    """构建 MetaDrive ScenarioDescription 所需的 dict。"""
    tracks_csv = pd.read_csv(os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv"), low_memory=False)
    tracks_meta = pd.read_csv(os.path.join(DATA_DIR, f"{rec_id:02d}_tracksMeta.csv"))

    sdc_id = int(track_id)
    ego_sub = tracks_csv[tracks_csv["trackId"] == sdc_id].sort_values("frame")
    f0 = int(ego_sub["frame"].iloc[0])
    f1 = int(ego_sub["frame"].iloc[-1])
    t_len = f1 - f0 + 1

    window = tracks_csv[(tracks_csv["frame"] >= f0) & (tracks_csv["frame"] <= f1)]
    candidates = [int(x) for x in window["trackId"].unique() if int(x) != sdc_id]
    selected = [sdc_id] + candidates[: max_vehicles - 1]

    meta_by_id = tracks_meta.set_index("trackId")
    ts_arr = np.arange(t_len, dtype=np.float32) / 25.0
    tracks = {}

    # Vehicle class → height mapping (matches MetaDrive vehicle types)
    _CLASS_HEIGHT = {"car": 1.5, "van": 1.85, "truck": 2.8, "motorcycle": 1.37}

    for tid in selected:
        try:
            mrow = meta_by_id.loc[tid]
            if isinstance(mrow, pd.DataFrame):
                mrow = mrow.iloc[0]
            default_len = float(mrow["length"])
            default_wid = float(mrow["width"])
            default_height = _CLASS_HEIGHT.get(str(mrow.get("class", "car")), 1.5)
        except (KeyError, TypeError):
            default_len, default_wid, default_height = 4.5, 2.0, 1.5

        sub = window[window["trackId"] == tid].sort_values("frame")
        if len(sub) == 0:
            continue

        pos = np.zeros((t_len, 3), dtype=np.float32)
        heading = np.zeros(t_len, dtype=np.float32)
        vel = np.zeros((t_len, 2), dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)

        for _, r in sub.iterrows():
            fi = int(r["frame"]) - f0
            if 0 <= fi < t_len:
                pos[fi, 0] = float(r["xCenter"]) + off_x
                pos[fi, 1] = float(r["yCenter"]) + off_y
                heading[fi] = math.radians(float(r["heading"]))
                vel[fi, 0] = float(r["xVelocity"])
                vel[fi, 1] = float(r["yVelocity"])
                valid[fi] = True

        if not valid.any():
            continue

        tracks[str(tid)] = {
            "type": MT.VEHICLE,
            "state": {
                "position": pos, "velocity": vel, "heading": heading, "valid": valid,
                "length": np.full(t_len, default_len, dtype=np.float32),
                "width": np.full(t_len, default_wid, dtype=np.float32),
                "height": np.full(t_len, default_height, dtype=np.float32),
            },
            "metadata": {"type": MT.VEHICLE, "object_id": str(tid), "dataset": "exiD"},
        }

    return {
        "id": f"exid-{rec_id:02d}-track{sdc_id}",
        "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": MT.COORDINATE_METADRIVE,
            "ts": ts_arr,
            "sdc_id": str(sdc_id),
            "scenario_id": f"exid_{rec_id:02d}_{sdc_id}",
            "dataset": "exiD",
            "source_file": f"recording_{rec_id:02d}",
            "ego_vehicle_class": "car",
            "frame_range": (f0, f1),
            "location_id": -1,  # caller sets this
        },
        "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": map_features,
    }, t_len


def collect_one_trajectory(env, rgb_cam, engine_origin, ego, t_len, capture_every=1):
    """在 MetaDrive 中回放一条轨迹，收集 BEV 图像。

    Returns:
        bev_images: list of (H, W, 3) uint8 arrays
    """
    bev_images = []

    for i in range(t_len):
        env.step([0.0, 0.0])

        # Fix z-height every step to prevent wheels floating/sinking
        if ego is not None:
            p = ego.position
            ego.set_position(
                [float(p[0]), float(p[1])],
                height=float(ego.HEIGHT) / 2,
            )

        if i % capture_every == 0:
            ego_pos = ego.position if ego else [0, 0, 0]
            ego_hpr = ego.origin.getHpr()

            bev_pos = (ego_pos[0], ego_pos[1], BEV_HEIGHT)
            bev_hpr_val = (ego_hpr.getX(), BEV_PITCH, 0)

            bev_img = rgb_cam.perceive(
                to_float=False,
                new_parent_node=engine_origin,
                position=bev_pos,
                hpr=bev_hpr_val,
            )
            if bev_img is not None and bev_img.ndim == 3 and bev_img.shape[2] == 3:
                bev_images.append(bev_img)

    return bev_images


def process_recording(rec_id, ramp_ll, main_ll, main_speed, loc_id,
                      map_features, off_x, off_y, dry_run=False,
                      specific_tid=None, out_dir=OUT_DIR, capture_every=1,
                      sumo_net=None):
    """处理一个 recording 的所有汇入轨迹。"""
    merges = find_merge_tracks(rec_id, ramp_ll, main_ll, sumo_net, loc_id)
    if not merges:
        print(f"  rec {rec_id:02d}: 0 merges, skip")
        return 0

    if specific_tid is not None:
        merges = [m for m in merges if m["track_id"] == specific_tid]
        if not merges:
            print(f"  rec {rec_id:02d}: track {specific_tid} not a merge vehicle")
            return 0

    print(f"  rec {rec_id:02d}: {len(merges)} merges")

    env = None
    rgb_cam = None
    engine_origin = None

    if not dry_run:
        from metadrive.scenario.scenario_description import ScenarioDescription
        from metadrive.envs.scenario_env import ScenarioOnlineEnv
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy
        from metadrive.component.sensors.rgb_camera import RGBCamera

    saved = 0
    for merge_info in merges:
        tid = merge_info["track_id"]
        out_path = os.path.join(out_dir, f"rec{rec_id:02d}", f"track{tid}.npz")
        if os.path.exists(out_path):
            saved += 1
            continue  # 跳过已处理

        # 加载 ego 的 CSV 数据
        csv_path = os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv")
        df = pd.read_csv(csv_path, low_memory=False)
        ego_df = df[df["trackId"] == tid].sort_values("frame")
        if len(ego_df) < 20:
            continue

        # 计算 actions 和 rewards
        actions, rewards, dones, ll_ids, lon_vel, positions = extract_actions_and_rewards(
            ego_df, ramp_ll, main_ll, main_speed,
        )

        if dry_run:
            saved += 1
            continue

        # 在 MetaDrive 中回放并收集 BEV 图像（每条轨迹新建 env，避免复用问题）
        try:
            scenario_dict, t_len = build_scenario_dict(
                rec_id, tid, map_features, off_x, off_y,
            )
            scenario_dict["metadata"]["location_id"] = loc_id
            scenario = ScenarioDescription(scenario_dict)

            # 每条轨迹都新建 env，确保地图和车辆正确加载
            if env is not None:
                env.close()
                del env
            env = ScenarioOnlineEnv(config=dict(
                use_render=False,
                image_observation=True,
                agent_policy=ReplayEgoCarPolicy,
                horizon=t_len + 50,
                set_static=True,
                camera_smooth=False,
                vehicle_config=dict(
                    no_wheel_friction=True,
                    show_navi_mark=False,
                    image_source="rgb_camera",
                ),
                sensors={"rgb_camera": (RGBCamera, BEV_W, BEV_H)},
                norm_pixel=False,
                height_scale=0.01,
            ))

            env.set_scenario(scenario)
            env.reset()

            ego = env.agent
            if ego is not None:
                p = ego.position
                ego.set_position(
                    [float(p[0]), float(p[1])],
                    height=float(ego.HEIGHT) / 2,
                )

            rgb_cam = env.engine.sensors.get("rgb_camera")
            engine_origin = env.engine.origin

            # 回放完整轨迹
            bev_images = collect_one_trajectory(env, rgb_cam, engine_origin, ego, t_len,
                                                capture_every=capture_every)

            # 对齐 actions/rewards 与 BEV 图像
            T_img = len(bev_images)
            if T_img == 0:
                continue

            # 按捕获频率对齐
            T_full = len(actions)
            indices = list(range(0, T_full, capture_every))
            # 确保最后一帧被包含
            if indices[-1] != T_full - 1:
                indices.append(T_full - 1)
            indices = indices[:T_img]
            actions_ds = actions[indices]
            rewards_ds = rewards[indices]
            dones_ds = dones[indices]
            ll_ids_ds = ll_ids[indices]
            lon_vel_ds = lon_vel[indices]
            positions_ds = positions[indices]

            # 堆叠 BEV 图像
            bev_stack = np.stack(bev_images, axis=0)  # (T, H, W, 3)

            # 保存 npz
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(
                out_path,
                bev_images=bev_stack,
                actions=actions_ds,
                rewards=rewards_ds,
                dones=dones_ds,
                lanelet_ids=ll_ids_ds,
                lon_velocities=lon_vel_ds,
                positions=positions_ds,
                recording_id=rec_id,
                track_id=tid,
                location_id=loc_id,
                merge_frame_idx=merge_info["merge_frame_idx"],
            )

            # 保存视频（每个 recording 只存前 3 个，用于抽查验证）
            import glob as _glob
            _existing_videos = len(_glob.glob(os.path.join(os.path.dirname(out_path), "*_bev.mp4")))
            if _existing_videos < 3:
                import cv2
                vid_path = out_path.replace(".npz", "_bev.mp4")
                h, w = bev_stack.shape[1], bev_stack.shape[2]
                for codec in ["avc1", "mp4v", "XVID"]:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(vid_path, fourcc, FPS, (w, h))
                    if writer.isOpened():
                        break
                    writer.release()
                else:
                    writer = None
                if writer is not None:
                    for frame in bev_stack:
                        writer.write(frame)
                    writer.release()
                    print(f"    track {tid}: {T_img} BEV images → .npz + .mp4 saved")
                else:
                    print(f"    track {tid}: {T_img} BEV images → .npz saved (video codec failed)")
            else:
                print(f"    track {tid}: {T_img} BEV images → .npz saved")

            saved += 1

        except Exception as e:
            print(f"    track {tid}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            continue

    if env is not None:
        env.close()

    return saved


def main():
    parser = argparse.ArgumentParser(description="Collect DreamerV3 training data from exiD merge scenarios")
    parser.add_argument("--recording", type=int, help="Process a single recording")
    parser.add_argument("--track-id", type=int, help="Process a specific track within the recording")
    parser.add_argument("--location", type=int, help="Process all recordings in a location")
    parser.add_argument("--dry-run", action="store_true", help="Only compute stats, no MetaDrive")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--capture-every", type=int, default=1,
                        help="Capture BEV every N frames (default=1, no downsampling)")
    args = parser.parse_args()

    out_dir = args.out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 确定要处理的 recording 列表
    rec_ids = []
    loc_id = None

    if args.recording is not None:
        rec_ids = [args.recording]
        meta = pd.read_csv(os.path.join(DATA_DIR, f"{args.recording:02d}_recordingMeta.csv"))
        loc_id = int(meta["locationId"].iloc[0])
    elif args.location is not None:
        loc_id = args.location
        for f in sorted(os.listdir(DATA_DIR)):
            if f.endswith("_recordingMeta.csv"):
                meta = pd.read_csv(os.path.join(DATA_DIR, f))
                if int(meta["locationId"].iloc[0]) == loc_id:
                    rec_ids.append(int(meta["recordingId"].iloc[0]))
    else:
        print("Specify --recording ID or --location ID")
        return

    print(f"Location {loc_id} ({LOC_NAMES[loc_id]})")
    print(f"Recordings: {len(rec_ids)}")

    # 加载地图
    speed_limit = pd.read_csv(
        os.path.join(DATA_DIR, f"{rec_ids[0]:02d}_recordingMeta.csv")
    )["speedLimit"].iloc[0]

    print(f"Loading map for location {loc_id}...")
    map_features, off_x, off_y = build_map_features(loc_id)
    print(f"  Map features: {len(map_features)}, offset=({off_x:.1f}, {off_y:.1f})")

    # 加载 SUMO 网络（用于车道位置过滤）
    net_xml = get_map_file(loc_id)
    sumo_net = sumolib.net.readNet(net_xml, withInternal=False)
    print(f"  SUMO network loaded: {len(list(sumo_net.getEdges()))} edges")

    # 分类 lanelet
    print("Classifying lanelets...")
    ramp_ll, main_ll, main_speed = classify_lanelets(rec_ids, speed_limit)
    print(f"  Ramp lanelets: {len(ramp_ll)}, Main lanelets: {len(main_ll)}")
    print(f"  Main road avg speed: {main_speed:.1f} m/s")

    # 处理每个 recording
    t0 = time.time()
    total_saved = 0
    for rec_id in sorted(rec_ids):
        n = process_recording(
            rec_id, ramp_ll, main_ll, main_speed, loc_id,
            map_features, off_x, off_y,
            dry_run=args.dry_run,
            specific_tid=args.track_id,
            out_dir=out_dir,
            capture_every=args.capture_every,
            sumo_net=sumo_net,
        )
        total_saved += n

    elapsed = time.time() - t0
    print(f"\nDone: {total_saved} trajectories saved in {elapsed:.1f}s")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
