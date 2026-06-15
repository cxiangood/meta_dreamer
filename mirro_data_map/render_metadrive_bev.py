"""
用 MetaDrive 渲染全部 7 个 Location 的 BEV 俯视图。
"""
import os, sys, math
import cv2
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

SUMO_HOME = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo"
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
os.environ["SUMO_HOME"] = SUMO_HOME
os.environ["METADRIVE_HEADLESS"] = "1"

import sumolib
from metadrive.type import MetaDriveType as MT
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.utils.draw_top_down_map import draw_top_down_map

MAP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1/data"
OUT_DIR = os.path.join(MAP_DIR, "exid_merge_preview")
os.makedirs(OUT_DIR, exist_ok=True)

LOC_MAPS = {
    0: "exid_loc0_orig.net.xml", 1: "exid_loc1_orig.net.xml",
    2: "exid_loc2_orig.net.xml", 3: "exid_loc3_orig.net.xml",
    4: "exid_loc4_orig.net.xml", 5: "exid_loc5_orig.net.xml",
    6: "exid_loc6_orig.net.xml",
}
LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}

# 每个 location 取第一条 recording 的第一辆车作为 dummy ego
# 仅用于让 ScenarioOnlineEnv 加载地图
DUMMY_REC = {0: 0, 1: 19, 2: 39, 3: 53, 4: 61, 5: 73, 6: 78}


def build_sumo_map_features(location_id):
    net_xml = os.path.join(MAP_DIR, LOC_MAPS[location_id])
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
                pts = [pl[0]]
                d = pl[-1] - pl[0]
                length = float(np.hypot(d[0], d[1]))
                n = max(int(length / SPACING), 4)
                for i in range(1, n):
                    pts.append(pl[0] + (i / n) * d)
                pts.append(pl[-1])
                v["polyline"] = np.array(pts, dtype=np.float32)

    return features, off_x, off_y


for loc_id in range(7):
    print(f"\n{'='*60}")
    print(f"Location {loc_id} ({LOC_NAMES[loc_id]})")
    print(f"{'='*60}")

    # 加载地图
    map_features, off_x, off_y = build_sumo_map_features(loc_id)
    print(f"  地图 features: {len(map_features)}, offset=({off_x:.1f}, {off_y:.1f})")

    # 构造 dummy scenario（仅 1 帧，让 env 加载地图）
    rid = DUMMY_REC[loc_id]
    tracks_csv = pd.read_csv(os.path.join(DATA_DIR, f"{rid:02d}_tracks.csv"), low_memory=False)
    tracks_meta = pd.read_csv(os.path.join(DATA_DIR, f"{rid:02d}_tracksMeta.csv"))

    # 取第一个 track 作为 ego
    ego_tid = int(tracks_csv["trackId"].iloc[0])
    ego_data = tracks_csv[tracks_csv["trackId"] == ego_tid].sort_values("frame")
    ego_meta = tracks_meta[tracks_meta["trackId"] == ego_tid]

    f0 = int(ego_data["frame"].iloc[0])
    f1 = int(ego_data["frame"].iloc[-1])
    t_len = f1 - f0 + 1

    # 只保留 ego 车辆，简化 scenario
    default_len = float(ego_meta["length"].iloc[0]) if len(ego_meta) > 0 else 4.5
    default_wid = float(ego_meta["width"].iloc[0]) if len(ego_meta) > 0 else 2.0

    pos = np.zeros((t_len, 3), dtype=np.float32)
    heading = np.zeros(t_len, dtype=np.float32)
    vel = np.zeros((t_len, 2), dtype=np.float32)
    valid = np.zeros(t_len, dtype=bool)

    for _, r in ego_data.iterrows():
        fi = int(r["frame"]) - f0
        if 0 <= fi < t_len:
            pos[fi, 0] = float(r["xCenter"]) + off_x
            pos[fi, 1] = float(r["yCenter"]) + off_y
            heading[fi] = math.radians(float(r["heading"]))
            vel[fi, 0] = float(r["xVelocity"])
            vel[fi, 1] = float(r["yVelocity"])
            valid[fi] = True

    tracks = {
        str(ego_tid): {
            "type": MT.VEHICLE,
            "state": {
                "position": pos, "velocity": vel, "heading": heading,
                "valid": valid,
                "length": np.full(t_len, default_len, dtype=np.float32),
                "width": np.full(t_len, default_wid, dtype=np.float32),
                "height": np.full(t_len, 1.5, dtype=np.float32),
            },
            "metadata": {"type": MT.VEHICLE, "object_id": str(ego_tid), "dataset": "exiD"},
        }
    }

    scenario_dict = {
        "id": f"exid-loc{loc_id}-map",
        "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": MT.COORDINATE_METADRIVE,
            "ts": np.arange(t_len, dtype=np.float32) / 25.0,
            "sdc_id": str(ego_tid),
            "scenario_id": f"loc{loc_id}_map",
            "dataset": "exiD",
            "source_file": f"recording_{rid:02d}",
            "ego_vehicle_class": "car",
            "frame_range": (f0, f1),
            "location_id": loc_id,
        },
        "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": map_features,
    }

    SD = ScenarioDescription(scenario_dict)

    # 创建环境并渲染
    env = ScenarioOnlineEnv(config=dict(
        use_render=False,
        agent_policy=ReplayEgoCarPolicy,
        horizon=t_len + 10,
        set_static=True,
        camera_smooth=False,
        vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False),
        height_scale=0.01,
    ))

    env.set_scenario(SD)
    env.reset()

    # 修正 ego z 高度
    ego = env.agent
    if ego is not None:
        p = ego.position
        ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)

    # 渲染 BEV 地图
    map_img = draw_top_down_map(env.current_map, resolution=(2048, 1024), semantic_map=True)
    out_path = os.path.join(OUT_DIR, f"loc{loc_id}_metadrive_bev.png")
    cv2.imwrite(out_path, map_img)
    print(f"  ✓ MetaDrive BEV → {out_path} ({map_img.shape})")

    env.close()
    del env

print(f"\n{'='*60}")
print("全部完成!")
print(f"{'='*60}")
