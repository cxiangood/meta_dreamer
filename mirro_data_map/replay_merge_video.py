"""
在 MetaDrive 中回放 exiD 匝道汇入场景，BEV 相机视角跟随 ego，保存视频。

相机设置在 ego 正上方，朝下看（pitch=-90），用 RGBCamera 渲染 3D BEV 视角。

用法:
    python3 mirro_data_map/replay_merge_video.py --recording 0 --track-id 59
    python3 mirro_data_map/replay_merge_video.py --recording 0 --track-id 59 --chase
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SUMO_HOME = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo"
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
if SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)
os.environ["SUMO_HOME"] = SUMO_HOME
os.environ["METADRIVE_HEADLESS"] = "1"

import sumolib
from metadrive.type import MetaDriveType as MT
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.utils.draw_top_down_map import draw_top_down_map

DATASET_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1"
DATA_DIR = os.path.join(DATASET_DIR, "data")
MAP_DIR = os.path.dirname(os.path.abspath(__file__))

LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}

# BEV 相机参数
BEV_HEIGHT = 50.0    # 相机高度 (米)
BEV_HEADING = 0.0    # 跟随 ego 朝向
BEV_PITCH = -89.0    # 几乎正下方 (Panda3D HPR: heading, pitch, roll)
BEV_W, BEV_H = 800, 600
CHASE_W, CHASE_H = 800, 600
FPS = 10


def build_sumo_map_features(location_id):
    net_xml = os.path.join(MAP_DIR, f"exid_loc{location_id}_orig.net.xml")
    if not os.path.exists(net_xml):
        net_xml = os.path.join(MAP_DIR, f"exid_loc{location_id}.net.xml")
    raw_net = sumolib.net.readNet(net_xml, withInternal=True)
    xmin, ymin, xmax, ymax = raw_net.getBoundary()
    off_x = -(xmax + xmin) / 2
    off_y = -(ymax + ymin) / 2
    graph = RoadLaneJunctionGraph(net_xml)
    features = extract_map_features(graph)
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


def build_tracks(recording_id, track_id, off_x, off_y, max_vehicles=50):
    tracks_csv = pd.read_csv(os.path.join(DATA_DIR, f"{recording_id:02d}_tracks.csv"), low_memory=False)
    tracks_meta = pd.read_csv(os.path.join(DATA_DIR, f"{recording_id:02d}_tracksMeta.csv"))
    t = tracks_csv[tracks_csv['recordingId'] == recording_id]
    tm = tracks_meta[tracks_meta['recordingId'] == recording_id]
    sdc_id = int(track_id)
    ego_sub = t[t['trackId'] == sdc_id].sort_values('frame')
    f0 = int(ego_sub['frame'].iloc[0])
    f1 = int(ego_sub['frame'].iloc[-1])
    t_len = f1 - f0 + 1
    window = t[(t['frame'] >= f0) & (t['frame'] <= f1)].copy()
    candidates = [int(x) for x in window['trackId'].unique() if int(x) != sdc_id]
    selected = [sdc_id] + candidates[:max_vehicles - 1]
    meta_by_id = tm.set_index('trackId')
    ts_arr = np.arange(t_len, dtype=np.float32) / 25.0
    tracks = {}
    for tid in selected:
        try:
            mrow = meta_by_id.loc[tid]
            if isinstance(mrow, pd.DataFrame): mrow = mrow.iloc[0]
            default_len = float(mrow['length']); default_wid = float(mrow['width'])
        except (KeyError, TypeError):
            default_len, default_wid = 4.5, 2.0
        sub = window[window['trackId'] == tid].sort_values('frame')
        if len(sub) == 0: continue
        pos = np.zeros((t_len, 3), dtype=np.float32)
        heading = np.zeros(t_len, dtype=np.float32)
        vel = np.zeros((t_len, 2), dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)
        for _, r in sub.iterrows():
            fi = int(r['frame']) - f0
            if 0 <= fi < t_len:
                pos[fi, 0] = float(r['xCenter']) + off_x
                pos[fi, 1] = float(r['yCenter']) + off_y
                heading[fi] = math.radians(float(r['heading']))
                vel[fi, 0] = float(r['xVelocity']); vel[fi, 1] = float(r['yVelocity'])
                valid[fi] = True
        if not valid.any(): continue
        tracks[str(tid)] = {
            "type": MT.VEHICLE,
            "state": {"position": pos, "velocity": vel, "heading": heading, "valid": valid,
                      "length": np.full(t_len, default_len, dtype=np.float32),
                      "width": np.full(t_len, default_wid, dtype=np.float32),
                      "height": np.full(t_len, 1.5, dtype=np.float32)},
            "metadata": {"type": MT.VEHICLE, "object_id": str(tid), "dataset": "exiD"},
        }
    return tracks, sdc_id, f0, f1, t_len, ts_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=int, default=0)
    parser.add_argument("--track-id", type=int, default=59)
    parser.add_argument("--chase", action="store_true", help="Also save chase camera view")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bev-height", type=float, default=BEV_HEIGHT)
    args = parser.parse_args()

    rid = args.recording
    tid = args.track_id
    recording_meta = pd.read_csv(os.path.join(DATA_DIR, f"{rid:02d}_recordingMeta.csv"))
    loc_id = int(recording_meta['locationId'].iloc[0])
    print(f"Recording {rid}, Track {tid}, Location {loc_id} ({LOC_NAMES[loc_id]})")

    # Build scenario
    map_features, off_x, off_y = build_sumo_map_features(loc_id)
    tracks, sdc_id, f0, f1, t_len, ts_arr = build_tracks(rid, tid, off_x, off_y)
    print(f"  {t_len} frames, {len(tracks)} vehicles")

    scenario_dict = {
        "id": f"exid-{rid:02d}-track{sdc_id}", "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {
            "metadrive_processed": True, "coordinate": MT.COORDINATE_METADRIVE,
            "ts": ts_arr, "sdc_id": str(sdc_id), "scenario_id": f"exid_{rid:02d}_{sdc_id}",
            "dataset": "exiD", "source_file": f"recording_{rid:02d}",
            "ego_vehicle_class": "car", "frame_range": (f0, f1), "location_id": loc_id,
        },
        "tracks": tracks, "dynamic_map_states": {}, "map_features": map_features,
    }
    scenario = ScenarioDescription(scenario_dict)

    # Create env with RGBCamera for BEV
    env = ScenarioOnlineEnv(config=dict(
        use_render=False,
        image_observation=True,
        agent_policy=ReplayEgoCarPolicy,
        horizon=t_len + 50,
        set_static=True,
        camera_smooth=False,
        vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False, image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, BEV_W, BEV_H)},
        norm_pixel=False,
        height_scale=0.01,
    ))

    env.set_scenario(scenario)
    obs = env.reset()

    ego = env.agent
    if ego is not None:
        p = ego.position
        ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)

    # Get camera and engine references
    rgb_cam = env.engine.sensors.get("rgb_camera")
    engine_origin = env.engine.origin

    # Output dir
    out_dir = args.out_dir or os.path.join(MAP_DIR, "exid_merge_preview", f"rec{rid:02d}_track{tid}_video")
    os.makedirs(out_dir, exist_ok=True)

    # Static BEV map
    map_img = draw_top_down_map(env.current_map, resolution=(2048, 1024), semantic_map=True)
    cv2.imwrite(os.path.join(out_dir, "bev_map.png"), map_img)
    print(f"  BEV map saved")

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    bev_writer = cv2.VideoWriter(os.path.join(out_dir, "bev_follow.mp4"), fourcc, FPS, (BEV_W, BEV_H))
    chase_writer = None

    print(f"  Simulating {t_len} steps...")
    for i in range(t_len):
        obs, *_ = env.step([0.0, 0.0])

        # Get ego position and Panda3D heading directly
        ego_pos = ego.position if ego else [0, 0, 0]
        ego_hpr = ego.origin.getHpr()  # Vec3(H, P, R) in Panda3D degrees

        # BEV camera: directly above ego, looking down, same heading as vehicle
        if i % 2 == 0:
            bev_pos = (ego_pos[0], ego_pos[1], args.bev_height)
            bev_hpr = (ego_hpr.getX(), BEV_PITCH, 0)
            bev_img = rgb_cam.perceive(
                to_float=False,
                new_parent_node=engine_origin,
                position=bev_pos,
                hpr=bev_hpr,
            )
            if bev_img is not None:
                if bev_img.ndim == 3 and bev_img.shape[2] == 3:
                    bev_writer.write(bev_img)

        # Chase camera from obs
        if args.chase and i % 3 == 0:
            frame = None
            if isinstance(obs, dict) and "image" in obs:
                frame = obs["image"]
            if frame is not None:
                if frame.ndim == 4: frame = frame[..., -1]
                if frame.dtype in (np.float32, np.float64):
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                if chase_writer is None:
                    h, w = frame.shape[:2]
                    chase_writer = cv2.VideoWriter(os.path.join(out_dir, "chase_h264.mp4"), fourcc, FPS, (w, h))
                chase_writer.write(frame)

        if i % 100 == 0:
            print(f"    step {i}/{t_len}")

    bev_writer.release()
    if chase_writer:
        chase_writer.release()
    env.close()

    print(f"\n  ✓ Output: {out_dir}/")
    print(f"    bev_follow.mp4  (3D BEV 视角, h={args.bev_height}m)")
    if chase_writer:
        print(f"    chase_h264.mp4  (追车视角)")
    print(f"    bev_map.png     (静态地图)")


if __name__ == "__main__":
    main()
