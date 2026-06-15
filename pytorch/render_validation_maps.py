"""
Quick render test: verify exiD loc1/loc3 SUMO maps load correctly in MetaDrive.

Renders BEV images for the first few tracks and saves them for visual inspection.
"""

import argparse
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
os.environ.setdefault("METADRIVE_HEADLESS", "1")

import sumolib
from metadrive.type import MetaDriveType as MT
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.scenario_env import ScenarioOnlineEnv

LOC_NAMES = {
    1: "cologne_fortiib", 3: "bergheim_roemer",
}
BEV_W, BEV_H = 400, 300


def get_map_file(loc_id):
    map_dir = os.path.join(os.path.dirname(__file__), "../mirro_data_map")
    orig = os.path.join(map_dir, f"exid_loc{loc_id}_orig.net.xml")
    if os.path.exists(orig):
        return orig
    return os.path.join(map_dir, f"exid_loc{loc_id}.net.xml")


def build_map_features(loc_id):
    net_xml = get_map_file(loc_id)
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
                d = pl[-1] - pl[0]
                length = float(np.hypot(d[0], d[1]))
                n = max(int(length / SPACING), 4)
                pts = [pl[0]] + [pl[0] + (i / n) * d for i in range(1, n)] + [pl[-1]]
                v["polyline"] = np.array(pts, dtype=np.float32)
    return features, off_x, off_y


def build_scenario_dict(rec_id, track_id, loc_id, map_features, off_x, off_y, data_dir, max_vehicles=100):
    import pandas as pd

    tracks_csv = pd.read_csv(os.path.join(data_dir, f"{rec_id:02d}_tracks.csv"), low_memory=False)
    tracks_meta = pd.read_csv(os.path.join(data_dir, f"{rec_id:02d}_tracksMeta.csv"))
    rec_meta = pd.read_csv(os.path.join(data_dir, f"{rec_id:02d}_recordingMeta.csv"))

    sdc_id = int(track_id)
    ego_sub = tracks_csv[tracks_csv["trackId"] == sdc_id].sort_values("frame")
    f0 = int(ego_sub["frame"].iloc[0])
    f1 = int(ego_sub["frame"].iloc[-1])
    t_len = f1 - f0 + 1

    window = tracks_csv[(tracks_csv["frame"] >= f0) & (tracks_csv["frame"] <= f1)]
    candidates = [int(x) for x in window["trackId"].unique() if int(x) != sdc_id]
    selected = [sdc_id] + candidates[:max_vehicles - 1]

    meta_by_id = tracks_meta.set_index("trackId")
    ts_arr = np.arange(t_len, dtype=np.float32) / 25.0
    _CLASS_HEIGHT = {"car": 1.5, "van": 1.85, "truck": 2.8, "motorcycle": 1.37}
    tracks = {}

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
            "location_id": loc_id,
        },
        "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": map_features,
    }, t_len


def render_check(loc_id, rec_id, track_id, data_dir, out_path_fn=None):
    """Render a single trajectory and save BEV snapshots."""
    import pandas as pd
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy

    features, off_x, off_y = build_map_features(loc_id)
    scenario_dict, t_len = build_scenario_dict(
        rec_id, track_id, loc_id, features, off_x, off_y, data_dir)
    scenario = ScenarioDescription(scenario_dict)

    env = ScenarioOnlineEnv(config=dict(
        use_render=False,
        image_observation=True,
        agent_policy=ReplayEgoCarPolicy,
        horizon=t_len + 50,
        store_map=False,
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

    # Fix ego z-height
    ego = env.agent
    if ego is not None:
        p = ego.position
        ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)

    rgb_cam = env.engine.sensors.get("rgb_camera")
    engine_origin = env.engine.origin

    # Render snapshots at start, merge, end
    snap_frames = [0, t_len // 2, t_len - 1]
    images = []

    for i in range(t_len):
        env.step([0.0, 0.0])

        if ego is not None:
            p = ego.position
            ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)

        if i in snap_frames:
            ego_pos = ego.position if ego else [0, 0, 0]
            ego_hpr = ego.origin.getHpr()
            bev = rgb_cam.perceive(
                to_float=False,
                new_parent_node=engine_origin,
                position=(float(ego_pos[0]), float(ego_pos[1]), 50.0),
                hpr=(ego_hpr.getX(), -89, 0),
            )
            if bev is not None and bev.ndim == 3:
                H, W = bev.shape[:2]
                size = min(H, W)
                dh = (H - size) // 2
                dw = (W - size) // 2
                bev = bev[dh:dh + size, dw:dw + size]
                images.append((i, bev))

    env.close()
    return images


def main():
    parser = argparse.ArgumentParser(description="Verify exiD validation maps render correctly")
    parser.add_argument("--data-dir", required=True, help="Path to exiD data/ directory")
    parser.add_argument("--loc", type=int, nargs="+", default=[1, 3],
                        help="Location IDs to check")
    parser.add_argument("--n", type=int, default=2, help="Tracks to check per location")
    parser.add_argument("--out", default="./map_check", help="Output dir for PNGs")
    args = parser.parse_args()

    import json
    import cv2

    # Cache is in the meta_repo root, not relative to this script
    meta_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_path = os.path.join(meta_root, "mirro_data_map", "exid_merge_cache_selected.json")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(meta_root, "mirro_data_map", "exid_merge_cache.json")
    with open(cache_path) as f:
        cache = json.load(f)

    os.makedirs(args.out, exist_ok=True)

    for loc_id in args.loc:
        loc_str = str(loc_id)
        items = cache.get(loc_str, [])
        if not items:
            print(f"Location {loc_id}: no trajectories in cache!")
            continue

        print(f"\nLocation {loc_id} ({LOC_NAMES[loc_id]}): {len(items)} trajectories")

        for j in range(min(args.n, len(items))):
            traj = items[j]
            rec_id, tid = traj["rid"], traj["tid"]
            print(f"  rec{rec_id} track{tid}...")

            try:
                images = render_check(loc_id, rec_id, tid, args.data_dir)
                for frame_idx, img in images:
                    out_path = os.path.join(args.out, f"loc{loc_id}_rec{rec_id}_t{tid}_f{frame_idx}.png")
                    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    print(f"    Saved frame {frame_idx} -> {out_path}")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nDone. Images saved to {args.out}/")


if __name__ == "__main__":
    main()
