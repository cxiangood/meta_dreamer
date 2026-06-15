"""
Replay exiD highway exit-ramp scenario.

Loads SUMO .net.xml (with internal edges for junction connections) for correct
road surface rendering, then replays vehicle trajectories from CSV data.

Usage:
    python3 mirro_data_map/replay_exid.py --recording 01 --track-id 45 --chase
    python3 mirro_data_map/replay_exid.py --recording 01 --track-id 45 --render
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

DATASET_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1"
DATA_DIR = os.path.join(DATASET_DIR, "data")
MAP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# locationId → net.xml mapping
LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}


def build_sumo_map_features(location_id, map_path=None):
    """Load map_features from SUMO .net.xml with internal edges.

    Uses the version with junction internal edges so that connecting
    segments are rendered correctly (no gaps between road sections).

    Returns (features, offset_x, offset_y).
    """
    os.environ["SUMO_HOME"] = SUMO_HOME
    import sumolib
    from metadrive.type import MetaDriveType as MT
    from metadrive.utils.sumo.map_utils import extract_map_features, RoadLaneJunctionGraph

    if map_path:
        net_xml = map_path
    else:
        # Prefer the _orig version (with internal edges), fall back to regular
        net_xml = os.path.join(MAP_DIR, f"exid_loc{location_id}_orig.net.xml")
        if not os.path.exists(net_xml):
            net_xml = os.path.join(MAP_DIR, f"exid_loc{location_id}.net.xml")
    print(f"Loading SUMO map: {net_xml}")

    # Compute the same centering offset that RoadLaneJunctionGraph applies
    raw_net = sumolib.net.readNet(
        net_xml, withInternal=True,
        withPedestrianConnections=True, withPrograms=True,
    )
    xmin, ymin, xmax, ymax = raw_net.getBoundary()
    offset_x = -(xmax + xmin) / 2
    offset_y = -(ymax + ymin) / 2

    graph = RoadLaneJunctionGraph(net_xml)
    features = extract_map_features(graph)

    # Resample broken lane lines for 3D rendering (need >= 4 points)
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

    n_lanes = sum(1 for v in features.values() if v.get("type") == MT.LANE_SURFACE_STREET)
    print(f"SUMO map: {len(features)} features ({n_lanes} lanes), "
          f"offset=({offset_x:.1f}, {offset_y:.1f})")
    return features, offset_x, offset_y


def build_exid_tracks(tracks_csv, tracks_meta_csv, recording_id,
                      offset_x=0.0, offset_y=0.0, ego_track_id=None,
                      max_vehicles=50):
    """Build MetaDrive-compatible tracks from exiD CSV data."""
    from metadrive.type import MetaDriveType as MT

    t = tracks_csv[tracks_csv['recordingId'] == recording_id]
    tm = tracks_meta_csv[tracks_meta_csv['recordingId'] == recording_id]

    if ego_track_id is not None:
        sdc_id = int(ego_track_id)
    else:
        # Default: first track with lane change
        lc_tracks = t[t['laneChange'] != 0]['trackId'].unique()
        sdc_id = int(lc_tracks[0]) if len(lc_tracks) > 0 else int(t['trackId'].iloc[0])

    ego_sub = t[t['trackId'] == sdc_id].sort_values('frame')
    f0 = int(ego_sub['frame'].iloc[0])
    f1 = int(ego_sub['frame'].iloc[-1])
    t_len = f1 - f0 + 1

    # Window: frames where ego is active
    window = t[(t['frame'] >= f0) & (t['frame'] <= f1)].copy()

    # Select vehicles in window
    candidates = [int(x) for x in window['trackId'].unique() if int(x) != sdc_id]
    selected = [sdc_id] + candidates[:max_vehicles - 1]
    meta_by_id = tm.set_index('trackId')

    ts_arr = np.arange(t_len, dtype=np.float32) / 25.0  # 25fps

    tracks = {}
    for tid in selected:
        try:
            vclass = str(meta_by_id.loc[tid, 'class'])
        except (KeyError, TypeError):
            vclass = "car"

        sub = window[window['trackId'] == tid].sort_values('frame')
        if len(sub) == 0:
            continue

        pos = np.zeros((t_len, 3), dtype=np.float32)
        heading = np.zeros(t_len, dtype=np.float32)
        vel = np.zeros((t_len, 2), dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)

        try:
            mrow = meta_by_id.loc[tid]
            if isinstance(mrow, pd.DataFrame):
                mrow = mrow.iloc[0]
            default_len = float(mrow['length'])
            default_wid = float(mrow['width'])
        except (KeyError, TypeError):
            default_len = 4.5
            default_wid = 2.0

        length = np.full(t_len, default_len, dtype=np.float32)
        width_arr = np.full(t_len, default_wid, dtype=np.float32)
        height_arr = np.ones(t_len, dtype=np.float32) * 1.5

        for _, r in sub.iterrows():
            fi = int(r['frame']) - f0
            if fi < 0 or fi >= t_len:
                continue
            pos[fi, 0] = float(r['xCenter']) + offset_x
            pos[fi, 1] = float(r['yCenter']) + offset_y
            pos[fi, 2] = 0.0

            # exiD heading: standard math (0=East, CCW, degrees) → radians
            heading[fi] = math.radians(float(r['heading']))

            # Velocity: already in global frame, same coord as position
            vel[fi, 0] = float(r['xVelocity'])
            vel[fi, 1] = float(r['yVelocity'])
            valid[fi] = True

        if not valid.any():
            continue

        tracks[str(tid)] = {
            "type": MT.VEHICLE,
            "state": {
                "position": pos, "velocity": vel, "heading": heading,
                "valid": valid, "length": length, "width": width_arr,
                "height": height_arr,
            },
            "metadata": {
                "type": MT.VEHICLE, "object_id": str(tid),
                "dataset": "exiD",
            },
        }

    ego_class = "car"
    try:
        mrow = meta_by_id.loc[sdc_id]
        if isinstance(mrow, pd.DataFrame):
            mrow = mrow.iloc[0]
        ego_class = str(mrow['class'])
    except (KeyError, TypeError):
        pass

    return tracks, sdc_id, f0, f1, ego_class, t_len, ts_arr


def build_scenario(recording_id, track_id=None, map_path=None):
    """Build ScenarioDescription from exiD data."""
    from metadrive.scenario.scenario_description import ScenarioDescription as SD
    from metadrive.type import MetaDriveType as MT

    rid = int(recording_id)
    tracks_csv = pd.read_csv(os.path.join(DATA_DIR, f"{rid:02d}_tracks.csv"), low_memory=False)
    tracks_meta = pd.read_csv(os.path.join(DATA_DIR, f"{rid:02d}_tracksMeta.csv"))
    recording_meta = pd.read_csv(os.path.join(DATA_DIR, f"{rid:02d}_recordingMeta.csv"))

    loc_id = int(recording_meta['locationId'].iloc[0])

    # Load SUMO map (with internal edges for junction connections)
    map_features, off_x, off_y = build_sumo_map_features(loc_id, map_path=map_path)
    tracks, sdc_id, f0, f1, ego_class, t_len, ts_arr = build_exid_tracks(
        tracks_csv, tracks_meta, rid,
        offset_x=off_x, offset_y=off_y,
        ego_track_id=track_id,
    )

    sdc_str = str(sdc_id)
    scenario_dict = {
        "id": f"exid-{rid:02d}-track{sdc_id}",
        "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": MT.COORDINATE_METADRIVE,
            "ts": ts_arr,
            "sdc_id": sdc_str,
            "scenario_id": f"exid_{rid:02d}_{sdc_id}",
            "dataset": "exiD",
            "source_file": f"recording_{rid:02d}",
            "ego_vehicle_class": ego_class,
            "frame_range": (f0, f1),
            "location_id": loc_id,
        },
        "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": map_features,
    }
    SD.sanity_check(scenario_dict, check_self_type=True)
    return SD(scenario_dict)


def main():
    parser = argparse.ArgumentParser(description="Replay exiD scenario with SUMO map")
    parser.add_argument("--recording", type=int, default=1, help="Recording ID (e.g. 1, 30, 60)")
    parser.add_argument("--track-id", type=int, default=None, help="Ego vehicle trackId")
    parser.add_argument("--render", action="store_true", help="Open 3D viewer")
    parser.add_argument("--chase", action="store_true", help="Capture chase camera video")
    parser.add_argument("--bev", action="store_true", help="Save BEV screenshots")
    parser.add_argument("--bev-every", type=int, default=50)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--map", default=None, help="Custom SUMO .net.xml path (overrides location-based lookup)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    scenario = build_scenario(args.recording, track_id=args.track_id, map_path=args.map)
    t_len = int(scenario["length"])
    n_steps = args.steps or t_len
    print(f"Scenario: {scenario['id']}, {t_len} frames ({t_len/25:.1f}s), "
          f"{len(scenario['tracks'])} vehicles")

    rid = args.recording
    tid = scenario["metadata"]["sdc_id"]
    out_dir = args.out_dir or os.path.join(OUTPUT_DIR, f"exid_replay/{rid:02d}_track{tid}")
    os.makedirs(out_dir, exist_ok=True)

    render = args.render
    if not render:
        os.environ["METADRIVE_HEADLESS"] = "1"

    if args.chase:
        from metadrive.component.sensors.rgb_camera import RGBCamera
        from metadrive.envs.scenario_env import ScenarioOnlineEnv
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy

        env = ScenarioOnlineEnv(config=dict(
            use_render=False,
            image_observation=True,
            agent_policy=ReplayEgoCarPolicy,
            horizon=n_steps + 50,
            set_static=True,
            camera_smooth=False,
            vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False,
                               image_source="rgb_camera"),
            sensors=dict(rgb_camera=(RGBCamera, 800, 600)),
            norm_pixel=False,
            height_scale=0.01,
        ))
    else:
        from metadrive.envs.scenario_env import ScenarioOnlineEnv
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy

        env = ScenarioOnlineEnv(config=dict(
            use_render=render,
            agent_policy=ReplayEgoCarPolicy,
            horizon=n_steps + 50,
            set_static=True,
            camera_smooth=False,
            vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False),
            height_scale=0.01,
        ))

    env.set_scenario(scenario)
    env.reset()

    # Fix ego z-height
    ego = env.agent
    if ego is not None:
        p = ego.position
        ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)

    # BEV map
    if args.bev or not args.chase:
        from metadrive.utils.draw_top_down_map import draw_top_down_map
        map_img = draw_top_down_map(env.current_map, resolution=(2048, 1024), semantic_map=True)
        cv2.imwrite(os.path.join(out_dir, "bev_map.png"), map_img)
        print(f"  BEV map -> {out_dir}/bev_map.png")

    writer = None
    saved = 0
    for i in range(n_steps):
        obs, *_ = env.step([0.0, 0.0])

        if args.bev and i % args.bev_every == 0:
            img = env.render(mode="topdown", window=False, screen_size=(2048, 1024),
                             film_size=(12000, 12000), semantic_map=False)
            if img is not None:
                cv2.imwrite(os.path.join(out_dir, f"bev_{i:04d}.png"), img)

        if args.chase and i % 3 == 0:
            frame = None
            if isinstance(obs, dict) and "image" in obs:
                frame = obs["image"]
            if frame is not None:
                if frame.ndim == 4:
                    frame = frame[..., -1]
                if frame.dtype in (np.float32, np.float64):
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), frame)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(out_dir, "chase_h264.mp4"),
                                             cv2.VideoWriter_fourcc(*"avc1"), 10, (w, h))
                writer.write(frame)
                saved += 1

    if writer:
        writer.release()
    env.close()
    print(f"  Saved {saved} chase frames" if args.chase else "  Done")
    print(f"  Output: {out_dir}/")


if __name__ == "__main__":
    main()
