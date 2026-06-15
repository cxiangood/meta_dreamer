"""
Replay highway-merge-in using SUMO-loaded map for correct road geometry.

Uses SumoMapManager to load highway_merge.net.xml, producing proper lane
polygons and topology. Vehicle tracks are built from CSV trajectory data.

Usage:
    python3 mirro_data_map/replay_with_sumo_map.py --render
    python3 mirro_data_map/replay_with_sumo_map.py --bev --bev-every 20
    python3 mirro_data_map/replay_with_sumo_map.py --chase
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

DATASET_DIR = "/Users/jiojio/Documents/课题组/毕设/mirro_dataset_on_ramp/Highway-merge-in"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
NET_XML = os.path.join(OUTPUT_DIR, "highway_merge.net.xml")


def build_sumo_map_features():
    """Load map_features from SUMO .net.xml.

    Returns (features, offset_x, offset_y) where offset is the centering
    transform applied by RoadLaneJunctionGraph (move to origin).
    """
    os.environ["SUMO_HOME"] = SUMO_HOME
    import sumolib
    from metadrive.utils.sumo.map_utils import extract_map_features, RoadLaneJunctionGraph

    # Compute the same centering offset that RoadLaneJunctionGraph applies
    raw_net = sumolib.net.readNet(NET_XML, withInternal=True,
                                  withPedestrianConnections=True, withPrograms=True)
    xmin, ymin, xmax, ymax = raw_net.getBoundary()
    offset_x = -(xmax + xmin) / 2
    offset_y = -(ymax + ymin) / 2

    graph = RoadLaneJunctionGraph(NET_XML)
    features = extract_map_features(graph)

    # MetaDrive broken line rendering requires >= 4 polyline points to draw
    # at least one dash segment. SUMO-generated dividers only have 2 points,
    # so we resample all broken lane lines to have enough intermediate points.
    from metadrive.type import MetaDriveType as MT
    SPACING = 2.0  # meters between sample points
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
                    t = i / n
                    pts.append(pl[0] + t * d)
                pts.append(pl[-1])
                v["polyline"] = np.array(pts, dtype=np.float32)

    print(f"SUMO map: {len(features)} features, offset=({offset_x:.1f}, {offset_y:.1f})")
    return features, offset_x, offset_y


def build_tracks(traj, meta, offset_x=0.0, offset_y=0.0, dataset_dir=DATASET_DIR,
                 sdc_id_override=None):
    """Build vehicle tracks from trajectory data.

    Args:
        offset_x, offset_y: Centering offset from SUMO map (applied to positions).
        sdc_id_override: If set, use this trackId as ego instead of first RampVehicle.
    """
    from metadrive.type import MetaDriveType as MT

    if sdc_id_override is not None:
        sdc_track_id = int(sdc_id_override)
    elif meta["RampVehicle"].any():
        sdc_track_id = int(meta.loc[meta["RampVehicle"], "trackId"].iloc[0])
    else:
        sdc_track_id = int(meta["trackId"].iloc[0])

    ego_row = meta.loc[meta["trackId"] == sdc_track_id].iloc[0]
    f0 = int(ego_row["InitialFrame"])
    f1 = f0 + int(ego_row["TotalFrame"]) - 1
    ego_class = str(ego_row["VehicleClass"])
    window = traj[(traj["frameId"] >= f0) & (traj["frameId"] <= f1)].copy()

    candidates = [int(t) for t in window["trackId"].unique() if int(t) != sdc_track_id][:48]
    selected = [sdc_track_id] + candidates
    meta_by_id = meta.set_index("trackId")
    t_len = f1 - f0 + 1
    ts_arr = np.arange(t_len, dtype=np.float32) * 0.1

    tracks = {}
    for tid in selected:
        try:
            vclass = str(meta_by_id.loc[tid, "VehicleClass"])
        except KeyError:
            vclass = "car"
        sub = window[window["trackId"] == tid].sort_values("frameId")
        pos = np.zeros((t_len, 3), dtype=np.float32)
        vel = np.zeros((t_len, 2), dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)
        length = np.zeros(t_len, dtype=np.float32)
        width_arr = np.zeros(t_len, dtype=np.float32)
        height = np.ones(t_len, dtype=np.float32) * 1.5

        for _, r in sub.iterrows():
            fi = int(float(r["frameId"])) - f0
            if fi < 0 or fi >= t_len:
                continue
            pos[fi, 0] = float(r["localY"]) + offset_x
            pos[fi, 1] = -float(r["localX"]) + offset_y
            pos[fi, 2] = 0.0
            vel[fi, 0] = float(r["xVelocity"])
            vel[fi, 1] = float(r["yVelocity"])
            valid[fi] = True
            w, h = float(r["width"]), float(r["height"])
            ref = 4.5 if "truck" not in vclass.lower() else 8.0
            sc = ref / max(w, h, 1.0)
            length[fi] = max(w, h) * sc
            width_arr[fi] = min(w, h) * sc

        if not valid.any():
            continue
        heading = np.zeros(t_len, dtype=np.float32)
        idx = np.flatnonzero(valid)
        for k, i in enumerate(idx):
            i = int(i)
            if k == 0 and idx.size == 1:
                h = 0.0
            elif k == 0:
                d = pos[int(idx[k + 1])] - pos[i]
            elif k == idx.size - 1:
                d = pos[i] - pos[int(idx[k - 1])]
            else:
                d = pos[int(idx[k + 1])] - pos[int(idx[k - 1])]
            dn = float(np.hypot(d[0], d[1]))
            heading[i] = math.atan2(float(d[1]), float(d[0])) if dn > 0.05 else (
                float(heading[int(idx[k - 1])]) if k > 0 else 0.0)

        vel_out = np.zeros((t_len, 2), dtype=np.float32)
        for i in idx:
            i = int(i)
            spd = math.hypot(float(vel[i, 0]), float(vel[i, 1]))
            h = float(heading[i])
            vel_out[i, 0] = spd * math.cos(h)
            vel_out[i, 1] = spd * math.sin(h)

        tid_str = str(tid)
        tracks[tid_str] = {
            "type": MT.VEHICLE,
            "state": {"position": pos, "velocity": vel_out, "heading": heading,
                      "valid": valid, "length": length, "width": width_arr, "height": height},
            "metadata": {"type": MT.VEHICLE, "object_id": tid_str,
                         "dataset": "highway_merge_in"},
        }

    return tracks, sdc_track_id, f0, f1, ego_class, t_len, ts_arr


def build_scenario(dataset_dir=DATASET_DIR, track_id=None):
    """Build ScenarioDescription with SUMO-loaded map + CSV trajectories."""
    from metadrive.scenario.scenario_description import ScenarioDescription as SD
    from metadrive.type import MetaDriveType as MT

    traj = pd.read_csv(os.path.join(dataset_dir, "Trajectory.csv"))
    traj.columns = [c.strip() for c in traj.columns]
    meta = pd.read_csv(os.path.join(dataset_dir, "TrackIDstate.csv"))

    map_features, off_x, off_y = build_sumo_map_features()
    tracks, sdc_id, f0, f1, ego_class, t_len, ts_arr = build_tracks(
        traj, meta, offset_x=off_x, offset_y=off_y, dataset_dir=dataset_dir,
        sdc_id_override=track_id)

    sdc_str = str(sdc_id)
    scenario_dict = {
        "id": f"highway-merge-sumo-{sdc_id}",
        "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": MT.COORDINATE_METADRIVE,
            "ts": ts_arr, "sdc_id": sdc_str,
            "scenario_id": f"hmi_sumo_{sdc_id}",
            "dataset": "highway_merge_in",
            "source_file": os.path.basename(dataset_dir),
            "ego_vehicle_class": ego_class,
            "frame_range": (f0, f1),
        },
        "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": map_features,
    }
    SD.sanity_check(scenario_dict, check_self_type=True)
    return SD(scenario_dict)


def main():
    parser = argparse.ArgumentParser(description="Replay highway-merge-in with SUMO-loaded map")
    parser.add_argument("--render", action="store_true", help="Open 3D viewer")
    parser.add_argument("--bev", action="store_true", help="Save BEV screenshots")
    parser.add_argument("--bev-every", type=int, default=20)
    parser.add_argument("--chase", action="store_true", help="Capture chase camera frames")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--track-id", type=int, default=None, help="Ego vehicle trackId")
    parser.add_argument("--out-dir", default=os.path.join(OUTPUT_DIR, "sumo_replay"))
    args = parser.parse_args()

    scenario = build_scenario(track_id=args.track_id)
    t_len = int(scenario["length"])
    n_steps = args.steps or t_len
    print(f"Scenario: {scenario['id']}, {t_len} frames, {len(scenario['tracks'])} vehicles")

    out_dir = args.out_dir
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

    # Fix vehicle z-height: MetaDrive's scenario_map_manager forces ego z=0 at
    # reset, but the correct placement is HEIGHT/2 (chassis center above ground).
    # With set_static=True, physics doesn't correct z, so we fix it manually.
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
