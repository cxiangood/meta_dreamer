"""
Build scenario with proper lane polygons and topology.

Directly specifies lane centerlines (from data analysis) and uses shapely
to create correct-width lane polygons. Includes entry/exit lane connections.

Usage:
    python3 mirro_data_map/build_scenario_with_polygon.py
    python3 mirro_data_map/build_scenario_with_polygon.py --render
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATASET_DIR = "/Users/jiojio/Documents/课题组/毕设/mirro_dataset_on_ramp/Highway-merge-in"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

LANE_WIDTH_MAIN = 3.75
LANE_WIDTH_RAMP = 3.0


def to_sumo(localY, localX):
    return localY, -localX


def make_polygon(centerline, width):
    """Create a lane polygon via shapely buffer (flat caps)."""
    if len(centerline) < 2:
        return None
    line = LineString(centerline)
    poly = line.buffer(width / 2.0, cap_style=CAP_STYLE.flat,
                       join_style=JOIN_STYLE.round, mitre_limit=5.0)
    if poly.is_empty:
        return None
    return np.array(poly.exterior.coords, dtype=np.float32)[:, :2]


def get_ramp_centerline(traj):
    """Compute ramp (Lane 3) centerline from data slabs."""
    ramp_data = traj[traj["laneId"] == 3].dropna(subset=["localX", "localY"])
    lane2_data = traj[traj["laneId"] == 2].dropna(subset=["localX", "localY"])

    slab_step = 10.0
    ly_vals = ramp_data["localY"].values
    ly_min, ly_max = np.percentile(ly_vals, 1), np.percentile(ly_vals, 99)
    edges = np.arange(math.floor(ly_min / slab_step) * slab_step, ly_max + slab_step, slab_step)

    ramp_slabs = []
    for i in range(len(edges) - 1):
        m = (ramp_data["localY"] >= edges[i]) & (ramp_data["localY"] < edges[i + 1])
        if m.sum() < 5:
            continue
        ramp_slabs.append((0.5 * (edges[i] + edges[i + 1]),
                           float(ramp_data.loc[m, "localX"].median())))

    lane2_slabs = {}
    for i in range(len(edges) - 1):
        m = (lane2_data["localY"] >= edges[i]) & (lane2_data["localY"] < edges[i + 1])
        if m.sum() < 5:
            continue
        mid = 0.5 * (edges[i] + edges[i + 1])
        lane2_slabs[mid] = float(lane2_data.loc[m, "localX"].median())

    ramp_slabs = np.array(ramp_slabs)
    pts = []
    for ly, lx in ramp_slabs[np.argsort(-ramp_slabs[:, 0])]:
        pts.append(to_sumo(ly, lx))

    # Convergence curve to merge point
    last_sx, last_sy = pts[-1]
    merge_end_ly = 10.0
    merge_target_lx = lane2_slabs.get(merge_end_ly, -5.625)
    merge_sx, merge_sy = to_sumo(merge_end_ly, merge_target_lx)
    for i in range(1, 9):
        t = i / 8
        s = 3 * t ** 2 - 2 * t ** 3
        pts.append((last_sx + s * (merge_sx - last_sx),
                     last_sy + s * (merge_sy - last_sy)))
    return np.array(pts, dtype=np.float32)


def build_map_features(traj):
    """Build map_features with polygons, topology, and lane lines."""
    from metadrive.scenario.scenario_description import ScenarioDescription as SD
    from metadrive.type import MetaDriveType as MT

    # Lane centerlines in SUMO coords (X=localY, Y=-localX)
    lane1_sy = -(LANE_WIDTH_MAIN / 2)                            # 1.875
    lane2_sy = -(LANE_WIDTH_MAIN + LANE_WIDTH_MAIN / 2)          # 5.625
    x_start, x_end = 220.0, -10.0

    lane1_pts = np.array([[x_start, lane1_sy], [x_end, lane1_sy]], dtype=np.float32)
    lane2_pts = np.array([[x_start, lane2_sy], [x_end, lane2_sy]], dtype=np.float32)
    ramp_pts = get_ramp_centerline(traj)

    features = {}

    # ── Lane 1 (inner) ──
    features["lane_1"] = {
        SD.TYPE: MT.LANE_SURFACE_STREET,
        SD.POLYLINE: lane1_pts,
        SD.POLYGON: make_polygon(lane1_pts, LANE_WIDTH_MAIN),
        SD.ENTRY: [],
        SD.EXIT: [],
        SD.LEFT_NEIGHBORS: [],
        SD.RIGHT_NEIGHBORS: [{"feature_id": "lane_2"}],
    }

    # ── Lane 2 (outer, ramp merge target) ──
    features["lane_2"] = {
        SD.TYPE: MT.LANE_SURFACE_STREET,
        SD.POLYLINE: lane2_pts,
        SD.POLYGON: make_polygon(lane2_pts, LANE_WIDTH_MAIN),
        SD.ENTRY: ["ramp_0"],
        SD.EXIT: [],
        SD.LEFT_NEIGHBORS: [{"feature_id": "lane_1"}],
        SD.RIGHT_NEIGHBORS: [],
    }

    # ── Ramp (Lane 3) ──
    features["ramp_0"] = {
        SD.TYPE: MT.LANE_SURFACE_STREET,
        SD.POLYLINE: ramp_pts,
        SD.POLYGON: make_polygon(ramp_pts, LANE_WIDTH_RAMP),
        SD.ENTRY: [],
        SD.EXIT: ["lane_2"],
        SD.LEFT_NEIGHBORS: [],
        SD.RIGHT_NEIGHBORS: [{"feature_id": "lane_2"}],
    }

    # ── Lane divider: Lane 1 ↔ Lane 2 ──
    b12_y = (lane1_sy + lane2_sy) / 2  # 3.75
    features["divider_1_2"] = {
        SD.TYPE: MT.LINE_BROKEN_SINGLE_WHITE,
        SD.POLYLINE: np.array([[x_start, b12_y], [x_end, b12_y]], dtype=np.float32),
    }

    # ── Edge boundaries ──
    features["edge_inner"] = {
        SD.TYPE: MT.BOUNDARY_LINE,
        SD.POLYLINE: np.array([[x_start, lane1_sy - LANE_WIDTH_MAIN / 2],
                                [x_end, lane1_sy - LANE_WIDTH_MAIN / 2]], dtype=np.float32),
    }
    features["edge_outer"] = {
        SD.TYPE: MT.BOUNDARY_LINE,
        SD.POLYLINE: np.array([[x_start, lane2_sy + LANE_WIDTH_MAIN / 2],
                                [x_end, lane2_sy + LANE_WIDTH_MAIN / 2]], dtype=np.float32),
    }

    # ── Ramp outer edge ──
    ramp_edge = np.array([(sx, sy - LANE_WIDTH_RAMP / 2) for sx, sy in ramp_pts], dtype=np.float32)
    features["edge_ramp"] = {SD.TYPE: MT.BOUNDARY_LINE, SD.POLYLINE: ramp_edge}

    for k, v in features.items():
        has_poly = SD.POLYGON in v
        entry = v.get(SD.ENTRY, [])
        exit_l = v.get(SD.EXIT, [])
        print(f"  {k}: polygon={has_poly}, entry={entry}, exit={exit_l}")

    return features


def build_tracks(traj, meta, dataset_dir=DATASET_DIR):
    """Build vehicle tracks from trajectory data."""
    from metadrive.type import MetaDriveType as MT

    if meta["RampVehicle"].any():
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
            pos[fi, 0] = float(r["localY"])
            pos[fi, 1] = -float(r["localX"])
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


def build_scenario(dataset_dir=DATASET_DIR):
    """Build complete ScenarioDescription."""
    from metadrive.scenario.scenario_description import ScenarioDescription as SD
    from metadrive.type import MetaDriveType as MT

    traj = pd.read_csv(os.path.join(dataset_dir, "Trajectory.csv"))
    traj.columns = [c.strip() for c in traj.columns]
    meta = pd.read_csv(os.path.join(dataset_dir, "TrackIDstate.csv"))

    map_features = build_map_features(traj)
    tracks, sdc_id, f0, f1, ego_class, t_len, ts_arr = build_tracks(traj, meta, dataset_dir)

    sdc_str = str(sdc_id)
    scenario_dict = {
        "id": f"highway-merge-poly-{sdc_id}",
        "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": MT.COORDINATE_METADRIVE,
            "ts": ts_arr, "sdc_id": sdc_str,
            "scenario_id": f"hmi_poly_{sdc_id}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    scenario = build_scenario()
    t_len = int(scenario["length"])
    print(f"\nScenario: {scenario['id']}, {t_len} frames, {len(scenario['tracks'])} vehicles")

    if not args.render:
        os.environ["METADRIVE_HEADLESS"] = "1"

    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy

    out_dir = os.path.join(OUTPUT_DIR, "polygon_chase")
    os.makedirs(out_dir, exist_ok=True)

    env = ScenarioOnlineEnv(config=dict(
        use_render=args.render,
        image_observation=not args.render,
        agent_policy=ReplayEgoCarPolicy,
        horizon=t_len + 50,
        set_static=True,
        camera_smooth=False,
        vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False,
                           image_source="rgb_camera"),
        sensors=dict(rgb_camera=(RGBCamera, 800, 600)),
        norm_pixel=False,
    ))
    env.set_scenario(scenario)
    env.reset()

    # BEV map
    from metadrive.utils.draw_top_down_map import draw_top_down_map
    map_img = draw_top_down_map(env.current_map, resolution=(2048, 1024), semantic_map=True)
    cv2.imwrite(os.path.join(out_dir, "bev_map.png"), map_img)
    print(f"  BEV map → {out_dir}/bev_map.png")

    writer = None
    saved = 0
    for i in range(t_len):
        obs, *_ = env.step([0.0, 0.0])

        if i % args.every != 0:
            continue

        frame = None
        if not args.render and isinstance(obs, dict) and "image" in obs:
            frame = obs["image"]
            if frame.ndim == 4:
                frame = frame[..., -1]
            if frame.dtype in (np.float32, np.float64):
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        if frame is not None:
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}.png"), frame)
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(os.path.join(out_dir, "chase.mp4"), fourcc, args.fps, (w, h))
            writer.write(frame)
            saved += 1

    if writer:
        writer.release()
    env.close()
    print(f"  {saved} frames + video → {out_dir}/")


if __name__ == "__main__":
    main()
