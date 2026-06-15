"""
Standalone BEV renderer for model training.

Renders ego-centered bird's-eye-view images directly from SUMO .net.xml
geometry and CSV trajectory data — no MetaDrive env or terrain shader needed.

Outputs per-frame:
  - RGB image  (256, 256, 3)  — human-readable visualization
  - Semantic   (256, 256, 6)  — multi-channel for model input
    Ch 0: road surface mask
    Ch 1: lane divider lines (broken white)
    Ch 2: edge divider lines (solid yellow)
    Ch 3: ego vehicle mask
    Ch 4: surrounding vehicle mask
    Ch 5: vehicle heading (sin encoded)

Usage:
    # exiD dataset:
    python3 mirro_data_map/bev_renderer.py --recording 01 --track-id 45
    python3 mirro_data_map/bev_renderer.py --recording 01 --track-id 45 --every 5 --format both
    # mirro dataset:
    python3 mirro_data_map/bev_renderer.py --dataset mirro --track-id 132
    python3 mirro_data_map/bev_renderer.py --dataset mirro --every 2 --format npy
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
os.environ["SUMO_HOME"] = SUMO_HOME
if SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)

EXID_DATASET_DIR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1"
EXID_DATA_DIR = os.path.join(EXID_DATASET_DIR, "data")
MIRRO_DATASET_DIR = "/Users/jiojio/Documents/课题组/毕设/mirro_dataset_on_ramp/Highway-merge-in"
MAP_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# SUMO network loader
# ---------------------------------------------------------------------------

class SUMONetwork:
    """Load lane geometry from SUMO .net.xml."""

    def __init__(self, net_xml: str):
        import sumolib
        import sumolib.geomhelper

        raw_net = sumolib.net.readNet(
            net_xml, withInternal=True,
            withPedestrianConnections=True, withPrograms=True,
        )
        xmin, ymin, xmax, ymax = raw_net.getBoundary()
        self.cx = (xmax + xmin) / 2
        self.cy = (ymax + ymin) / 2

        self.lane_polygons = []   # list of (polygon_xy, lane_type_str)
        self.lane_dividers = []   # list of polyline_xy
        self.edge_dividers = []   # list of polyline_xy

        edge_borders = []
        for edge in raw_net.getEdges(withInternal=True):
            lanes = edge.getLanes()
            is_internal = edge.getFunction() in ("internal", "walkingarea", "crossing")

            for lane in lanes:
                shape = lane.getShape()
                width = lane.getWidth()
                # Center shape at origin
                shape = [(p[0] - self.cx, p[1] - self.cy) for p in shape]

                # Build lane polygon by buffering centerline
                left = sumolib.geomhelper.move2side(shape, -width / 2)
                right = sumolib.geomhelper.move2side(shape, width / 2)
                polygon = list(left) + list(reversed(right))

                # Determine type
                edge_type = edge.getType()
                parts = edge_type.split("|") if "|" in edge_type else []
                idx = lane.getIndex()
                if len(parts) > idx:
                    lane_type = parts[idx].strip()
                elif lane.allows("pedestrian") and not lane.allows("passenger"):
                    lane_type = "sidewalk"
                elif edge.getFunction() == "walkingarea":
                    lane_type = "sidewalk"
                else:
                    lane_type = "driving"

                self.lane_polygons.append((polygon, lane_type))

                # Dividers (only from non-internal edges)
                if not is_internal:
                    left_border = sumolib.geomhelper.move2side(shape, -width / 2)
                    right_border = sumolib.geomhelper.move2side(shape, width / 2)
                    edge_borders.append(right_border)
                    if idx < len(lanes) - 1:
                        self.lane_dividers.append(list(left_border))
                    else:
                        edge_borders.append(left_border)

        # Compute edge dividers from overlapping borders
        for i in range(len(edge_borders) - 1):
            for j in range(i + 1, len(edge_borders)):
                bi = np.array([edge_borders[i][0], edge_borders[i][-1]])
                bj = np.array([edge_borders[j][-1], edge_borders[j][0]])
                if np.linalg.norm(bi - bj) < 1.0:
                    self.edge_dividers.append(edge_borders[i])

        n_driving = sum(1 for _, t in self.lane_polygons if t == "driving")
        print(f"SUMO network: {len(self.lane_polygons)} lanes ({n_driving} driving), "
              f"{len(self.lane_dividers)} lane dividers, {len(self.edge_dividers)} edge dividers, "
              f"center=({self.cx:.1f}, {self.cy:.1f})")


# ---------------------------------------------------------------------------
# Trajectory loader (exiD)
# ---------------------------------------------------------------------------

def load_exid_tracks(recording_id: int, ego_track_id: int | None = None,
                     max_vehicles: int = 50):
    """Load exiD trajectory data for one recording.

    Returns:
        tracks: dict  track_id_str -> {pos, heading, valid, length, width}
        sdc_id: int
        f0, f1: int  (frame range)
        t_len: int
    """
    rid = int(recording_id)
    tracks_csv = pd.read_csv(os.path.join(DATA_DIR, f"{rid:02d}_tracks.csv"), low_memory=False)
    tracks_meta = pd.read_csv(os.path.join(DATA_DIR, f"{rid:02d}_tracksMeta.csv"))

    t = tracks_csv[tracks_csv["recordingId"] == rid]
    tm = tracks_meta[tracks_meta["recordingId"] == rid]

    if ego_track_id is not None:
        sdc_id = int(ego_track_id)
    else:
        lc = t[t["laneChange"] != 0]["trackId"].unique()
        sdc_id = int(lc[0]) if len(lc) > 0 else int(t["trackId"].iloc[0])

    ego_sub = t[t["trackId"] == sdc_id].sort_values("frame")
    f0 = int(ego_sub["frame"].iloc[0])
    f1 = int(ego_sub["frame"].iloc[-1])
    t_len = f1 - f0 + 1

    window = t[(t["frame"] >= f0) & (t["frame"] <= f1)].copy()
    candidates = [int(x) for x in window["trackId"].unique() if int(x) != sdc_id]
    selected = [sdc_id] + candidates[:max_vehicles - 1]
    meta_by_id = tm.set_index("trackId")

    tracks = {}
    for tid in selected:
        try:
            mrow = meta_by_id.loc[tid]
            if isinstance(mrow, pd.DataFrame):
                mrow = mrow.iloc[0]
            default_len = float(mrow["length"])
            default_wid = float(mrow["width"])
        except (KeyError, TypeError):
            default_len, default_wid = 4.5, 2.0

        sub = window[window["trackId"] == tid].sort_values("frame")
        if len(sub) == 0:
            continue

        pos = np.zeros((t_len, 2), dtype=np.float32)
        heading = np.zeros(t_len, dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)

        for _, r in sub.iterrows():
            fi = int(r["frame"]) - f0
            if fi < 0 or fi >= t_len:
                continue
            pos[fi, 0] = float(r["xCenter"])
            pos[fi, 1] = float(r["yCenter"])
            heading[fi] = math.radians(float(r["heading"]))
            valid[fi] = True

        if not valid.any():
            continue

        tracks[str(tid)] = {
            "pos": pos, "heading": heading, "valid": valid,
            "length": default_len, "width": default_wid,
        }

    print(f"Tracks: {len(tracks)} vehicles, ego={sdc_id}, frames={f0}-{f1} ({t_len})")
    return tracks, sdc_id, f0, f1, t_len


# ---------------------------------------------------------------------------
# Trajectory loader (mirro highway-merge-in)
# ---------------------------------------------------------------------------

def load_mirro_tracks(ego_track_id: int | None = None, max_vehicles: int = 50,
                      dataset_dir: str = MIRRO_DATASET_DIR):
    """Load mirro highway-merge-in trajectory data.

    Coordinate transform: SUMO_X = localY, SUMO_Y = -localX
    Heading computed from trajectory differences.

    Returns:
        tracks: dict  track_id_str -> {pos, heading, valid, length, width}
        sdc_id: int
        f0, f1: int  (frame range)
        t_len: int
    """
    traj = pd.read_csv(os.path.join(dataset_dir, "Trajectory.csv"))
    traj.columns = [c.strip() for c in traj.columns]
    meta = pd.read_csv(os.path.join(dataset_dir, "TrackIDstate.csv"))

    if ego_track_id is not None:
        sdc_id = int(ego_track_id)
    elif meta["RampVehicle"].any():
        sdc_id = int(meta.loc[meta["RampVehicle"], "trackId"].iloc[0])
    else:
        sdc_id = int(meta["trackId"].iloc[0])

    ego_row = meta.loc[meta["trackId"] == sdc_id].iloc[0]
    f0 = int(ego_row["InitialFrame"])
    f1 = f0 + int(ego_row["TotalFrame"]) - 1
    t_len = f1 - f0 + 1

    window = traj[(traj["frameId"] >= f0) & (traj["frameId"] <= f1)].copy()
    candidates = [int(t) for t in window["trackId"].unique() if int(t) != sdc_id][:max_vehicles - 1]
    selected = [sdc_id] + candidates
    meta_by_id = meta.set_index("trackId")

    tracks = {}
    for tid in selected:
        try:
            mrow = meta_by_id.loc[tid]
            if isinstance(mrow, pd.DataFrame):
                mrow = mrow.iloc[0]
            vclass = str(mrow["VehicleClass"])
        except (KeyError, TypeError):
            vclass = "car"

        sub = window[window["trackId"] == tid].sort_values("frameId")
        if len(sub) == 0:
            continue

        pos = np.zeros((t_len, 2), dtype=np.float32)
        heading = np.zeros(t_len, dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)

        # Default dimensions (rescaled like replay_with_sumo_map.py)
        ref = 8.0 if "truck" in vclass.lower() else 4.5
        default_len = ref
        default_wid = 2.0

        for _, r in sub.iterrows():
            fi = int(float(r["frameId"])) - f0
            if fi < 0 or fi >= t_len:
                continue
            # Coordinate transform: SUMO_X = localY, SUMO_Y = -localX
            pos[fi, 0] = float(r["localY"])
            pos[fi, 1] = -float(r["localX"])
            valid[fi] = True

            # Rescale dimensions
            w, h = float(r["width"]), float(r["height"])
            sc = ref / max(w, h, 1.0)

        if not valid.any():
            continue

        # Compute heading from trajectory differences
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

        tracks[str(tid)] = {
            "pos": pos, "heading": heading, "valid": valid,
            "length": default_len, "width": default_wid,
        }

    print(f"Tracks: {len(tracks)} vehicles, ego={sdc_id}, frames={f0}-{f1} ({t_len})")
    return tracks, sdc_id, f0, f1, t_len


# ---------------------------------------------------------------------------
# BEV Renderer
# ---------------------------------------------------------------------------

class BEVRenderer:
    """Render ego-centered BEV images from SUMO geometry + vehicle tracks."""

    # Semantic channel indices
    CH_ROAD = 0
    CH_LANE_DIV = 1
    CH_EDGE_DIV = 2
    CH_EGO = 3
    CH_VEH = 4
    CH_HEADING = 5
    NUM_CHANNELS = 6

    def __init__(self, network: SUMONetwork, resolution=256, view_range=100.0):
        """
        Args:
            network: loaded SUMONetwork
            resolution: output image size (pixels)
            view_range: meters visible in each direction (total = 2 * view_range)
        """
        self.resolution = resolution
        self.view_range = view_range
        self.ppm = resolution / (2 * view_range)  # pixels per meter

        # Pre-transform road geometry from world coords to numpy arrays
        self.road_polys = []
        for poly, ltype in network.lane_polygons:
            if ltype == "driving":
                self.road_polys.append(np.array(poly, dtype=np.float32))

        self.lane_divs = [np.array(d, dtype=np.float32) for d in network.lane_dividers]
        self.edge_divs = [np.array(d, dtype=np.float32) for d in network.edge_dividers]

    def _world_to_ego_pixel(self, pts: np.ndarray, ego_pos, ego_heading):
        """Transform world-coord points to ego-centered pixel coords.

        ego is at image center, facing UP (+y direction in image).
        """
        # Translate
        pts = pts - np.array(ego_pos, dtype=np.float32)

        # Rotate by -ego_heading so ego faces UP
        c = math.cos(-ego_heading)
        s = math.sin(-ego_heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = pts @ R.T

        # To pixel: +x right, +y up → pixel (col, row) with y flipped
        px = pts[:, 0] * self.ppm + self.resolution / 2
        py = -pts[:, 1] * self.ppm + self.resolution / 2  # flip y
        return np.stack([px, py], axis=-1)

    def _draw_rotated_rect(self, img, center_px, length, width, heading, color,
                           thickness=-1):
        """Draw a rotated rectangle (vehicle) on image."""
        hw, hl = width / 2, length / 2
        corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]],
                           dtype=np.float32)
        c, s = math.cos(heading), math.sin(heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        corners = corners @ R.T + center_px
        cv2.fillPoly(img, [corners.astype(np.int32)], color)

    def render_frame(self, tracks, sdc_id, frame_idx):
        """Render one BEV frame.

        Args:
            tracks: dict from load_exid_tracks
            sdc_id: ego track id (int)
            frame_idx: frame index (0-based within window)

        Returns:
            rgb: (H, W, 3) uint8
            semantic: (H, W, 6) uint8
        """
        sdc_str = str(sdc_id)
        ego = tracks[sdc_str]
        if not ego["valid"][frame_idx]:
            # Ego not valid in this frame, return blank
            rgb = np.full((self.resolution, self.resolution, 3), 40, dtype=np.uint8)
            sem = np.zeros((self.resolution, self.resolution, self.NUM_CHANNELS), dtype=np.uint8)
            return rgb, sem

        ego_pos = ego["pos"][frame_idx]  # (2,) — already centered by CLI
        ego_heading = float(ego["heading"][frame_idx])

        # --- RGB image ---
        rgb = np.full((self.resolution, self.resolution, 3), 40, dtype=np.uint8)

        # --- Separate contiguous channel images (cv2 needs contiguous arrays) ---
        ch_road = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        ch_lane_div = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        ch_edge_div = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        ch_ego = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        ch_veh = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        ch_heading = np.zeros((self.resolution, self.resolution), dtype=np.uint8)

        # Draw road surface
        for poly in self.road_polys:
            pts_px = self._world_to_ego_pixel(poly, ego_pos, ego_heading)
            pts_int = pts_px.astype(np.int32)
            pts_int[:, 0] = np.clip(pts_int[:, 0], -500, self.resolution + 500)
            pts_int[:, 1] = np.clip(pts_int[:, 1], -500, self.resolution + 500)
            cv2.fillPoly(rgb, [pts_int], (80, 80, 80))
            cv2.fillPoly(ch_road, [pts_int], 255)

        # Draw lane dividers
        for div in self.lane_divs:
            pts_px = self._world_to_ego_pixel(div, ego_pos, ego_heading)
            pts_int = pts_px.astype(np.int32)
            cv2.polylines(rgb, [pts_int], False, (200, 200, 200), 1)
            cv2.polylines(ch_lane_div, [pts_int], False, 255, 1)

        # Draw edge dividers
        for div in self.edge_divs:
            pts_px = self._world_to_ego_pixel(div, ego_pos, ego_heading)
            pts_int = pts_px.astype(np.int32)
            cv2.polylines(rgb, [pts_int], False, (60, 60, 200), 1)
            cv2.polylines(ch_edge_div, [pts_int], False, 255, 1)

        # Draw vehicles
        for tid_str, tdata in tracks.items():
            if not tdata["valid"][frame_idx]:
                continue
            vpos = tdata["pos"][frame_idx]
            vheading = float(tdata["heading"][frame_idx])
            vlen = tdata["length"]
            vwid = tdata["width"]

            vp = np.array([vpos], dtype=np.float32)
            vpx = self._world_to_ego_pixel(vp, ego_pos, ego_heading)[0]

            margin = 20
            if vpx[0] < -margin or vpx[0] > self.resolution + margin:
                continue
            if vpx[1] < -margin or vpx[1] > self.resolution + margin:
                continue

            rel_heading = vheading - ego_heading

            if tid_str == sdc_str:
                self._draw_rotated_rect(rgb, vpx, vlen * self.ppm, vwid * self.ppm,
                                        rel_heading, (0, 200, 0))
                self._draw_rotated_rect(ch_ego, vpx, vlen * self.ppm, vwid * self.ppm,
                                        rel_heading, 255)
            else:
                self._draw_rotated_rect(rgb, vpx, vlen * self.ppm, vwid * self.ppm,
                                        rel_heading, (200, 100, 0))
                self._draw_rotated_rect(ch_veh, vpx, vlen * self.ppm, vwid * self.ppm,
                                        rel_heading, 255)
                heading_val = int((math.sin(rel_heading) + 1) / 2 * 255)
                self._draw_rotated_rect(ch_heading, vpx, vlen * self.ppm, vwid * self.ppm,
                                        rel_heading, heading_val)

        # Stack semantic channels
        sem = np.stack([ch_road, ch_lane_div, ch_edge_div, ch_ego, ch_veh, ch_heading],
                       axis=-1)
        return rgb, sem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone BEV renderer for model training")
    parser.add_argument("--dataset", choices=["exid", "mirro"], default="exid",
                        help="Dataset to use (exid or mirro)")
    parser.add_argument("--recording", type=int, default=None,
                        help="exiD recording ID (e.g. 1). Required for exiD.")
    parser.add_argument("--track-id", type=int, default=None, help="Ego vehicle trackId")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--view-range", type=float, default=50.0,
                        help="Meters visible in each direction (total = 2x)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--every", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--max-vehicles", type=int, default=50)
    parser.add_argument("--format", choices=["png", "npy", "both"], default="both")
    args = parser.parse_args()

    if args.dataset == "exid":
        if args.recording is None:
            parser.error("--recording is required for exiD dataset")
        rid = args.recording

        # Determine location from recording
        rec_meta = pd.read_csv(os.path.join(EXID_DATA_DIR, f"{rid:02d}_recordingMeta.csv"))
        loc_id = int(rec_meta["locationId"].iloc[0])

        # Load SUMO map
        net_xml = os.path.join(MAP_DIR, f"exid_loc{loc_id}_orig.net.xml")
        if not os.path.exists(net_xml):
            net_xml = os.path.join(MAP_DIR, f"exid_loc{loc_id}.net.xml")
        print(f"Loading map: {net_xml}")
        network = SUMONetwork(net_xml)

        # Load tracks
        tracks, sdc_id, f0, f1, t_len = load_exid_tracks(
            rid, ego_track_id=args.track_id, max_vehicles=args.max_vehicles)

        loc_label = f"{rid:02d}"
    else:
        # mirro
        net_xml = os.path.join(MAP_DIR, "highway_merge.net.xml")
        print(f"Loading map: {net_xml}")
        network = SUMONetwork(net_xml)

        tracks, sdc_id, f0, f1, t_len = load_mirro_tracks(
            ego_track_id=args.track_id, max_vehicles=args.max_vehicles)

        loc_label = "mirro"

    # Center track positions to match road geometry
    for tid, tdata in tracks.items():
        tdata["pos"][:, 0] -= network.cx
        tdata["pos"][:, 1] -= network.cy

    # Output dir
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"bev_output/{loc_label}_track{sdc_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}/")

    # Render
    renderer = BEVRenderer(network, resolution=args.resolution, view_range=args.view_range)
    n_rendered = 0
    for fi in range(0, t_len, args.every):
        rgb, sem = renderer.render_frame(tracks, sdc_id, fi)

        if args.format in ("png", "both"):
            cv2.imwrite(os.path.join(out_dir, f"{fi:06d}_rgb.png"), rgb)
        if args.format in ("npy", "both"):
            np.save(os.path.join(out_dir, f"{fi:06d}_semantic.npy"), sem)
        n_rendered += 1

        if n_rendered % 100 == 0:
            print(f"  Rendered {n_rendered} frames ({fi}/{t_len})")

    print(f"Done: {n_rendered} frames -> {out_dir}/")

    # Save metadata
    meta = {
        "dataset": args.dataset, "sdc_id": sdc_id, "f0": f0, "f1": f1,
        "t_len": t_len, "resolution": args.resolution,
        "view_range": args.view_range, "every": args.every,
        "center": (network.cx, network.cy),
    }
    if args.dataset == "exid":
        meta["recording"] = rid
        meta["location_id"] = loc_id
    np.save(os.path.join(out_dir, "meta.npy"), meta)
    print(f"Metadata saved to {out_dir}/meta.npy")


if __name__ == "__main__":
    main()
