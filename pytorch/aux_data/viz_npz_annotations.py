#!/usr/bin/env python3
"""
Visualize .npz BEV frames with road mask, vehicle positions, and ego trace.

Road mask is rendered ON-THE-FLY at 300×300 (native camera crop resolution)
using Panda3D perspective camera projection + SUMO lane polygons.

Usage:
    python aux_data/viz_npz_annotations.py \
        --npz mirro_data_map/exid_dreamer_data/rec61/track143.npz \
        --frames 0,30,60,90,125
"""

import os, sys, math, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── BEV Camera Parameters (exact MetaDrive RGBCamera model) ──
BEV_HEIGHT = 50.0
CAMERA_FOV = 65.0
BEV_W, BEV_H = 400, 300
CROP_SIZE = 300  # center-crop to square

# Derived intrinsics: fx = fy = (H/2) / tan(FOV/2) ≈ 235.44
_FY = (BEV_H / 2.0) / math.tan(math.radians(CAMERA_FOV) / 2.0)

# Pitch offset from straight-down: cos(89°) and sin(89°) for pitch=-89°
# R = R_z(heading-90°) * R_x(-89°) → local X_cam=-local_y, Y_cam=cp*lx+sp*H, Z_cam=sp*lx-cp*H
_CP = math.cos(math.radians(89.0))  # ≈ 0.01745
_SP = math.sin(math.radians(89.0))  # ≈ 0.99985


# ═══════════════════════════════════════════════════════════════════════════════
# Camera projection
# ═══════════════════════════════════════════════════════════════════════════════

def project_local_to_bev(local_x, local_y, bev_size=300):
    """Project ego-centric local coords → center-cropped BEV pixel.

    Panda3D camera model:
      Camera at (ego, 50), HPR = (exiD_heading - 90°, -89°, 0)
      X_cam = -local_y   (camera right = world right)
      Y_cam = cp*lx + sp*50   (view direction ≈ 50m)
      Z_cam = sp*lx - cp*50   (camera up ≈ local_x)
      u = fx * X_cam/Y_cam + cx, v = fy * Z_cam/Y_cam + cy
      v_flipped = 299 - v, u_crop = u - 50

    Args:
        local_x: forward (m), positive = ahead
        local_y: positive = left
        bev_size: 300 for visualization
    Returns (col, row).
    """
    X_cam = -local_y
    Y_cam = _CP * local_x + _SP * BEV_HEIGHT
    Z_cam = _SP * local_x - _CP * BEV_HEIGHT

    u_full = _FY * X_cam / Y_cam + BEV_W / 2.0
    v_full = _FY * Z_cam / Y_cam + BEV_H / 2.0

    # get_rgb_array_cpu [::-1] flip + center-crop
    v_flipped = (BEV_H - 1) - v_full
    u_crop = u_full - (BEV_W - CROP_SIZE) / 2.0

    if bev_size != CROP_SIZE:
        scale = bev_size / CROP_SIZE
        return u_crop * scale, v_flipped * scale
    return u_crop, v_flipped


def world_to_local(wx, wy, ego_x, ego_y, ego_heading):
    """World coords → ego-centric local coords."""
    dx = wx - ego_x
    dy = wy - ego_y
    cos_h = math.cos(-ego_heading)
    sin_h = math.sin(-ego_heading)
    local_x = cos_h * dx - sin_h * dy
    local_y = sin_h * dx + cos_h * dy
    return local_x, local_y


def compute_headings(positions):
    T = len(positions)
    headings = np.zeros(T, dtype=np.float32)
    for i in range(T - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        headings[i] = math.atan2(dy, dx)
    headings[-1] = headings[-2] if T >= 2 else 0.0
    return headings


# ═══════════════════════════════════════════════════════════════════════════════
# Road mask rendering at 300×300 (on-the-fly from SUMO polygons)
# ═══════════════════════════════════════════════════════════════════════════════

_ROAD_POLYGON_CACHE = {}  # loc_id → [(polygon, is_ramp), ...]


def _load_road_polygons(loc_id):
    """Load SUMO lane polygons, excluding junctions (to preserve highway medians)."""
    if loc_id in _ROAD_POLYGON_CACHE:
        return _ROAD_POLYGON_CACHE[loc_id]

    import sumolib
    from shapely.geometry import Polygon

    map_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'mirro_data_map')
    net_xml = os.path.join(map_dir, f'exid_loc{loc_id}_orig.net.xml')
    if not os.path.exists(net_xml):
        net_xml = os.path.join(map_dir, f'exid_loc{loc_id}.net.xml')

    try:
        from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
        from metadrive.scenario import ScenarioDescription as SD
        from metadrive.type import MetaDriveType

        graph = RoadLaneJunctionGraph(net_xml)
        features = extract_map_features(graph)

        onramp_edges = set()
        for rid, road in graph.roads.items():
            for lane in road.lanes:
                if lane.type in ('onRamp', 'offRamp'):
                    onramp_edges.add(rid)

        polygons = []
        for fid, f in features.items():
            poly_coords = f.get(SD.POLYGON, [])
            if len(poly_coords) < 3:
                continue
            ftype = f.get(SD.TYPE, '')
            is_road = ftype in (MetaDriveType.LANE_SURFACE_STREET, MetaDriveType.LANE_SURFACE_STREET)
            if not is_road:
                continue

            is_ramp = False
            if fid.startswith('lane_'):
                lane_id = fid[5:]
                for rid, road in graph.roads.items():
                    for l in road.lanes:
                        if l.name == lane_id:
                            if rid in onramp_edges:
                                is_ramp = True
                            break

            try:
                poly = Polygon(poly_coords).buffer(0.5)
                if not poly.is_empty:
                    polygons.append((poly, is_ramp))
            except Exception:
                pass

    except ImportError:
        from shapely.geometry import LineString
        raw_net = sumolib.net.readNet(net_xml, withInternal=True)
        xmin, ymin, xmax, ymax = raw_net.getBoundary()
        cx = (xmax + xmin) / 2.0
        cy = (ymax + ymin) / 2.0
        raw_net.move(-cx, -cy)
        polygons = []
        for edge in raw_net.getEdges(withInternal=False):
            is_ramp = edge.getFunction() in ('onRamp',)
            for lane in edge.getLanes():
                shape = lane.getShape()
                if len(shape) < 2:
                    continue
                try:
                    w = lane.getWidth()
                    ls = LineString([(float(p[0]), float(p[1])) for p in shape])
                    poly = ls.buffer(w / 2.0, cap_style=1, join_style=1)
                    if not poly.is_empty:
                        polygons.append((poly, is_ramp))
                except Exception:
                    continue

    _ROAD_POLYGON_CACHE[loc_id] = polygons
    print(f"  Loaded {len(polygons)} road polygons for loc {loc_id} "
          f"(ramp={sum(1 for _, r in polygons if r)})")
    return polygons


def render_road_mask_300(polygons, ego_x_meta, ego_y_meta, heading_rad):
    """Render road and ramp masks at 300×300 using camera projection.

    Lane polygons with minimal buffer (0.5m) to bridge narrow inter-lane gaps.
    Morphological close handles residual 1-2px cracks.

    Returns: (road_300, ramp_300) each (300, 300) uint8
    """
    from shapely import affinity
    import cv2

    road = np.zeros((300, 300), dtype=np.uint8)
    ramp = np.zeros((300, 300), dtype=np.uint8)

    for poly, is_ramp in polygons:
        poly_t = affinity.translate(poly, -ego_x_meta, -ego_y_meta)
        poly_local = affinity.rotate(poly_t, -math.degrees(heading_rad), origin=(0, 0))
        if poly_local.is_empty:
            continue
        exterior = poly_local.exterior.coords
        if len(exterior) < 3:
            continue
        px_coords = []
        for x, y in exterior:
            c, r = project_local_to_bev(x, y, 300)
            px_coords.append((int(round(c)), int(round(r))))
        pts = np.array(px_coords, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(road, [pts], 1)
        if is_ramp:
            cv2.fillPoly(ramp, [pts], 1)

    # Close tiny cracks between adjacent lane polygons
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)
    road = (road > 0).astype(np.uint8)
    if ramp.sum() > 0:
        ramp = cv2.morphologyEx(ramp, cv2.MORPH_CLOSE, kernel)
        ramp = (ramp > 0).astype(np.uint8)

    return road, ramp


# ═══════════════════════════════════════════════════════════════════════════════
# Vehicle loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_surrounding_vehicles(csv_path, ego_track_id, n_frames):
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, low_memory=False,
                         usecols=['trackId', 'frame', 'xCenter', 'yCenter', 'heading',
                                  'lonVelocity', 'width', 'length'])
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False,
                         usecols=['trackId', 'frame', 'xCenter', 'yCenter', 'heading',
                                  'lonVelocity'])

    ego_rows = df[df['trackId'] == ego_track_id]['frame']
    if len(ego_rows) == 0:
        return {fi: [] for fi in range(n_frames)}
    ego_start_frame = int(ego_rows.min())

    others_by_frame = {}
    for fi in range(n_frames):
        csv_fi = ego_start_frame + fi
        frame_data = df[df['frame'] == csv_fi]
        vehicles = []
        for _, row in frame_data.iterrows():
            tid = int(row['trackId'])
            if tid == ego_track_id:
                continue
            vehicles.append({
                'x': float(row['xCenter']),
                'y': float(row['yCenter']),
                'heading': float(row['heading']),
                'width': float(row.get('width', 2.0)),
                'length': float(row.get('length', 4.5)),
            })
        others_by_frame[fi] = vehicles
    return others_by_frame


# ═══════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def center_crop_bev(bev_img):
    """Center-crop 400×300 BEV to 300×300 square."""
    H, W = bev_img.shape[:2]
    size = min(H, W)
    dh, dw = (H - size) // 2, (W - size) // 2
    return bev_img[dh:dh + size, dw:dw + size]


def overlay_road_mask(ax, road_300, ramp_300):
    """Overlay road mask (green) and ramp mask (red) on current axis."""
    overlay = np.zeros((300, 300, 3), dtype=np.float32)
    overlay[road_300 > 0, 1] = 0.35   # green = road
    overlay[ramp_300 > 0, 0] = 0.45   # red = ramp
    overlay[ramp_300 > 0, 1] = 0.0
    ax.imshow(overlay, alpha=0.4)


def render_vehicles(ax, others, ego_x, ego_y, ego_heading):
    """Draw surrounding vehicles as cyan boxes + direction arrows."""
    for veh in others:
        local_x, local_y = world_to_local(veh['x'], veh['y'], ego_x, ego_y, ego_heading)
        c, r = project_local_to_bev(local_x, local_y, 300)

        if not (0 <= c < 300 and 0 <= r < 300):
            continue

        # Vehicle heading relative to ego
        rel_heading = math.radians(veh['heading']) - ego_heading
        arrow_len = 10
        dc = -arrow_len * math.sin(rel_heading)
        dr = -arrow_len * math.cos(rel_heading)

        ax.arrow(c, r, dc, dr, color='cyan', head_width=4, head_length=4,
                 alpha=0.85, linewidth=0.8)

        # Perspective-aware meters-per-pixel
        mpp = (_CP * local_x + _SP * BEV_HEIGHT) / _FY
        w_px = veh['width'] / mpp
        l_px = veh['length'] / mpp
        angle_deg = -90 + math.degrees(rel_heading)
        rect = plt.Rectangle((c - l_px / 2, r - w_px / 2), l_px, w_px,
                             angle=angle_deg, rotation_point='center',
                             fill=False, edgecolor='cyan', linewidth=0.8, alpha=0.8)
        ax.add_patch(rect)


def render_ego_trace(ax, positions, fi, ego_heading, trail_len=20):
    """Draw ego trajectory trace in yellow + star at current position."""
    start = max(0, fi - trail_len)
    ego_x, ego_y = positions[fi, 0], positions[fi, 1]

    px_list = []
    for t in range(start, fi + 1):
        lx, ly = world_to_local(positions[t, 0], positions[t, 1], ego_x, ego_y, ego_heading)
        c, r = project_local_to_bev(lx, ly, 300)
        px_list.append((c, r))

    for t in range(1, len(px_list)):
        alpha = 0.3 + 0.7 * t / len(px_list)
        ax.plot([px_list[t-1][0], px_list[t][0]],
                [px_list[t-1][1], px_list[t][1]],
                color='yellow', linewidth=0.8, alpha=alpha)

    ax.plot(px_list[-1][0], px_list[-1][1], 'y*', markersize=10,
            markeredgecolor='orange', markeredgewidth=1.5)


def compute_map_offset(loc_id):
    """Compute SUMO map center offset (raw CSV → MetaDrive coords)."""
    map_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'mirro_data_map')
    net_xml = os.path.join(map_dir, f'exid_loc{loc_id}_orig.net.xml')
    if not os.path.exists(net_xml):
        net_xml = os.path.join(map_dir, f'exid_loc{loc_id}.net.xml')
    try:
        import sumolib
        net = sumolib.net.readNet(net_xml, withInternal=True)
        xmin, ymin, xmax, ymax = net.getBoundary()
        off_x = -(xmax + xmin) / 2
        off_y = -(ymax + ymin) / 2
        return off_x, off_y
    except Exception as e:
        print(f"  WARN: cannot compute offset for loc {loc_id}: {e}, using (0,0)")
        return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--frames', default='0,60,125,200,300')
    parser.add_argument('--output', default=None)
    parser.add_argument('--csv', default=None)
    args = parser.parse_args()

    data = dict(np.load(args.npz, allow_pickle=True))
    bev_images = data['bev_images']
    positions = data['positions']
    merge_idx = int(data.get('merge_frame_idx', -1))
    rec_id = int(data.get('recording_id', -1))
    track_id = int(data.get('track_id', -1))
    loc_id = int(data.get('location_id', -1))

    T = len(bev_images)
    headings = compute_headings(positions)

    # Map offset: .npz positions are raw CSV; BEV + SUMO are MetaDrive (centered)
    off_x, off_y = compute_map_offset(loc_id)
    print(f"  Map offset: ({off_x:.1f}, {off_y:.1f})")

    # Load SUMO road polygons
    polygons = _load_road_polygons(loc_id)

    # Load vehicles
    if args.csv is None:
        csv_path = (f'/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/'
                    f'exiD-dataset-v2.1/data/{rec_id:02d}_tracks.csv')
    else:
        csv_path = args.csv

    try:
        others_raw = load_surrounding_vehicles(csv_path, track_id, T)
        # Apply map offset to vehicle coords (raw → MetaDrive)
        others = {}
        for fi in others_raw:
            others[fi] = []
            for veh in others_raw[fi]:
                veh['x'] += off_x
                veh['y'] += off_y
                others[fi].append(veh)
        n_veh_frames = sum(1 for fi in others if others[fi])
        print(f"  Vehicles: {n_veh_frames} frames with others")
    except Exception as e:
        print(f"  WARN: no vehicles ({e})")
        others = {fi: [] for fi in range(T)}

    # Parse frames
    frames = [int(x) for x in args.frames.split(',')]
    frames = [f for f in frames if 0 <= f < T]
    n = len(frames)

    # Layout: 3 rows per frame
    #  Row 0: raw BEV | road_overlay (300×300 rendered on-the-fly)
    #  Row 1: vehicles | combined
    #  Row 2: road_only | ramp_only
    fig, axes = plt.subplots(3 * n, 2, figsize=(10, 7.5 * n))

    for i, fi in enumerate(frames):
        ax_bev = axes[3 * i, 0]
        ax_road = axes[3 * i, 1]
        ax_veh = axes[3 * i + 1, 0]
        ax_comb = axes[3 * i + 1, 1]
        ax_rdonly = axes[3 * i + 2, 0]
        ax_rmonly = axes[3 * i + 2, 1]

        bev_crop = center_crop_bev(bev_images[fi])

        # Ego in MetaDrive coords for this frame
        ego_x_meta = float(positions[fi, 0]) + off_x
        ego_y_meta = float(positions[fi, 1]) + off_y
        heading = float(headings[fi])

        # Render road mask at 300×300 using camera model
        road_300, ramp_300 = render_road_mask_300(
            polygons, ego_x_meta, ego_y_meta, heading)

        # Determine ramp phase
        ramp_phase = data.get('ramp_phase')
        phase = "RAMP" if (ramp_phase is not None and ramp_phase[fi]) else "MAIN"

        # (0,0): Raw BEV + ego marker
        ax_bev.imshow(bev_crop)
        c0, r0 = project_local_to_bev(0, 0, 300)
        ax_bev.plot(c0, r0, 'rx', markersize=8, markeredgewidth=1.5)
        ax_bev.set_title(f"BEV raw (ego={c0:.0f},{r0:.0f})", fontsize=9)
        ax_bev.axis('off')

        # (0,1): BEV + road mask overlay (300×300, camera model)
        ax_road.imshow(bev_crop)
        overlay_road_mask(ax_road, road_300, ramp_300)
        road_cov = road_300.mean() * 100
        ramp_cov = ramp_300.mean() * 100
        ax_road.set_title(f"BEV + Road [{phase}] cov={road_cov:.0f}%/ramp={ramp_cov:.0f}%",
                          fontsize=9)
        ax_road.axis('off')

        # (1,0): BEV + vehicles + ego trace (no road mask)
        ax_veh.imshow(bev_crop)
        render_ego_trace(ax_veh, positions, fi, heading)
        render_vehicles(ax_veh, others.get(fi, []), ego_x_meta, ego_y_meta, heading)
        ax_veh.set_title(f"BEV + Vehicles + Ego Trace", fontsize=9)
        ax_veh.axis('off')

        # (1,1): Combined (BEV + road + vehicles + trace)
        ax_comb.imshow(bev_crop)
        overlay_road_mask(ax_comb, road_300, ramp_300)
        render_ego_trace(ax_comb, positions, fi, heading)
        render_vehicles(ax_comb, others.get(fi, []), ego_x_meta, ego_y_meta, heading)

        info = f"Frame {fi}/{T-1} [{phase}]"
        if merge_idx > 0:
            info += f"  merge@{merge_idx}"
            if fi == merge_idx:
                info += " <<<"
        ax_comb.set_title(f"Combined: {info}", fontsize=9)
        ax_comb.axis('off')

        # (2,0): Road mask only
        ax_rdonly.imshow(road_300, cmap='Greens', vmin=0, vmax=1)
        ax_rdonly.set_title("Road mask (300×300 camera proj)", fontsize=9)
        ax_rdonly.axis('off')

        # (2,1): Ramp mask only
        ax_rmonly.imshow(ramp_300, cmap='Reds', vmin=0, vmax=1)
        ax_rmonly.set_title("Ramp mask (300×300 camera proj)", fontsize=9)
        ax_rmonly.axis('off')

    suptitle = f"rec{rec_id:02d} track{track_id} loc{loc_id} ({T} frames)"
    plt.suptitle(suptitle, fontsize=12, y=0.995)
    plt.tight_layout()

    out_path = args.output
    if out_path is None:
        out_dir = os.path.join(os.path.dirname(args.npz))
        base = os.path.basename(args.npz).replace('.npz', '')
        out_path = os.path.join(out_dir, f'{base}_viz.png')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
