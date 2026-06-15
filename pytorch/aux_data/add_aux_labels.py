#!/usr/bin/env python3
"""
Generate road_mask and vehicle_heatmap for each .npz trajectory.

Road mask: SUMO lane polygons → ego-centric projection → 64×64 binary mask
Vehicle heatmap: GT vehicle positions → Gaussian blobs on 64×64 grid

Usage:
    python aux_data/add_aux_labels.py --loc 0 --data-dir mirro_data_map/exid_dreamer_data

Output:
    Each .npz gains: road_mask (T, 64, 64) uint8, vehicle_heatmap (T, 64, 64) float32
"""

import os, sys, math, argparse, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import sumolib
except ImportError:
    print("ERROR: pip install sumolib")
    sys.exit(1)

try:
    from shapely.geometry import Point, Polygon
    from shapely import affinity
except ImportError:
    print("ERROR: pip install shapely")
    sys.exit(1)


# ── BEV Camera Parameters (must match collect_merge_data.py) ──
BEV_HEIGHT = 50.0       # camera Z
BEV_PITCH = -89.0       # degrees, nearly straight down
CAMERA_FOV = 65.0       # vertical FOV degrees
BEV_W, BEV_H = 400, 300  # RGBCamera resolution
CROP_SIZE = 300          # center-crop to square

# Derived intrinsics
_FY = (BEV_H / 2.0) / math.tan(math.radians(CAMERA_FOV) / 2.0)  # fx = fy ≈ 235.44

# Pitch offset from straight-down: cos(89°) and sin(89°)
_CP = math.cos(math.radians(89.0))  # ≈ 0.01745
_SP = math.sin(math.radians(89.0))  # ≈ 0.99985

TARGET_SIZE = 64


def get_map_file(loc_id, map_dir):
    """Find SUMO .net.xml for a location."""
    orig = os.path.join(map_dir, f"exid_loc{loc_id}_orig.net.xml")
    plain = os.path.join(map_dir, f"exid_loc{loc_id}.net.xml")
    for p in [orig, plain]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No SUMO map for loc {loc_id} in {map_dir}")


def compute_map_offset(loc_id, map_dir):
    """Compute SUMO boundary center offset (must match build_map_features).

    .npz positions are raw CSV coords. BEV is rendered and road polygons
    are centered in MetaDrive coords (raw + off_x, raw + off_y).
    Without this correction, road_mask shifts by 200-400m.
    """
    net_xml = get_map_file(loc_id, map_dir)
    net = sumolib.net.readNet(net_xml, withInternal=True)
    xmin, ymin, xmax, ymax = net.getBoundary()
    off_x = -(xmax + xmin) / 2
    off_y = -(ymax + ymin) / 2
    return off_x, off_y


def load_lane_polygons_meta(loc_id, map_dir):
    """Load lane polygons, excluding junctions (to preserve highway medians)."""
    net_xml = get_map_file(loc_id, map_dir)

    try:
        from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
        from metadrive.scenario import ScenarioDescription as SD
        from metadrive.type import MetaDriveType

        graph = RoadLaneJunctionGraph(net_xml)
        features = extract_map_features(graph)
        print(f"  loc {loc_id}: {len(features)} map features extracted")

        onramp_edges = set()
        onramp_lanelet_ids = load_lanelet2_onramp_ids(loc_id)
        if onramp_lanelet_ids:
            onramp_edges = _map_lanelet_to_sumo_edges(graph, onramp_lanelet_ids)
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
                            if rid in onramp_edges or l.type in ('onRamp', 'offRamp'):
                                is_ramp = True
                            break
            try:
                poly = Polygon(poly_coords).buffer(0.5)
                if not poly.is_empty:
                    polygons.append((poly, is_ramp))
            except Exception:
                pass

        print(f"  loc {loc_id}: {len(polygons)} road polygons "
              f"(ramp={sum(1 for _, r in polygons if r)})")
        return polygons

    except ImportError:
        return _load_polygons_fallback(loc_id, map_dir)


def load_lanelet2_onramp_ids(loc_id):
    """Load Lanelet2 onramp lanelet IDs from .osm map."""
    import xml.etree.ElementTree as ET
    dataset_dir = os.environ.get('EXID_DATASET',
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Downloads', 'exiD-dataset-v2.1'))
    # Try local path
    osm_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mirro_data_map',
                           'maps', 'lanelet2')
    if not os.path.exists(osm_path):
        # Try another path
        osm_path = os.path.join(dataset_dir, 'maps', 'lanelet2')

    # Find location dir
    for d in os.listdir(osm_path) if os.path.exists(osm_path) else []:
        if d.startswith(f'{loc_id}_'):
            osm_file = os.path.join(osm_path, d, f'location{loc_id}.osm')
            if os.path.exists(osm_file):
                tree = ET.parse(osm_file)
                root = tree.getroot()
                onramp_ids = set()
                for rel in root.findall('relation'):
                    tags = {t.get('k'): t.get('v') for t in rel.findall('tag')}
                    if tags.get('type') == 'lanelet' and tags.get('onramp') == 'yes':
                        onramp_ids.add(int(rel.get('id')))
                if onramp_ids:
                    return onramp_ids
    return set()


def _map_lanelet_to_sumo_edges(graph, onramp_lanelet_ids):
    """Map Lanelet2 onramp IDs to SUMO edge IDs using trajectory data."""
    # Simplified: find SUMO edges whose lane types suggest onramp
    onramp_edges = set()
    for rid, road in graph.roads.items():
        for lane in road.lanes:
            if lane.type in ('onRamp', 'offRamp'):
                onramp_edges.add(rid)
    return onramp_edges


def _load_polygons_fallback(loc_id, map_dir):
    """Fallback: manual sumolib approach."""
    net_xml = get_map_file(loc_id, map_dir)
    raw_net = sumolib.net.readNet(net_xml, withInternal=True)
    xmin, ymin, xmax, ymax = raw_net.getBoundary()
    center_x = (xmax + xmin) / 2.0
    center_y = (ymax + ymin) / 2.0
    raw_net.move(-center_x, -center_y)

    polygons = []
    for edge in raw_net.getEdges(withInternal=False):  # skip internal (junction) edges
        is_ramp = edge.getFunction() in ('onRamp',)
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            try:
                w = lane.getWidth()
                poly = buffered_shape(shape, w).buffer(0)  # clean geometry
                if not poly.is_empty:
                    polygons.append((poly, is_ramp))
            except Exception:
                continue

    print(f"  loc {loc_id}: {len(polygons)} polygons loaded (fallback)")
    return polygons


def buffered_shape(centerline_shape, width):
    """Create polygon from centerline + width using shapely buffer."""
    from shapely.geometry import LineString
    ls = LineString([(float(p[0]), float(p[1])) for p in centerline_shape])
    return ls.buffer(width / 2.0, cap_style=1, join_style=1)


def compute_headings(positions):
    """Compute heading from position deltas (finite differences)."""
    T = len(positions)
    headings = np.zeros(T, dtype=np.float32)
    for i in range(T - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        headings[i] = math.atan2(dy, dx)
    headings[-1] = headings[-2] if T >= 2 else 0.0
    return headings


def project_local_to_bev(local_x, local_y, bev_size=TARGET_SIZE):
    """Project ego-centric local coords → BEV pixel using Panda3D camera model.

    Camera model matching MetaDrive RGBCamera:
      - Position: (ego, 50), HPR = (heading-90°, -89°, 0)
      - Intrinsics: fx=fy≈235.44, (cx,cy)=(200,150)
      - Post: [::-1] vertical flip + center-crop then scale to bev_size

    Args:
        local_x: forward distance (m), positive = ahead
        local_y: left/right distance (m), positive = left
        bev_size: output resolution (64 for masks)

    Returns (col, row) in flipped, cropped image coords.
    """
    X_cam = -local_y
    Y_cam = _CP * local_x + _SP * BEV_HEIGHT
    Z_cam = _SP * local_x - _CP * BEV_HEIGHT

    # Panda3D perspective projection (Y_cam = view direction)
    u_full = _FY * X_cam / Y_cam + BEV_W / 2.0
    v_full = _FY * Z_cam / Y_cam + BEV_H / 2.0

    # [::-1] flip from get_rgb_array_cpu
    v_flipped = (BEV_H - 1) - v_full

    # Center-crop → scale to output size
    u_crop = u_full - (BEV_W - CROP_SIZE) / 2.0
    v_crop = v_flipped

    if bev_size != CROP_SIZE:
        scale = bev_size / CROP_SIZE
        return u_crop * scale, v_crop * scale
    return u_crop, v_crop


def transform_polygon(poly, ego_x, ego_y, heading):
    """Transform polygon from global MetaDrive coords to ego-centric coords (meters)."""
    # Translate to ego
    poly_t = affinity.translate(poly, -ego_x, -ego_y)
    # Rotate by -heading (heading in radians from x-axis)
    deg = -math.degrees(heading)
    poly_r = affinity.rotate(poly_t, deg, origin=(0, 0))
    return poly_r


def render_road_and_ramp_mask(polygons, ego_x, ego_y, heading, size=TARGET_SIZE):
    """Render road_mask (all road) and ramp_mask (on-ramp only) at `size`x`size`.

    Uses exact Panda3D camera projection (perspective), replacing the
    previous orthographic approximation.
    """
    import cv2
    road = np.zeros((size, size), dtype=np.uint8)
    ramp = np.zeros((size, size), dtype=np.uint8)

    for poly, is_ramp in polygons:
        # Quick bounding-box cull in local coords
        bx, by = poly.centroid.x - ego_x, poly.centroid.y - ego_y
        cos_h = math.cos(-heading)
        sin_h = math.sin(-heading)
        bx_r = cos_h * bx - sin_h * by
        by_r = sin_h * bx + cos_h * by
        # Rough cull at ~40m (generous, camera coverage ≈ 32m from center)
        if abs(bx_r) > 40 or abs(by_r) > 40:
            continue

        try:
            poly_local = transform_polygon(poly, ego_x, ego_y, heading)
        except Exception:
            continue

        if poly_local.is_empty:
            continue

        exterior = poly_local.exterior.coords
        if len(exterior) < 3:
            continue

        px_coords = []
        for x, y in exterior:
            c, r = project_local_to_bev(x, y, size)
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


def generate_vehicle_heatmap(positions, ego_positions, ego_headings, size=TARGET_SIZE):
    """Generate vehicle heatmaps from GT vehicle positions.

    For now, returns a placeholder. Vehicle positions are NOT in the current .npz.
    We need to read them from the exiD CSV files.
    """
    # TODO: extract vehicle positions from exiD CSVs
    return np.zeros((len(ego_positions), size, size), dtype=np.float32)


def process_one_file(npz_path, polygons_cache, offsets_cache, dry_run=False):
    """Process a single .npz file, adding road_mask and ramp_mask."""
    data = dict(np.load(npz_path, allow_pickle=True))

    positions = data.get('positions')
    if positions is None:
        return

    T = len(positions)
    headings = compute_headings(positions)

    loc_id = int(data.get('location_id', -1))
    if loc_id < 0:
        return

    polygons = polygons_cache.get(loc_id)
    if polygons is None:
        return

    # Apply map offset: .npz positions are raw CSV coords,
    # but BEV/polygons are in MetaDrive coords (raw + offset)
    off_x, off_y = offsets_cache.get(loc_id, (0.0, 0.0))

    road_masks = np.zeros((T, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    ramp_masks = np.zeros((T, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    for t in range(T):
        road_masks[t], ramp_masks[t] = render_road_and_ramp_mask(
            polygons,
            float(positions[t, 0]) + off_x,
            float(positions[t, 1]) + off_y,
            float(headings[t]),
        )

    # Vehicle heatmap placeholder
    vehicle_heatmaps = generate_vehicle_heatmap(None, positions, headings)

    # Per-frame ramp phase label: True if ego is on ramp (before merge point)
    merge_idx = int(data.get('merge_frame_idx', -1))
    ramp_phase = np.zeros(T, dtype=np.bool_)
    if merge_idx > 0:
        ramp_phase[:merge_idx] = True

    if dry_run:
        road_cov = road_masks.mean() * 100
        ramp_cov = ramp_masks.mean() * 100
        empty = (road_masks.sum(axis=(1,2)) == 0).sum()
        ramp_frames = ramp_phase.sum()
        print(f"  DRY-RUN {os.path.basename(npz_path)}: T={T}, road_cov={road_cov:.1f}%, "
              f"ramp_cov={ramp_cov:.1f}%, empty={empty}/{T}, ramp_frames={ramp_frames}")
        return

    # Save
    for key in ['road_mask', 'ramp_mask', 'vehicle_heatmap', 'ramp_phase']:
        if key in data:
            del data[key]
    data['road_mask'] = road_masks
    data['ramp_mask'] = ramp_masks
    data['vehicle_heatmap'] = vehicle_heatmaps.astype(np.float32)
    data['ramp_phase'] = ramp_phase
    np.savez_compressed(npz_path, **data)
    print(f"  OK {os.path.basename(npz_path)}: T={T}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to exid_dreamer_data/')
    parser.add_argument('--map-dir', default=None, help='Path to SUMO .net.xml files')
    parser.add_argument('--loc', type=int, default=None, help='Process single location')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Max files to process')
    args = parser.parse_args()

    if args.map_dir is None:
        args.map_dir = os.path.join(os.path.dirname(args.data_dir), '..')  # try mirro_data_map/
        if not os.path.exists(os.path.join(args.map_dir, 'exid_loc0_orig.net.xml')):
            # Try sibling dir
            alt = os.path.join(os.path.dirname(__file__), '..', '..', 'mirro_data_map')
            if os.path.exists(os.path.join(alt, 'exid_loc0_orig.net.xml')):
                args.map_dir = alt

    # Pre-load lane polygons per location
    if args.loc is not None:
        locs = [args.loc]
    else:
        locs = list(range(7))  # loc 0-6

    polygons_cache = {}
    offsets_cache = {}
    for lid in locs:
        try:
            polygons_cache[lid] = load_lane_polygons_meta(lid, args.map_dir)
            offsets_cache[lid] = compute_map_offset(lid, args.map_dir)
            print(f"  loc {lid}: offset=({offsets_cache[lid][0]:.1f}, {offsets_cache[lid][1]:.1f})")
        except FileNotFoundError as e:
            print(f"  WARN: {e}")

    # Find all npz files
    pattern = os.path.join(args.data_dir, "**", "track*.npz")
    npz_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(npz_files)} npz files")

    if args.limit > 0:
        npz_files = npz_files[:args.limit]

    for i, fpath in enumerate(npz_files):
        try:
            process_one_file(fpath, polygons_cache, offsets_cache, dry_run=args.dry_run)
        except Exception as e:
            print(f"  ERROR {os.path.basename(fpath)}: {e}")
            import traceback
            traceback.print_exc()

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(npz_files)}")

    print(f"Done. Processed {len(npz_files)} files.")


if __name__ == "__main__":
    main()
