"""
Convert nuPlan GeoPackage map data to MetaDrive map_features format.

Usage:
    python dreamer/tools/gpkg_to_metadrive.py \
        --pkl /share/home/u23516/code/navsim_mini/mini_navsim_logs/mini/SCENE.pkl \
        --gpkg /share/home/u23516/code/navsim_mini/maps_download/nuplan-maps-v1.0/us-nv-las-vegas-strip/9.15.1915/map.gpkg \
        --epsg 32611 \
        --output logs/map_features/SCENE.pkl

Pipeline:
    1. Load NAVSIM scene → ego trajectory in UTM
    2. Query gpkg for nearby lanes, connectors, boundaries
    3. Convert WGS84 → local metric coords (origin = ego start)
    4. Output MetaDrive-compatible map_features dict
"""

import argparse
import os
import pickle
from collections import defaultdict

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPoint, box


# ---------------------------------------------------------------------------
# gpkg → map_features conversion
# ---------------------------------------------------------------------------

def load_navsim_ego(pkl_path):
    """Load NAVSIM scene and return ego trajectory in UTM coords."""
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)
    positions = np.array([fr["ego2global_translation"][:2] for fr in frames])
    headings = []
    for fr in frames:
        # ego2global_rotation is quaternion [w, x, y, z]
        q = fr["ego2global_rotation"]
        # Convert quaternion to yaw (heading)
        w, x, y, z = q
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        headings.append(yaw)
    return positions, np.array(headings), frames


def query_gpkg(gpkg_path, ego_positions, buffer_m=150, target_epsg=None):
    """
    Query gpkg for lanes, connectors, boundaries near ego trajectory.

    Returns GeoDataFrames in target_epsg (UTM metric) coordinates.
    """
    if target_epsg is None:
        target_epsg = 32611  # default UTM zone 11N (Las Vegas)

    # Build bounding box from ego trajectory
    xmin, ymin = ego_positions.min(axis=0) - buffer_m
    xmax, ymax = ego_positions.max(axis=0) + buffer_m
    bbox_utm = box(xmin, ymin, xmax, ymax)

    # Load and reproject vector layers
    results = {}
    for layer, name in [
        ("lanes_polygons", "lanes"),
        ("baseline_paths", "baselines"),
        ("lane_connectors", "connectors"),
        ("boundaries", "boundaries"),
        ("intersections", "intersections"),
    ]:
        gdf = gpd.read_file(
            gpkg_path, layer=layer, engine="pyogrio", fid_as_index=True
        ).to_crs(f"EPSG:{target_epsg}")
        # Filter by bbox
        gdf = gdf[gdf.geometry.intersects(bbox_utm)]
        results[name] = gdf

    return results


def utm_to_local(geom_series, origin):
    """Shift geometries so that origin becomes (0, 0). Returns list of coord arrays."""
    local_lines = []
    for geom in geom_series:
        if geom is None or geom.is_empty:
            local_lines.append(None)
            continue
        coords = np.array(geom.coords)
        coords[:, 0] -= origin[0]
        coords[:, 1] -= origin[1]
        local_lines.append(coords)
    return local_lines


def polygon_to_local(geom_series, origin):
    """Shift polygon exterior coords to local frame."""
    local_polys = []
    for geom in geom_series:
        if geom is None or geom.is_empty:
            local_polys.append(None)
            continue
        coords = np.array(geom.exterior.coords)[:, :2]
        coords[:, 0] -= origin[0]
        coords[:, 1] -= origin[1]
        local_polys.append(coords)
    return local_polys


def build_map_features(gpkg_data, origin):
    """
    Convert gpkg vector layers to MetaDrive map_features dict.

    Args:
        gpkg_data: dict of GeoDataFrames from query_gpkg
        origin: (x, y) UTM origin for local coordinate conversion

    Returns:
        dict[str, dict] — MetaDrive map_features
    """
    map_features = {}

    baselines = gpkg_data["baselines"]
    lanes = gpkg_data["lanes"]
    connectors = gpkg_data["connectors"]
    boundaries = gpkg_data["boundaries"]

    # --- Build lane fid → row index mapping ---
    lane_by_fid = {}
    for idx, row in lanes.iterrows():
        lane_fid = row.get("lane_fid")
        if lane_fid is not None:
            lane_by_fid[int(lane_fid)] = idx

    # --- Build baseline_path lane_fid → row mapping ---
    # Separate lane baselines from connector baselines
    lane_baselines = baselines[baselines["lane_fid"].notna()].copy()
    conn_baselines = baselines[baselines["lane_connector_fid"].notna()].copy()

    bl_by_lane_fid = {}
    for idx, row in lane_baselines.iterrows():
        fid = int(row["lane_fid"])
        bl_by_lane_fid[fid] = idx

    bl_by_conn_fid = {}
    for idx, row in conn_baselines.iterrows():
        fid = int(row["lane_connector_fid"])
        bl_by_conn_fid[fid] = idx

    # --- Connector metadata: entry/exit lanes ---
    # fid is the GeoDataFrame index (from fid_as_index=True)
    conn_entry_exit = {}
    for conn_fid, row in connectors.iterrows():
        entry = row.get("entry_lane_fid")
        exit_ = row.get("exit_lane_fid")
        conn_entry_exit[int(conn_fid)] = {
            "entry": int(entry) if entry is not None and not (isinstance(entry, float) and np.isnan(entry)) else None,
            "exit": int(exit_) if exit_ is not None and not (isinstance(exit_, float) and np.isnan(exit_)) else None,
        }

    # --- Determine lane neighbors ---
    # Group lanes by lane_group_fid to find left/right neighbors
    lane_groups = defaultdict(list)
    for idx, row in lanes.iterrows():
        lg_fid = row.get("lane_group_fid")
        lane_fid = row.get("lane_fid")
        if lg_fid is not None and lane_fid is not None:
            lane_groups[int(lg_fid)].append((int(lane_fid), int(row.get("lane_index", 0))))

    # Sort by lane_index within each group → left/right ordering
    lane_neighbors = {}
    for lg_fid, members in lane_groups.items():
        members.sort(key=lambda x: x[1])
        for i, (fid, _) in enumerate(members):
            left_fid = members[i - 1][0] if i > 0 else None
            right_fid = members[i + 1][0] if i < len(members) - 1 else None
            lane_neighbors[fid] = {"left": left_fid, "right": right_fid}

    # --- Build entry/exit lane connectivity ---
    # For each lane, find which connectors feed into it (entry) or out of it (exit)
    lane_entries = defaultdict(list)
    lane_exits = defaultdict(list)
    for conn_fid, info in conn_entry_exit.items():
        if info["exit"] is not None:
            lane_entries[info["exit"]].append(conn_fid)
        if info["entry"] is not None:
            lane_exits[info["entry"]].append(conn_fid)

    # --- Convert lane baselines to local coords ---
    lane_local_coords = utm_to_local(lane_baselines.geometry, origin)
    # Build lane_fid → polygon mapping
    lane_poly_map = {}
    for lane_fid, row in lanes.iterrows():
        lane_poly_map[int(lane_fid)] = row.geometry
    lane_local_polys = []
    for _, row in lane_baselines.iterrows():
        lf = int(row["lane_fid"])
        geom = lane_poly_map.get(lf)
        if geom is not None and not geom.is_empty:
            coords = np.array(geom.exterior.coords)[:, :2]
            coords[:, 0] -= origin[0]
            coords[:, 1] -= origin[1]
            lane_local_polys.append(coords)
        else:
            lane_local_polys.append(None)

    # --- Build LANE_SURFACE_STREET features ---
    lane_ids_used = set()
    for i, (idx, row) in enumerate(lane_baselines.iterrows()):
        lane_fid = int(row["lane_fid"])
        lane_id = f"lane_{lane_fid}"
        lane_ids_used.add(lane_fid)

        coords = lane_local_coords[i]
        if coords is None or len(coords) < 2:
            continue

        # Ensure 3D polyline (MetaDrive expects x, y, z)
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(len(coords))])

        feature = {
            "type": "LANE_SURFACE_STREET",
            "polyline": coords,
            "entry_lanes": [],
            "exit_lanes": [],
            "left_neighbor": [],
            "right_neighbor": [],
        }

        # Add polygon if available
        if i < len(lane_local_polys) and lane_local_polys[i] is not None:
            feature["polygon"] = lane_local_polys[i]

        # Add neighbor info
        nb = lane_neighbors.get(lane_fid, {})
        if nb.get("left") is not None and f"lane_{nb['left']}" in map_features:
            feature["left_neighbor"] = [f"lane_{nb['left']}"]
        if nb.get("right") is not None:
            feature["right_neighbor"] = [f"lane_{nb['right']}"]

        map_features[lane_id] = feature

    # --- Build connector lanes (LANE_SURFACE_UNSTRUCTURE) ---
    conn_local_coords = utm_to_local(conn_baselines.geometry, origin)
    for i, (idx, row) in enumerate(conn_baselines.iterrows()):
        conn_fid = int(row["lane_connector_fid"])
        conn_id = f"conn_{conn_fid}"

        coords = conn_local_coords[i]
        if coords is None or len(coords) < 2:
            continue

        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(len(coords))])

        info = conn_entry_exit.get(conn_fid, {})
        feature = {
            "type": "LANE_SURFACE_UNSTRUCTURE",
            "polyline": coords,
            "entry_lanes": [],
            "exit_lanes": [],
        }

        # Connect: entry_lane → this connector, and this connector → exit_lane
        entry_lane_fid = info.get("entry")
        exit_lane_fid = info.get("exit")
        if entry_lane_fid is not None:
            feature["entry_lanes"] = [f"lane_{entry_lane_fid}"]
        if exit_lane_fid is not None:
            feature["exit_lanes"] = [f"lane_{exit_lane_fid}"]

        map_features[conn_id] = feature

    # --- Back-fill entry/exit for lane features from connectors ---
    for conn_fid, info in conn_entry_exit.items():
        conn_id = f"conn_{conn_fid}"
        if conn_id not in map_features:
            continue
        # exit_lane → connector
        if info.get("exit") is not None:
            lane_id = f"lane_{info['exit']}"
            if lane_id in map_features:
                map_features[lane_id]["entry_lanes"].append(conn_id)
        # connector → entry_lane
        if info.get("entry") is not None:
            lane_id = f"lane_{info['entry']}"
            if lane_id in map_features:
                map_features[lane_id]["exit_lanes"].append(conn_id)

    # --- Build boundary lines (ROAD_LINE_SOLID_SINGLE_WHITE etc.) ---
    bound_local_coords = utm_to_local(boundaries.geometry, origin)
    for i, (idx, row) in enumerate(boundaries.iterrows()):
        coords = bound_local_coords[i]
        if coords is None or len(coords) < 2:
            continue

        bound_type = int(row.get("boundary_type_fid", 0))
        # Map boundary types to MetaDrive types
        # nuPlan boundary_type_fid: 0=curb, 1=line, 2=wall, etc.
        if bound_type == 0:
            md_type = "ROAD_LINE_SOLID_SINGLE_WHITE"
        else:
            md_type = "ROAD_LINE_SOLID_SINGLE_WHITE"

        bound_id = f"boundary_{idx}"
        map_features[bound_id] = {
            "type": md_type,
            "polyline": coords[:, :2],
        }

    # --- Convert numpy arrays to lists for pickling ---
    for fid, feat in map_features.items():
        for key in ("polyline", "polygon"):
            if key in feat and feat[key] is not None:
                feat[key] = np.array(feat[key]).tolist()

    return map_features


def convert_scene(pkl_path, gpkg_path, epsg=32611, buffer_m=150):
    """
    Full conversion: NAVSIM scene + gpkg → MetaDrive map_features.

    Returns:
        map_features: dict
        origin: (x, y) UTM origin
        ego_positions_local: np.array (N, 2) ego trajectory in local coords
        frames: raw NAVSIM frames
    """
    ego_positions, ego_headings, frames = load_navsim_ego(pkl_path)
    origin = ego_positions[0]  # Use ego start as local origin

    gpkg_data = query_gpkg(gpkg_path, ego_positions, buffer_m=buffer_m, target_epsg=epsg)
    map_features = build_map_features(gpkg_data, origin)

    ego_local = ego_positions - origin

    return map_features, origin, ego_local, ego_headings, frames


def batch_convert(data_dir, gpkg_dir, output_dir, epsg_map=None):
    """
    Convert all NAVSIM mini scenes.

    Args:
        data_dir: path to mini_navsim_logs/mini/
        gpkg_dir: path to nuplan-maps-v1.0/ (contains city subdirs)
        output_dir: where to save converted PKLs
        epsg_map: dict mapping map_location → EPSG code
    """
    if epsg_map is None:
        epsg_map = {
            "us-nv-las-vegas-strip": 32611,
            "us-ma-boston": 32619,
            "sg-one-north": 32648,
            "us-pa-pittsburgh-hazelwood": 32617,
        }

    os.makedirs(output_dir, exist_ok=True)

    pkl_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
    print(f"Found {len(pkl_files)} scenes")

    # Group by map_location → gpkg path
    gpkg_paths = {
        "us-nv-las-vegas-strip": os.path.join(
            gpkg_dir, "us-nv-las-vegas-strip/9.15.1915/map.gpkg"
        ),
        "us-ma-boston": os.path.join(gpkg_dir, "us-ma-boston/9.12.1817/map.gpkg"),
        "sg-one-north": os.path.join(gpkg_dir, "sg-one-north/9.17.1964/map.gpkg"),
        "us-pa-pittsburgh-hazelwood": os.path.join(
            gpkg_dir, "us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg"
        ),
    }

    stats = defaultdict(int)
    for pf in pkl_files:
        pkl_path = os.path.join(data_dir, pf)

        # Load scene to get map_location
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        map_loc = frames[0].get("map_location", "")
        stats[map_loc] += 1

        # Find gpkg
        gpkg_path = gpkg_paths.get(map_loc)
        if gpkg_path is None or not os.path.exists(gpkg_path):
            print(f"  SKIP {pf}: no gpkg for {map_loc}")
            continue

        epsg = epsg_map.get(map_loc, 32611)

        try:
            map_features, origin, ego_local, ego_headings, _ = convert_scene(
                pkl_path, gpkg_path, epsg=epsg
            )

            out_path = os.path.join(output_dir, pf.replace(".pkl", "_map.pkl"))
            with open(out_path, "wb") as f:
                pickle.dump(
                    {
                        "map_features": map_features,
                        "origin": origin.tolist(),
                        "ego_local": ego_local.tolist(),
                        "ego_headings": ego_headings.tolist(),
                        "map_location": map_loc,
                    },
                    f,
                )

            n_lanes = sum(
                1 for v in map_features.values() if "LANE_SURFACE" in v.get("type", "")
            )
            print(f"  {pf}: {map_loc}, {len(map_features)} features ({n_lanes} lanes)")
        except Exception as e:
            print(f"  ERROR {pf}: {e}")

    print(f"\nLocation distribution: {dict(stats)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert nuPlan gpkg to MetaDrive map_features")
    sub = parser.add_subparsers(dest="cmd")

    # Single scene
    p_single = sub.add_parser("convert")
    p_single.add_argument("--pkl", required=True, help="NAVSIM scene PKL")
    p_single.add_argument("--gpkg", required=True, help="nuPlan map.gpkg path")
    p_single.add_argument("--epsg", type=int, default=32611, help="UTM EPSG code")
    p_single.add_argument("--output", required=True, help="Output PKL path")

    # Batch convert
    p_batch = sub.add_parser("batch")
    p_batch.add_argument("--data_dir", required=True, help="mini_navsim_logs/mini/")
    p_batch.add_argument("--gpkg_dir", required=True, help="nuplan-maps-v1.0/")
    p_batch.add_argument("--output_dir", required=True, help="Output directory")

    args = parser.parse_args()

    if args.cmd == "convert":
        map_features, origin, ego_local, ego_headings, _ = convert_scene(
            args.pkl, args.gpkg, epsg=args.epsg
        )
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "wb") as f:
            pickle.dump(
                {
                    "map_features": map_features,
                    "origin": origin.tolist(),
                    "ego_local": ego_local.tolist(),
                    "ego_headings": ego_headings.tolist(),
                },
                f,
            )
        n_lanes = sum(
            1 for v in map_features.values() if "LANE_SURFACE" in v.get("type", "")
        )
        print(f"Saved: {len(map_features)} features ({n_lanes} lanes) → {args.output}")

    elif args.cmd == "batch":
        batch_convert(args.data_dir, args.gpkg_dir, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
