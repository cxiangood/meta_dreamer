"""
Convert nuPlan GeoPackage map to SUMO .net.xml via OSM intermediate format.

Pipeline: gpkg → OSM XML → netconvert → .net.xml → MetaDrive SumoMapManager

Usage:
    # Single scene
    python dreamer/tools/gpkg_to_sumo.py convert \
        --gpkg /path/to/map.gpkg \
        --center_x 664898 --center_y 3999658 \
        --buffer 200 \
        --epsg 32611 \
        --output logs/sumo_maps/scene_001

    # Batch all NAVSIM mini scenes
    python dreamer/tools/gpkg_to_sumo.py batch \
        --data_dir /share/home/u23516/code/navsim_mini/mini_navsim_logs/mini/ \
        --gpkg_dir /share/home/u23516/code/navsim_mini/maps_download/nuplan-maps-v1.0/ \
        --output_dir logs/sumo_maps/
"""

import argparse
import os
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box


# ---------------------------------------------------------------------------
# gpkg → OSM conversion
# ---------------------------------------------------------------------------

class OSMBuilder:
    """Build OSM XML from nuPlan gpkg vector layers."""

    def __init__(self, epsg=32611):
        self.nodes = []  # (id, lon, lat)
        self.ways = []   # (id, node_refs, tags)
        self.relations = []  # (id, members, tags)
        self._node_id = 0
        self._way_id = 0
        self._rel_id = 0
        self._node_coords = {}  # (x,y) → node_id  (dedup)
        # UTM → WGS84 transformer
        from pyproj import Transformer
        self._to_wgs84 = Transformer.from_crs(
            f"EPSG:{epsg}", "EPSG:4326", always_xy=True
        )

    def _next_node_id(self):
        self._node_id -= 1
        return self._node_id + 1  # negative ids

    def _next_way_id(self):
        self._way_id -= 1
        return self._way_id + 1

    def _next_rel_id(self):
        self._rel_id -= 1
        return self._rel_id + 1

    def _add_node(self, x, y):
        """Add a node. Convert UTM (x,y) to WGS84 (lon,lat)."""
        key = (round(x, 1), round(y, 1))
        if key in self._node_coords:
            return self._node_coords[key]
        nid = self._next_node_id()
        lon, lat = self._to_wgs84.transform(x, y)
        self.nodes.append((nid, lon, lat))
        self._node_coords[key] = nid
        return nid

    def add_road(self, line_coords, tags=None):
        """Add a road as OSM way from a LineString coordinate array."""
        if tags is None:
            tags = {}
        tags.setdefault("highway", "motorway")

        node_refs = []
        for pt in line_coords:
            nid = self._add_node(pt[0], pt[1])
            node_refs.append(nid)

        if len(node_refs) < 2:
            return None

        wid = self._next_way_id()
        self.ways.append((wid, node_refs, tags))
        return wid

    def to_xml(self, output_path):
        """Write OSM XML file."""
        osm = ET.Element("osm", version="0.6", generator="gpkg_to_osm")

        # Bounds (approximate)
        if self.nodes:
            xs = [n[1] for n in self.nodes]
            ys = [n[2] for n in self.nodes]
            ET.SubElement(osm, "bounds",
                          minlat=str(min(ys)), minlon=str(min(xs)),
                          maxlat=str(max(ys)), maxlon=str(max(xs)))

        # Nodes
        for nid, x, y in self.nodes:
            node_el = ET.SubElement(osm, "node",
                                    id=str(nid),
                                    lon=str(x),
                                    lat=str(y),
                                    visible="true")

        # Ways
        for wid, node_refs, tags in self.ways:
            way_el = ET.SubElement(osm, "way",
                                   id=str(wid), visible="true")
            for ref in node_refs:
                ET.SubElement(way_el, "nd", ref=str(ref))
            for k, v in tags.items():
                ET.SubElement(way_el, "tag", k=k, v=v)

        # Relations
        for rid, members, tags in self.relations:
            rel_el = ET.SubElement(osm, "relation",
                                   id=str(rid), visible="true")
            for mtype, mref, mrole in members:
                ET.SubElement(rel_el, "member",
                              type=mtype, ref=str(mref), role=mrole)
            for k, v in tags.items():
                ET.SubElement(rel_el, "tag", k=k, v=v)

        tree = ET.ElementTree(osm)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return len(self.nodes), len(self.ways)


def gpkg_to_osm(gpkg_path, center_x, center_y, buffer_m=200, epsg=32611):
    """
    Extract road network from gpkg near a center point and build OSM.

    Returns: OSMBuilder
    """
    bbox = box(center_x - buffer_m, center_y - buffer_m,
               center_x + buffer_m, center_y + buffer_m)

    # Load layers
    baselines = gpd.read_file(
        gpkg_path, layer="baseline_paths", engine="pyogrio", fid_as_index=True
    ).to_crs(f"EPSG:{epsg}")
    baselines = baselines[baselines.geometry.intersects(bbox)]

    lanes = gpd.read_file(
        gpkg_path, layer="lanes_polygons", engine="pyogrio", fid_as_index=True
    ).to_crs(f"EPSG:{epsg}")
    lanes = lanes[lanes.geometry.intersects(bbox)]

    connectors = gpd.read_file(
        gpkg_path, layer="lane_connectors", engine="pyogrio", fid_as_index=True
    ).to_crs(f"EPSG:{epsg}")
    connectors = connectors[connectors.geometry.intersects(bbox)]

    print(f"  baselines: {len(baselines)}, lanes: {len(lanes)}, "
          f"connectors: {len(connectors)}")

    # Build lane_fid → lane info mapping
    lane_info = {}
    for idx, row in lanes.iterrows():
        fid = int(row.get("lane_fid", idx))
        lane_info[fid] = {
            "lane_index": int(row.get("lane_index", 0)),
            "lane_group_fid": int(row.get("lane_group_fid", 0)),
            "speed_limit": float(row.get("speed_limit_mps", 15)),
            "width": float(row.get("width", 3.5)),
        }

    # Separate lane baselines from connector baselines
    lane_bls = baselines[baselines["lane_fid"].notna()].copy()
    conn_bls = baselines[baselines["lane_connector_fid"].notna()].copy()

    # Build OSM
    osm = OSMBuilder(epsg=epsg)

    # Add lane baselines as OSM ways (keep original UTM coords, OSMBuilder converts to WGS84)
    for idx, row in lane_bls.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            continue
        coords = np.array(geom.coords)

        lane_fid = int(row["lane_fid"])
        info = lane_info.get(lane_fid, {})
        n_lanes = 1  # default single lane per baseline

        tags = {
            "highway": "primary",
            "lanes": str(n_lanes),
            "name": f"lane_{lane_fid}",
        }
        if info.get("speed_limit"):
            tags["maxspeed"] = str(info["speed_limit"])

        osm.add_road(coords, tags)

    # Add connector baselines (keep UTM coords)
    for idx, row in conn_bls.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            continue
        coords = np.array(geom.coords)

        conn_fid = int(row["lane_connector_fid"])
        tags = {
            "highway": "primary",
            "name": f"conn_{conn_fid}",
        }
        osm.add_road(coords, tags)

    return osm


# ---------------------------------------------------------------------------
# OSM → .net.xml via netconvert
# ---------------------------------------------------------------------------

def osm_to_net_xml(osm_path, net_xml_path):
    """Convert OSM to SUMO .net.xml using netconvert."""
    cmd = [
        "netconvert",
        "--osm-files", osm_path,
        "--output-file", net_xml_path,
        "--remove-edges.is-railway", "true",
        "--remove-edges.is-water", "true",
        "--lefthand", "false",
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--no-internal-links",
        "--default.lanenumber", "1",
        "--default.speed", "15",
        "--default.width", "3.5",
    ]

    print(f"  Running netconvert...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"  netconvert stderr: {result.stderr[:500]}")
    if os.path.exists(net_xml_path):
        size = os.path.getsize(net_xml_path)
        print(f"  net.xml generated: {size} bytes")
        return True
    else:
        print(f"  netconvert failed!")
        return False


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def convert_scene(pkl_path, gpkg_path, epsg=32611, buffer_m=200, output_dir=None):
    """Convert a NAVSIM scene's map to SUMO .net.xml."""
    import pickle

    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)

    # Get ego position
    positions = np.array([fr["ego2global_translation"][:2] for fr in frames])
    cx, cy = positions[0].mean(axis=0), positions[0].mean(axis=0)
    # Use center of trajectory for better map coverage
    cx = (positions[:, 0].min() + positions[:, 0].max()) / 2
    cy = (positions[:, 1].min() + positions[:, 1].max()) / 2

    # Expand buffer to cover full trajectory
    traj_range = max(positions[:, 0].max() - positions[:, 0].min(),
                     positions[:, 1].max() - positions[:, 1].min()) / 2
    buf = max(buffer_m, traj_range + 50)

    scene_name = Path(pkl_path).stem
    if output_dir is None:
        output_dir = f"logs/sumo_maps/{scene_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {scene_name}...")
    print(f"  Center: ({cx:.0f}, {cy:.0f}), buffer: {buf:.0f}m")

    # Step 1: gpkg → OSM
    osm_path = os.path.join(output_dir, f"{scene_name}.osm")
    osm = gpkg_to_osm(gpkg_path, cx, cy, buffer_m=buf, epsg=epsg)
    n_nodes, n_ways = osm.to_xml(osm_path)
    print(f"  OSM: {n_nodes} nodes, {n_ways} ways → {osm_path}")

    # Step 2: OSM → .net.xml
    net_path = os.path.join(output_dir, f"{scene_name}.net.xml")
    success = osm_to_net_xml(osm_path, net_path)

    if success:
        print(f"  DONE: {net_path}")
    return net_path if success else None


def batch_convert(data_dir, gpkg_dir, output_dir, epsg_map=None):
    """Convert all NAVSIM mini scenes to .net.xml."""
    if epsg_map is None:
        epsg_map = {
            "us-nv-las-vegas-strip": 32611,
            "us-ma-boston": 32619,
            "sg-one-north": 32648,
            "us-pa-pittsburgh-hazelwood": 32617,
        }

    gpkg_paths = {
        "us-nv-las-vegas-strip": os.path.join(
            gpkg_dir, "us-nv-las-vegas-strip/9.15.1915/map.gpkg"),
        "us-ma-boston": os.path.join(gpkg_dir, "us-ma-boston/9.12.1817/map.gpkg"),
        "sg-one-north": os.path.join(gpkg_dir, "sg-one-north/9.17.1964/map.gpkg"),
        "us-pa-pittsburgh-hazelwood": os.path.join(
            gpkg_dir, "us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg"),
    }

    import pickle
    pkl_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
    print(f"Found {len(pkl_files)} scenes")

    stats = {"success": 0, "failed": 0, "skipped": 0}
    for pf in pkl_files:
        pkl_path = os.path.join(data_dir, pf)
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        map_loc = frames[0].get("map_location", "")

        gpkg_path = gpkg_paths.get(map_loc)
        if not gpkg_path or not os.path.exists(gpkg_path):
            print(f"  SKIP {pf}: no gpkg for {map_loc}")
            stats["skipped"] += 1
            continue

        epsg = epsg_map.get(map_loc, 32611)
        scene_dir = os.path.join(output_dir, pf.replace(".pkl", ""))

        try:
            result = convert_scene(pkl_path, gpkg_path, epsg=epsg,
                                   buffer_m=150, output_dir=scene_dir)
            if result:
                stats["success"] += 1
            else:
                stats["failed"] += 1
        except Exception as e:
            print(f"  ERROR {pf}: {e}")
            stats["failed"] += 1

    print(f"\nBatch stats: {stats}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert nuPlan gpkg to SUMO .net.xml")
    sub = parser.add_subparsers(dest="cmd")

    # Single scene
    p1 = sub.add_parser("convert")
    p1.add_argument("--pkl", required=True, help="NAVSIM scene PKL")
    p1.add_argument("--gpkg", required=True, help="nuPlan map.gpkg path")
    p1.add_argument("--epsg", type=int, default=32611)
    p1.add_argument("--buffer", type=int, default=200)
    p1.add_argument("--output", default=None)

    # Direct gpkg → OSM
    p2 = sub.add_parser("gpkg2osm")
    p2.add_argument("--gpkg", required=True)
    p2.add_argument("--center_x", type=float, required=True)
    p2.add_argument("--center_y", type=float, required=True)
    p2.add_argument("--buffer", type=int, default=200)
    p2.add_argument("--epsg", type=int, default=32611)
    p2.add_argument("--output", required=True)

    # Batch
    p3 = sub.add_parser("batch")
    p3.add_argument("--data_dir", required=True)
    p3.add_argument("--gpkg_dir", required=True)
    p3.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    if args.cmd == "convert":
        convert_scene(args.pkl, args.gpkg, args.epsg, args.buffer, args.output)
    elif args.cmd == "gpkg2osm":
        osm = gpkg_to_osm(args.gpkg, args.center_x, args.center_y,
                          args.buffer, args.epsg)
        n, w = osm.to_xml(args.output)
        print(f"OSM: {n} nodes, {w} ways → {args.output}")
    elif args.cmd == "batch":
        batch_convert(args.data_dir, args.gpkg_dir, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
