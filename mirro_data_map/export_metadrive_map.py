"""
Export MetaDrive-rendered map as a clean SUMO .net.xml.

Apply ALL MetaDrive filtering to the XML itself:
- Remove walkingarea/crossing edges
- Remove filtered junctions (bidirectional check)
- Remove non-rendered lanes (sidewalk, shoulder, etc.)
- Keep internal edges (with their lanes)
- Add computed lane/edge dividers as edges
- Add junction polygons that MetaDrive renders
"""
import sys, os, shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

SUMO_HOME = os.environ.get("SUMO_HOME",
    "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
os.environ["SUMO_HOME"] = SUMO_HOME

import sumolib
import numpy as np
import xml.etree.ElementTree as ET
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
from metadrive.type import MetaDriveType as MDT


def export(net_xml_path, out_path):
    print(f"Loading: {net_xml_path}")
    orig_net = sumolib.net.readNet(net_xml_path, withInternal=True, withPedestrianConnections=True)

    # Process through MetaDrive pipeline
    print("Processing through MetaDrive pipeline...")
    graph = RoadLaneJunctionGraph(net_xml_path)
    features = extract_map_features(graph)

    # Collect what MetaDrive renders
    rendered_lanes = set()   # lane IDs that become LANE_SURFACE_STREET
    rendered_junctions = set()
    for fid, feat in features.items():
        ftype = feat.get('type', '')
        if fid.startswith("junction_"):
            rendered_junctions.add(fid[9:])
        elif fid.startswith("lane_"):
            rendered_lanes.add(fid[5:])

    print(f"Rendered: {len(rendered_lanes)} lanes, {len(rendered_junctions)} junctions")
    print(f"Dividers: {len(graph.lane_dividers)} lane, {len(graph.edge_dividers)} edge")

    # Copy original and modify
    shutil.copy2(net_xml_path, out_path)
    tree = ET.parse(out_path)
    root = tree.getroot()

    # === 1. Remove walkingarea/crossing edges ===
    for edge_elem in root.findall('edge'):
        func = edge_elem.get('function', '')
        if func in ('walkingarea', 'crossing'):
            root.remove(edge_elem)

    # === 2. Remove non-rendered lanes from edges ===
    for edge_elem in root.findall('edge'):
        func = edge_elem.get('function', '')
        to_remove = []
        for lane_elem in edge_elem.findall('lane'):
            lid = lane_elem.get('id')
            # Keep internal edge lanes as-is (MetaDrive renders them)
            if func == 'internal':
                continue
            # Remove lane if MetaDrive doesn't render it
            if lid not in rendered_lanes:
                to_remove.append(lane_elem)
        for lane_elem in to_remove:
            edge_elem.remove(lane_elem)
        # Remove edge if all lanes removed
        if len(edge_elem.findall('lane')) == 0 and func != 'internal':
            root.remove(edge_elem)

    # === 3. Keep all junction nodes (edges reference them), clear shape for non-rendered ===
    for junction_elem in root.findall('junction'):
        jid = junction_elem.get('id')
        if jid not in rendered_junctions:
            # Don't remove the node, just clear the polygon shape
            junction_elem.set('shape', '')

    # === 4. Remove connections that reference removed lanes/edges ===
    # Keep connections only if both from-lane and to-lane still exist
    remaining_lanes = set()
    for edge_elem in root.findall('edge'):
        for lane_elem in edge_elem.findall('lane'):
            remaining_lanes.add(lane_elem.get('id'))

    for connection_elem in root.findall('connection'):
        from_lane = connection_elem.get('from', '')
        to_lane = connection_elem.get('to', '')
        via = connection_elem.get('via', '')
        # Check from/to edges exist
        from_edge_lanes = [l.get('id') for l in root.findall('.//edge[@id="{}"]/lane'.format(from_lane))]
        if from_lane not in [e.get('id') for e in root.findall('edge')] or \
           to_lane not in [e.get('id') for e in root.findall('edge')]:
            root.remove(connection_elem)
            continue
        if via and via not in remaining_lanes:
            root.remove(connection_elem)

    # === 5. Add junction polygons that MetaDrive renders ===
    for junction_id, junction in graph.junctions.items():
        if junction_id not in rendered_junctions:
            continue
        if len(junction.shape) <= 2:
            continue
        shape_str = " ".join(f"{p[0]:.4f},{p[1]:.4f}" for p in junction.shape)
        # Find existing junction element and update its shape
        for junction_elem in root.findall('junction'):
            if junction_elem.get('id') == junction_id:
                junction_elem.set('shape', shape_str)
                break

    # === 6. Add computed dividers as edges ===
    for i, div in enumerate(graph.lane_dividers):
        pts = np.array(div)
        if len(pts) < 2:
            continue
        shape_str = " ".join(f"{p[0]:.4f},{p[1]:.4f}" for p in pts)
        edge = ET.SubElement(root, "edge",
            id=f"__md_lane_div_{i}", function="internal")
        ET.SubElement(edge, "lane",
            id=f"__md_lane_div_{i}_0", index="0",
            speed="0", length="0", width="0.15",
            shape=shape_str)

    for i, div in enumerate(graph.edge_dividers):
        pts = np.array(div)
        if len(pts) < 2:
            continue
        shape_str = " ".join(f"{p[0]:.4f},{p[1]:.4f}" for p in pts)
        edge = ET.SubElement(root, "edge",
            id=f"__md_edge_div_{i}", function="internal")
        ET.SubElement(edge, "lane",
            id=f"__md_edge_div_{i}_0", index="0",
            speed="0", length="0", width="0.15",
            shape=shape_str)

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"\nExported: {out_path}")
    print(f"  Removed non-rendered lanes, junctions, walkingarea edges")
    print(f"  Added {len(graph.lane_dividers)} lane dividers, {len(graph.edge_dividers)} edge dividers")
    print(f"  Junctions rendered: {len(rendered_junctions)}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadrive_rendered_maps")
    os.makedirs(out_dir, exist_ok=True)
    net = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exid_loc4_orig.net.xml")
    out = os.path.join(out_dir, "exid_loc4_orig_metadrive_final.net.xml")
    export(net, out)
