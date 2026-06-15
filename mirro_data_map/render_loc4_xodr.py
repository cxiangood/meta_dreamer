"""Render loc4 xodr map as BEV image for visual comparison with SUMO version."""
import sys, os, math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from metadrive.utils.opendrive.map_load import load_opendrive_map
from metadrive.component.opendrive_block.opendrive_block import OpenDriveBlock
from metadrive.component.road_network.edge_road_network import OpenDriveRoadNetwork

XODR = "/Users/jiojio/Downloads/exiD-dataset-v2.1.zip/exiD-dataset-v2.1/maps/opendrive/4_cologne_klettenberg/cologne_klettenberg.xodr"

def extract_lane_polylines(xodr_path):
    """Extract lane centerlines and boundaries from OpenDRIVE map."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xodr_path)
    root = tree.getroot()

    all_polylines = []  # (pts, color, label)

    for road in root.findall('.//road'):
        road_id = road.get('id')
        plan_view = road.find('planView')
        if plan_view is None:
            continue

        # Get geometry
        geometries = []
        for geom in plan_view.findall('geometry'):
            s = float(geom.get('s'))
            x = float(geom.get('x'))
            y = float(geom.get('y'))
            hdg = float(geom.get('hdg'))
            length = float(geom.get('length'))
            line = geom.find('line')
            arc = geom.find('arc')
            if line is not None:
                geometries.append(('line', s, x, y, hdg, length))
            elif arc is not None:
                curvature = float(arc.get('curvature'))
                geometries.append(('arc', s, x, y, hdg, length, curvature))

        if not geometries:
            continue

        # Sample centerline
        center_pts = []
        for geom in geometries:
            if geom[0] == 'line':
                _, s0, x0, y0, hdg, length = geom
                for ds in np.linspace(0, length, max(int(length / 2), 2)):
                    center_pts.append([
                        x0 + ds * math.cos(hdg),
                        y0 + ds * math.sin(hdg)
                    ])
            elif geom[0] == 'arc':
                _, s0, x0, y0, hdg, length, curvature = geom
                r = 1.0 / curvature if abs(curvature) > 1e-6 else 1e6
                # Arc center
                cx = x0 - r * math.sin(hdg)
                cy = y0 + r * math.cos(hdg)
                for ds in np.linspace(0, length, max(int(length / 2), 2)):
                    angle = hdg + curvature * ds
                    center_pts.append([
                        x0 + ds * math.cos(hdg) if abs(curvature) < 1e-6 else cx + r * math.sin(angle),
                        y0 + ds * math.sin(hdg) if abs(curvature) < 1e-6 else cy - r * math.cos(angle)
                    ])

        if center_pts:
            all_polylines.append((np.array(center_pts), '#4a90d9', f'road_{road_id}'))

    return all_polylines


def extract_lanes_simple(xodr_path):
    """Simple approach: parse road reference line and lane offsets."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xodr_path)
    root = tree.getroot()

    road_polylines = []
    lane_boundaries = []

    for road in root.findall('.//road'):
        road_id = road.get('id')
        plan_view = road.find('planView')
        if plan_view is None:
            continue

        # Get plan view geometry segments
        geom_data = []
        for geom in plan_view.findall('geometry'):
            s = float(geom.get('s', 0))
            x = float(geom.get('x', 0))
            y = float(geom.get('y', 0))
            hdg = float(geom.get('hdg', 0))
            length = float(geom.get('length', 0))
            line_elem = geom.find('line')
            arc_elem = geom.find('arc')
            if line_elem is not None:
                geom_data.append({'type': 'line', 's': s, 'x': x, 'y': y, 'hdg': hdg, 'length': length})
            elif arc_elem is not None:
                curvature = float(arc_elem.get('curvature', 0))
                geom_data.append({'type': 'arc', 's': s, 'x': x, 'y': y, 'hdg': hdg,
                                  'length': length, 'curvature': curvature})

        if not geom_data:
            continue

        # Sample reference line at fine intervals
        ref_pts = []
        for g in geom_data:
            n_pts = max(int(g['length'] / 1.0), 2)
            for ds in np.linspace(0, g['length'], n_pts):
                if g['type'] == 'line':
                    px = g['x'] + ds * math.cos(g['hdg'])
                    py = g['y'] + ds * math.sin(g['hdg'])
                    ref_pts.append([g['s'] + ds, px, py, g['hdg']])
                elif g['type'] == 'arc':
                    c = g['curvature']
                    hdg0 = g['hdg']
                    angle = hdg0 + c * ds
                    if abs(c) > 1e-8:
                        r = 1.0 / c
                        cx = g['x'] - r * math.sin(hdg0)
                        cy = g['y'] + r * math.cos(hdg0)
                        px = cx + r * math.sin(angle)
                        py = cy - r * math.cos(angle)
                    else:
                        px = g['x'] + ds * math.cos(hdg0)
                        py = g['y'] + ds * math.sin(hdg0)
                    ref_pts.append([g['s'] + ds, px, py, angle])

        if len(ref_pts) < 2:
            continue

        ref_arr = np.array(ref_pts)

        # Get lane sections
        lanes_elem = road.find('lanes')
        if lanes_elem is None:
            continue

        for lsec in lanes_elem.findall('laneSection'):
            s_start = float(lsec.get('s', 0))

            # Center lane (id=0) has no width
            # Left lanes (id > 0) and right lanes (id < 0)
            for direction in ['left', 'right', 'center']:
                dir_elem = lsec.find(direction)
                if dir_elem is None:
                    continue
                for lane_elem in dir_elem.findall('lane'):
                    lane_id = int(lane_elem.get('id', 0))
                    lane_type = lane_elem.get('type', '')

                    if lane_type in ('none', 'shoulder', 'border', 'sidewalk'):
                        continue

                    # Get width
                    width_elem = lane_elem.find('width')
                    if width_elem is not None:
                        lane_width = float(width_elem.get('a', 0))
                    else:
                        continue

                    if lane_width < 0.5:
                        continue

                    # Compute offset from center
                    # For OpenDRIVE: right lanes (id < 0) are offset to the right
                    # We need to compute cumulative offset
                    # Simple approach: just use this lane's offset
                    if direction == 'right':
                        # Accumulate widths for all right lanes with id >= this lane's id
                        total_offset = 0
                        for other_lane in dir_elem.findall('lane'):
                            oid = int(other_lane.get('id', 0))
                            if oid < 0 and oid >= lane_id:
                                ow = other_lane.find('width')
                                if ow is not None:
                                    total_offset += float(ow.get('a', 0)) / 2 if oid == lane_id else float(ow.get('a', 0))
                        offset = -total_offset  # right = negative y offset in local coords
                    elif direction == 'left':
                        total_offset = 0
                        for other_lane in dir_elem.findall('lane'):
                            oid = int(other_lane.get('id', 0))
                            if oid > 0 and oid <= lane_id:
                                ow = other_lane.find('width')
                                if ow is not None:
                                    total_offset += float(ow.get('a', 0)) / 2 if oid == lane_id else float(ow.get('a', 0))
                        offset = total_offset
                    else:
                        continue

                    # Offset the reference line
                    lane_pts = []
                    for s_val, px, py, hdg in ref_pts:
                        nx = -math.sin(hdg) * offset
                        ny = math.cos(hdg) * offset
                        lane_pts.append([px + nx, py + ny])

                    if lane_type == 'driving':
                        road_polylines.append(np.array(lane_pts))

    return road_polylines


def main():
    print("Loading xodr...")
    road_polylines = extract_lanes_simple(XODR)
    print(f"Found {len(road_polylines)} lane polylines")

    fig, ax = plt.subplots(1, 1, figsize=(30, 20))
    for pts in road_polylines:
        ax.plot(pts[:, 0], pts[:, 1], color='#4a90d9', linewidth=1.5, alpha=0.7)

    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    ax.set_title("loc4 — OpenDRIVE (.xodr) Lane Geometry", fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "loc4_xodr_lanes.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
