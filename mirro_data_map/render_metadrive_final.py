"""
Render MetaDrive's final map features (exactly what gets rendered in BEV) as a BEV image.

Includes: lane polygons, junction polygons, lane dividers, edge dividers.
This is what MetaDrive actually draws - the ground truth for visual comparison.
"""
import sys, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
SUMO_HOME = os.environ.get("SUMO_HOME",
    "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
os.environ["SUMO_HOME"] = SUMO_HOME

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType


def render_metadrive_map(net_xml_path, out_path, title=None):
    print(f"Processing: {net_xml_path}")
    graph = RoadLaneJunctionGraph(net_xml_path)
    features = extract_map_features(graph)

    fig, ax = plt.subplots(1, 1, figsize=(40, 25))

    # Colors
    LANE_COLOR = '#808080'       # Road surface (gray)
    JUNCTION_COLOR = '#a0a0a0'   # Junction surface (lighter gray)
    SIDEWALK_COLOR = '#c8c8c8'   # Sidewalk
    LANE_DIVIDER_COLOR = '#ffffff'  # White dashed
    EDGE_DIVIDER_COLOR = '#ffd700'  # Yellow solid
    CROSSWALK_COLOR = '#e0e0e0'

    # Collect all geometry
    lane_patches = []
    junction_patches = []
    sidewalk_patches = []
    crosswalk_patches = []
    lane_divider_lines = []
    edge_divider_lines = []
    all_points = []

    for fid, feat in features.items():
        ftype = feat.get(SD.TYPE, '')
        polygon = feat.get(SD.POLYGON, None)
        polyline = feat.get(SD.POLYLINE, None)

        if fid.startswith("lane_divider_"):
            if polyline is not None and len(polyline) >= 2:
                pts = np.array(polyline)
                lane_divider_lines.append(pts)
                all_points.append(pts)

        elif fid.startswith("edge_divider_"):
            if polyline is not None and len(polyline) >= 2:
                pts = np.array(polyline)
                edge_divider_lines.append(pts)
                all_points.append(pts)

        elif fid.startswith("lane_"):
            if ftype == MetaDriveType.BOUNDARY_SIDEWALK:
                if polygon and len(polygon) >= 3:
                    pts = np.array(polygon)
                    sidewalk_patches.append(MplPolygon(pts, closed=True))
                    all_points.append(pts)
            elif ftype == MetaDriveType.CROSSWALK:
                if polygon and len(polygon) >= 3:
                    pts = np.array(polygon)
                    crosswalk_patches.append(MplPolygon(pts, closed=True))
                    all_points.append(pts)
            else:  # LANE_SURFACE_STREET and others
                if polygon and len(polygon) >= 3:
                    pts = np.array(polygon)
                    lane_patches.append(MplPolygon(pts, closed=True))
                    all_points.append(pts)

        elif fid.startswith("junction_"):
            if polygon and len(polygon) >= 3:
                pts = np.array(polygon)
                junction_patches.append(MplPolygon(pts, closed=True))
                all_points.append(pts)

    # Draw order: junctions first, then lanes, then markings on top
    if junction_patches:
        pc = PatchCollection(junction_patches, facecolor=JUNCTION_COLOR, edgecolor='none', alpha=0.8)
        ax.add_collection(pc)
        print(f"  Junctions: {len(junction_patches)}")

    if sidewalk_patches:
        pc = PatchCollection(sidewalk_patches, facecolor=SIDEWALK_COLOR, edgecolor='none', alpha=0.6)
        ax.add_collection(pc)
        print(f"  Sidewalks: {len(sidewalk_patches)}")

    if lane_patches:
        pc = PatchCollection(lane_patches, facecolor=LANE_COLOR, edgecolor='#606060', linewidth=0.3, alpha=0.9)
        ax.add_collection(pc)
        print(f"  Lanes: {len(lane_patches)}")

    if crosswalk_patches:
        pc = PatchCollection(crosswalk_patches, facecolor=CROSSWALK_COLOR, edgecolor='none', alpha=0.5)
        ax.add_collection(pc)
        print(f"  Crosswalks: {len(crosswalk_patches)}")

    # Lane dividers (white, dashed in MetaDrive)
    if lane_divider_lines:
        segments = [np.array(line) for line in lane_divider_lines if len(line) >= 2]
        lc = LineCollection(segments, colors=LANE_DIVIDER_COLOR, linewidths=0.8, alpha=0.7,
                           linestyles='dashed')
        ax.add_collection(lc)
        print(f"  Lane dividers: {len(lane_divider_lines)}")

    # Edge dividers (yellow, solid in MetaDrive)
    if edge_divider_lines:
        segments = [np.array(line) for line in edge_divider_lines if len(line) >= 2]
        lc = LineCollection(segments, colors=EDGE_DIVIDER_COLOR, linewidths=1.0, alpha=0.8)
        ax.add_collection(lc)
        print(f"  Edge dividers: {len(edge_divider_lines)}")

    # Set limits
    if all_points:
        all_pts = np.vstack(all_points)
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        margin = max(x_max - x_min, y_max - y_min) * 0.02
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_aspect('equal')
    ax.set_facecolor('#2d2d2d')
    ax.set_title(title or "MetaDrive Rendered Map (Final)", fontsize=18, color='white')
    fig.patch.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default=os.path.join(os.path.dirname(__file__), "exid_loc4_orig.net.xml"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    out = args.out or os.path.join(base,
        "metadrive_rendered_" + os.path.basename(args.net).replace(".net.xml", ".png"))
    render_metadrive_map(args.net, out)
