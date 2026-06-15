"""
Visualize on-ramp merge zone endpoints for all 7 exiD locations.

For each location:
1. Load the SUMO map + lanelet2 OSM onramp annotations
2. Identify on-ramp SUMO edges
3. Find the merge endpoint (ramp edge's last point → junction → main road)
4. Draw BEV-style map with ramp highlight and merge endpoint markers

Key insight: merge_idx in exid_merge_cache.json is per-VEHICLE (when
each driver chose to change lanes). The geometric merge endpoint is the
fixed point where the ramp lane physically ends at the junction with the main road.

Usage:
    python meta_dreamer_pytorch/visualize_merge_endpoints.py
    python meta_dreamer_pytorch/visualize_merge_endpoints.py --loc 0
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10, "axes.titlesize": 12,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

if "SUMO_HOME" not in os.environ:
    for candidate in [
        "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo",
        "/usr/share/sumo",
        "/share/apps/sumo-1.20",
    ]:
        if os.path.isdir(candidate):
            os.environ["SUMO_HOME"] = candidate
            break
sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib

# Auto-detect paths: local dev vs HPC
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _base in [
    os.path.join(_SCRIPT_DIR, ".."),
    "/share/home/u23516/code/meta_dreamer-main",
]:
    if os.path.isdir(os.path.join(_base, "mirro_data_map")):
        _REPO_ROOT = _base
        break
else:
    _REPO_ROOT = os.path.join(_SCRIPT_DIR, "..")

CACHE_PATH = os.path.join(_REPO_ROOT, "mirro_data_map", "exid_merge_cache.json")
MAP_DIR = os.path.join(_REPO_ROOT, "mirro_data_map")
OUT_DIR = os.path.join(_REPO_ROOT, "docs", "thesis_figures", "merge_endpoints")

# exiD dataset paths
_EXID_CANDIDATES = [
    os.path.join(_REPO_ROOT, "dataset"),
    "/share/home/u23516/data/exiD-dataset-v2.1",
    os.path.expanduser("~/Downloads/exiD-dataset-v2.1"),
]
EXID_DATA_DIR = None
for _c in _EXID_CANDIDATES:
    if os.path.isdir(os.path.join(_c, "data")):
        EXID_DATA_DIR = _c
        break

os.makedirs(OUT_DIR, exist_ok=True)
print(f"REPO_ROOT={_REPO_ROOT}")
print(f"EXID_DATA_DIR={EXID_DATA_DIR}")
print(f"OUT_DIR={OUT_DIR}")

LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}

# ── Colors ──
C_RAMP = "#4A7FD4"           # blue: ramp edges
C_RAMP_FILL = "#D6E4F7"      # light blue: ramp area fill
C_MERGE_ZONE = "#F5A623"     # orange: merge zone
C_MERGE_POINT = "#D0021B"    # red: merge endpoint marker
C_MAIN_ROAD = "#7ED321"      # green: main road edges connected to ramp
C_OTHER_ROAD = "#E0E0E0"     # grey: other roads
C_BG = "#FFFFFF"             # white background
C_EDGE = "#999999"           # grey edge lines


def get_map_file(loc_id):
    for name in [f"exid_loc{loc_id}_orig.net.xml", f"exid_loc{loc_id}.net.xml"]:
        path = os.path.join(MAP_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _lane_polygon(shape, width):
    n = len(shape)
    if n < 2:
        return None
    hw = max(width, 1.5) / 2.0
    left, right = [], []
    for i in range(n):
        if i == 0:
            dx, dy = shape[1][0] - shape[0][0], shape[1][1] - shape[0][1]
        elif i == n - 1:
            dx, dy = shape[-1][0] - shape[-2][0], shape[-1][1] - shape[-2][1]
        else:
            dx, dy = shape[i + 1][0] - shape[i - 1][0], shape[i + 1][1] - shape[i - 1][1]
        length = math.hypot(dx, dy)
        nx, ny = (-dy / length, dx / length) if length > 1e-6 else (0.0, 1.0)
        left.append((shape[i][0] + nx * hw, shape[i][1] + ny * hw))
        right.append((shape[i][0] - nx * hw, shape[i][1] - ny * hw))
    px = [p[0] for p in left] + [p[0] for p in reversed(right)]
    py = [p[1] for p in left] + [p[1] for p in reversed(right)]
    return px, py


def load_lanelet2_onramp_ids(loc_id):
    """Read onramp=yes lanelet IDs from Lanelet2 .osm."""
    import xml.etree.ElementTree as ET
    osm_path = os.path.join(
        EXID_DATA_DIR or "", "maps", "lanelet2",
        f"{loc_id}_{LOC_NAMES[loc_id]}", f"location{loc_id}.osm",
    )

    if not os.path.exists(osm_path):
        print(f"  [WARN] No lanelet2 OSM for location {loc_id} at {osm_path}")
        return set(), set()

    tree = ET.parse(osm_path)
    root = tree.getroot()
    onramp_ids = set()
    highway_ids = set()
    for rel in root.findall("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        lid = int(rel.get("id"))
        if tags.get("onramp") == "yes":
            onramp_ids.add(lid)
        elif tags.get("subtype") == "highway":
            highway_ids.add(lid)
    print(f"  Lanelet2: {len(onramp_ids)} onramp, {len(highway_ids)} highway lanelets")
    return onramp_ids, highway_ids


def get_ramp_trajectory_coords(entries_by_loc):
    """Get trajectory coordinates for ramp phase (before merge_idx)."""
    import pandas as pd
    ramp_coords = defaultdict(list)

    for loc_id, entries in entries_by_loc.items():
        # Group by recording
        by_rec = defaultdict(list)
        for e in entries:
            by_rec[e["rid"]].append(e)

        for rid, rec_entries in sorted(by_rec.items()):
            csv_path = os.path.join(EXID_DATA_DIR or "", "data", f"{rid:02d}_tracks.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                csv = pd.read_csv(csv_path, usecols=["trackId", "frame", "xCenter", "yCenter"])
            except Exception:
                continue

            for e in rec_entries:
                sub = csv[csv["trackId"] == e["tid"]].sort_values("frame")
                if len(sub) < 20:
                    continue
                mi = e["merge_idx"]
                x = sub["xCenter"].values
                y = sub["yCenter"].values
                # Collect ramp phase coords (before merge_idx, sampled)
                if mi > 0:
                    step = max(1, mi // 20)
                    for i in range(0, min(mi, len(x)), step):
                        ramp_coords[loc_id].append((x[i], y[i]))
    return ramp_coords


def find_onramp_edges(net, ramp_coords, loc_id, onramp_lanelet_ids):
    """Identify SUMO edges that belong to on-ramps.

    Uses two methods:
    1. Spatial matching: edges near ramp-phase trajectory coords (primary)
    2. Manual corrections per location (from plot_merge_scenes.py)
    """
    _EXTRA = {
        1: {"-23#2"},
        4: {"3#1"},
    }
    _EXCLUDE = {
        0: {"-3", "-16#0"},
    }
    extra = _EXTRA.get(loc_id, set())
    exclude = _EXCLUDE.get(loc_id, set())

    coords = np.array(ramp_coords.get(loc_id, []))
    if len(coords) == 0:
        return extra - exclude

    onramp_edges = set()
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(":") or eid in exclude:
            continue
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            # Check if edge midpoint is near any ramp trajectory point
            mid = shape[len(shape) // 2]
            dists = np.sqrt(np.sum((coords - np.array([mid[0], mid[1]])) ** 2, axis=1))
            if dists.min() < 15:  # within 15m of ramp trajectory
                onramp_edges.add(eid)
                break

    result = (onramp_edges | extra) - exclude
    return result


def analyze_ramp_topology(net, onramp_edge_ids):
    """
    For each on-ramp, find its merge endpoint(s).

    Algorithm:
    1. Group connected on-ramp edges (they form a single ramp)
    2. For each ramp group, find edges that connect to a junction
    3. The junction's outgoing edges include main road edges
    4. The endpoint of the ramp edge at that junction = merge zone endpoint

    Returns:
        ramp_groups: list of dicts with:
            - edges: set of edge IDs in this ramp
            - merge_endpoints: list of (x, y) where ramp meets main road
            - main_road_edges: set of main road edge IDs at the junction
    """
    # Build adjacency: which edges connect through junctions
    edge_to_junctions = defaultdict(set)
    junction_to_edges_out = defaultdict(set)
    junction_to_edges_in = defaultdict(set)

    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(":"):
            continue
        from_j = edge.getFromNode()
        to_j = edge.getToNode()
        if from_j:
            from_jid = from_j.getID()
            edge_to_junctions[eid].add(("from", from_jid))
            junction_to_edges_out[from_jid].add(eid)
        if to_j:
            to_jid = to_j.getID()
            edge_to_junctions[eid].add(("to", to_jid))
            junction_to_edges_in[to_jid].add(eid)

    # Find groups of connected on-ramp edges
    onramp_set = set(onramp_edge_ids)
    visited = set()
    ramp_groups = []

    for eid in onramp_set:
        if eid in visited:
            continue
        # BFS through junctions to find connected on-ramp edges
        group = set()
        queue = [eid]
        while queue:
            curr = queue.pop()
            if curr in visited or curr not in onramp_set:
                continue
            visited.add(curr)
            group.add(curr)
            # Check connected edges through junctions
            for conn_type, jid in edge_to_junctions.get(curr, set()):
                for neighbor in junction_to_edges_in.get(jid, set()) | junction_to_edges_out.get(jid, set()):
                    if neighbor in onramp_set and neighbor not in visited:
                        queue.append(neighbor)

        if group:
            ramp_groups.append(group)

    # For each ramp group, find merge endpoints
    ramp_info = []
    for group in ramp_groups:
        merge_endpoints = []
        main_road_edges_at_merge = set()

        for eid in group:
            edge = net.getEdge(eid)
            if edge is None:
                continue
            shape = edge.getLanes()[0].getShape()
            if len(shape) < 2:
                continue

            # Check if this edge connects to main road through a junction
            to_j = edge.getToNode()
            if to_j:
                to_jid = to_j.getID()
                outgoing = junction_to_edges_out.get(to_jid, set())
                # Main road edges: edges at this junction NOT in onramp set
                non_ramp_out = outgoing - onramp_set - {eid} - {'': ''}
                # Filter out internal edges
                main_candidates = {o for o in non_ramp_out if not o.startswith(':')}
                if main_candidates:
                    # This ramp edge connects to main road! Endpoint = last point of ramp lane
                    last_pt = shape[-1]
                    merge_endpoints.append((float(last_pt[0]), float(last_pt[1])))
                    main_road_edges_at_merge.update(main_candidates)

            # Also check if THIS edge is the endpoint (no further ramp edges)
            # Some ramps are single edges that end at a junction

        if merge_endpoints:
            ramp_info.append({
                "edges": group,
                "merge_endpoints": merge_endpoints,
                "main_roads": main_road_edges_at_merge,
            })
        else:
            # Even if we can't find explicit connection, use the downstream-most edge's endpoint
            best_edge = None
            best_end = None
            for eid in group:
                edge = net.getEdge(eid)
                if edge is None:
                    continue
                shape = edge.getLanes()[0].getShape()
                if len(shape) < 2:
                    continue
                end = (float(shape[-1][0]), float(shape[-1][1]))
                if best_end is None:
                    best_end = end
                    best_edge = eid
            if best_end:
                ramp_info.append({
                    "edges": group,
                    "merge_endpoints": [best_end],
                    "main_roads": set(),
                })

    return ramp_info


def plot_location(loc_id, net, ramp_info, onramp_edges, title=None):
    """Draw a BEV-style map with on-ramp highlights and merge endpoints."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Compute crop bounds from ramp edges
    all_x, all_y = [], []
    for eid in onramp_edges:
        edge = net.getEdge(eid)
        if edge is None:
            continue
        for lane in edge.getLanes():
            for pt in lane.getShape():
                all_x.append(pt[0])
                all_y.append(pt[1])

    if not all_x:
        print(f"  No ramp edges for location {loc_id}")
        return

    pad = 100
    xmin, xmax = min(all_x) - pad, max(all_x) + pad
    ymin, ymax = min(all_y) - pad, max(all_y) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    # Draw all roads (light grey background)
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(":"):
            continue
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            px, py = _lane_polygon(shape, lane.getWidth())
            if px is None:
                continue
            if eid in onramp_edges:
                # Will draw ramp later with highlight
                pass
            else:
                ax.fill(px, py, color="#F5F5F5", alpha=0.5, linewidth=0)
                ax.plot(px + [px[0]], py + [py[0]], color="#CCCCCC", linewidth=0.3)

    # Draw on-ramp edges (blue highlight)
    for eid in onramp_edges:
        edge = net.getEdge(eid)
        if edge is None:
            continue
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            px, py = _lane_polygon(shape, lane.getWidth())
            if px is None:
                continue
            ax.fill(px, py, color=C_RAMP_FILL, alpha=0.5, linewidth=0)
            ax.plot(px + [px[0]], py + [py[0]], color=C_RAMP, linewidth=1.5, alpha=0.8)

    # Draw main road edges at merge junctions (green highlight)
    for ri in ramp_info:
        for mre in ri.get("main_roads", set()):
            edge = net.getEdge(mre)
            if edge is None:
                continue
            for lane in edge.getLanes():
                shape = lane.getShape()
                if len(shape) < 2:
                    continue
                px, py = _lane_polygon(shape, lane.getWidth())
                if px is None:
                    continue
                ax.fill(px, py, color="#E8F5E9", alpha=0.4, linewidth=0)
                ax.plot(px + [px[0]], py + [py[0]], color=C_MAIN_ROAD, linewidth=0.8, alpha=0.6)

    # Draw merge zone (orange rectangle at each endpoint)
    for ri in ramp_info:
        for i, (mx, my) in enumerate(ri["merge_endpoints"]):
            # Draw merge zone: 30m x 15m rectangle at endpoint, oriented along the ramp
            circle = plt.Circle((mx, my), 25, color=C_MERGE_ZONE, fill=True, alpha=0.15, zorder=2)
            ax.add_patch(circle)
            circle_outline = plt.Circle((mx, my), 25, color=C_MERGE_ZONE, fill=False,
                                         linewidth=2, linestyle="--", alpha=0.6, zorder=2)
            ax.add_patch(circle_outline)

            # Mark merge endpoint with triangle
            ax.plot(mx, my, marker="^", color=C_MERGE_POINT, markersize=15,
                    markeredgecolor="darkred", markeredgewidth=1.5, zorder=5)
            ax.annotate(f"M{i+1}", (mx + 15, my + 15), fontsize=10, fontweight="bold",
                       color=C_MERGE_POINT, zorder=5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_RAMP_FILL, edgecolor=C_RAMP, label="On-ramp road"),
        mpatches.Patch(facecolor="#E8F5E9", edgecolor=C_MAIN_ROAD, label="Main road (at merge)"),
        mpatches.Patch(facecolor=C_MERGE_ZONE + "30", edgecolor=C_MERGE_ZONE,
                      label="Merge zone (25m radius)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_MERGE_POINT,
                   markersize=10, label="Merge endpoint (ramp lane end)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    ax.set_title(title or f"Location {loc_id}: {LOC_NAMES[loc_id]} — Merge Endpoints", fontsize=14)
    ax.set_xlabel("X (m, MetaDrive coordinate)")
    ax.set_ylabel("Y (m, MetaDrive coordinate)")

    out_path = os.path.join(OUT_DIR, f"loc{loc_id}_merge_endpoints.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def plot_all_locations_comparison(all_ramp_info):
    """Create a summary figure showing merge endpoints for all locations."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for loc_id in range(7):
        ax = axes[loc_id]
        if loc_id not in all_ramp_info:
            ax.text(0.5, 0.5, f"Location {loc_id}\nNo data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f"Loc {loc_id}: {LOC_NAMES[loc_id]}")
            continue

        ramp_info = all_ramp_info[loc_id]
        # Simple visualization: just show merge endpoints relative to each other
        all_pts = []
        labels = []
        for ri in ramp_info:
            for j, (mx, my) in enumerate(ri["merge_endpoints"]):
                all_pts.append((mx, my))
                labels.append(f"M{j+1}")

        if all_pts:
            pts = np.array(all_pts)
            ax.scatter(pts[:, 0], pts[:, 1], c=C_MERGE_POINT, s=100, marker="^", zorder=3)
            for i, (mx, my) in enumerate(all_pts):
                circ = plt.Circle((mx, my), 25, color=C_MERGE_ZONE, fill=True, alpha=0.1, zorder=1)
                ax.add_patch(circ)
            ax.set_xlim(pts[:, 0].min() - 50, pts[:, 0].max() + 50)
            ax.set_ylim(pts[:, 1].min() - 50, pts[:, 1].max() + 50)
            ax.set_aspect("equal")

        ax.set_title(f"Loc {loc_id}: {LOC_NAMES[loc_id]}\n{len(ramp_info)} ramp(s)", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide extra subplot
    axes[7].set_visible(False)

    fig.suptitle("exiD Highway On-Ramp Merge Endpoints (Geometric)", fontsize=16, y=1.01)
    out_path = os.path.join(OUT_DIR, "all_locations_merge_endpoints.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=int, default=None, help="Single location to plot")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    # Load merge cache for trajectory coords
    with open(CACHE_PATH) as f:
        merge_cache = json.load(f)

    entries_by_loc = defaultdict(list)
    for loc_str, entries in merge_cache.items():
        entries_by_loc[int(loc_str)] = entries

    # Load trajectory ramp coords
    print("Loading ramp trajectory coordinates...")
    ramp_coords = get_ramp_trajectory_coords(entries_by_loc)

    all_ramp_info = {}
    locations = [args.loc] if args.loc is not None else range(7)

    for loc_id in locations:
        print(f"\nLocation {loc_id}: {LOC_NAMES[loc_id]}")
        net_xml = get_map_file(loc_id)
        if net_xml is None:
            print(f"  No SUMO map found")
            continue

        net = sumolib.net.readNet(net_xml, withInternal=True)

        # Load lanelet2 onramp labels
        onramp_lanelet_ids, highway_ids = load_lanelet2_onramp_ids(loc_id)

        # Find on-ramp SUMO edges
        onramp_edges = find_onramp_edges(net, ramp_coords, loc_id, onramp_lanelet_ids)
        print(f"  On-ramp SUMO edges: {len(onramp_edges)} — {sorted(onramp_edges)}")

        # Analyze ramp topology
        ramp_info = analyze_ramp_topology(net, onramp_edges)
        print(f"  Ramp groups: {len(ramp_info)}")
        for i, ri in enumerate(ramp_info):
            eps = ri["merge_endpoints"]
            print(f"    Ramp {i+1}: edges={sorted(ri['edges'])}, "
                  f"endpoints={[(f'{x:.1f}', f'{y:.1f}') for x, y in eps]}, "
                  f"main_roads={sorted(ri['main_roads'])}")

        all_ramp_info[loc_id] = ramp_info

        if not args.summary_only:
            plot_location(loc_id, net, ramp_info, onramp_edges)

    # Summary figure
    plot_all_locations_comparison(all_ramp_info)

    # Print summary table
    print("\n" + "=" * 70)
    print("MERGE ENDPOINT SUMMARY")
    print("=" * 70)
    for loc_id in range(7):
        ri = all_ramp_info.get(loc_id, [])
        n_endpoints = sum(len(r["merge_endpoints"]) for r in ri)
        n_ramps = len(ri)
        endpoints_str = "; ".join(
            f"({x:.1f}, {y:.1f})" for r in ri for x, y in r["merge_endpoints"]
        )
        print(f"  Loc {loc_id} ({LOC_NAMES[loc_id]}): "
              f"{n_ramps} ramp(s), {n_endpoints} endpoint(s): {endpoints_str or 'N/A'}")

    print(f"\nOutput: {OUT_DIR}/")


if __name__ == "__main__":
    main()
