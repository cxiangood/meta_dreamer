#!/usr/bin/env python3
"""
Extract SUMO on-ramp edge endpoint coordinates for all 7 exiD locations.

The on-ramp edge ENDPOINT is the physical merge point where the ramp joins
the main road. For each trajectory, we find the frame closest to the nearest
endpoint → that becomes the corrected merge_idx.

Output: exid_merge_endpoints.json with per-location endpoint data.

Usage:
    python3 mirro_data_map/dump_ramp_endpoints.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo",
)
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
os.environ["SUMO_HOME"] = SUMO_HOME

import sumolib
from mirro_data_map.collect_merge_data import (
    get_map_file, LOC_NAMES, DATA_DIR,
)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "exid_merge_cache.json")

# Location-specific manual corrections (from plot_merge_scenes.py)
_ONRAMP_EXTRA = {
    1: {"-23#2"},
    4: {"3#1"},
}
_ONRAMP_EXCLUDE = {
    0: {"-3", "-16#0"},
}


def get_onramp_sumo_edges(loc_id, net, ramp_pts):
    """Find on-ramp SUMO edges by spatial matching with pre-merge points."""
    extra = _ONRAMP_EXTRA.get(loc_id, set())
    exclude = _ONRAMP_EXCLUDE.get(loc_id, set())

    if len(ramp_pts) == 0:
        return extra

    onramp_edges = set()
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(":") or len(edge.getLanes()) > 2:
            continue
        if eid in exclude:
            continue
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            mid = shape[len(shape) // 2]
            dists = np.sqrt(np.sum((ramp_pts - mid) ** 2, axis=1))
            if dists.min() < 10:
                onramp_edges.add(eid)
                break
    return (onramp_edges | extra) - exclude


def load_trajectory_coords(rid, tid):
    """Load (x, y) coordinates for a single trajectory from CSV."""
    csv_path = os.path.join(DATA_DIR, f"{rid:02d}_tracks.csv")
    df = pd.read_csv(csv_path, low_memory=False)
    ego = df[df["trackId"] == tid].sort_values("frame")
    x = ego["xCenter"].values.astype(np.float64)
    y = ego["yCenter"].values.astype(np.float64)
    return np.column_stack([x, y])


def main():
    print("=== Extracting on-ramp endpoint coordinates ===\n")

    # Load merge cache (format: {"0": [{rid, tid, merge_idx, loc_id}, ...], ...})
    with open(CACHE_PATH, "r") as f:
        cache = json.load(f)

    # cache is keyed by string loc_id
    n_total = sum(len(v) for v in cache.values())
    print(f"Loaded {n_total} merge entries from cache")
    print(f"Locations: {sorted(cache.keys(), key=int)}")
    for loc_key in sorted(cache.keys(), key=int):
        print(f"  Loc {loc_key}: {len(cache[loc_key])} trajectories")

    results = {}

    for loc_key in sorted(cache.keys(), key=int):
        loc_id = int(loc_key)
        entries = cache[loc_key]
        print(f"\n--- Loc {loc_id}: {LOC_NAMES[loc_id]} ---")

        # Load SUMO net
        net_xml = get_map_file(loc_id)
        net = sumolib.net.readNet(net_xml, withInternal=False)
        print(f"  SUMO net: {os.path.basename(net_xml)}")

        # Load ramp_pts: use first 60% of each trajectory
        all_ramp_pts = []
        traj_coords = {}
        for entry in entries:
            rid, tid = entry["rid"], entry["tid"]
            try:
                coords = load_trajectory_coords(rid, tid)
                T = len(coords)
                pre_n = max(1, int(T * 0.6))
                step = max(1, pre_n // 20)
                for i in range(0, pre_n, step):
                    all_ramp_pts.append(coords[i])
                traj_coords[(rid, tid)] = coords
            except Exception as e:
                continue

        if not all_ramp_pts:
            print("  WARNING: no trajectory data loaded")
            continue

        ramp_pts = np.array(all_ramp_pts)
        print(f"  Pre-merge reference points: {len(ramp_pts)} (from {len(traj_coords)} trajs)")

        # Find on-ramp edges
        onramp_edges = get_onramp_sumo_edges(loc_id, net, ramp_pts)
        print(f"  On-ramp edges ({len(onramp_edges)}): {sorted(onramp_edges)}")

        # Get endpoints for each on-ramp edge
        endpoints = {}
        for eid in sorted(onramp_edges):
            edge = net.getEdge(eid)
            last_lane = edge.getLanes()[-1]
            shape = last_lane.getShape()
            endpoint = (float(shape[-1][0]), float(shape[-1][1]))
            endpoints[eid] = {
                "endpoint": endpoint,
                "from_node": edge.getFromNode().getID(),
                "to_node": edge.getToNode().getID(),
            }
            print(f"    {eid}: endpoint=({endpoint[0]:.1f}, {endpoint[1]:.1f})")

        # For each trajectory, find closest approach to nearest endpoint
        endpoint_coords = np.array([v["endpoint"] for v in endpoints.values()])
        endpoint_ids = list(endpoints.keys())

        corrected_merges = []
        for (rid, tid), coords in traj_coords.items():
            T = len(coords)
            min_dists = np.full(T, np.inf)
            for ep in endpoint_coords:
                dists = np.sqrt(np.sum((coords - ep) ** 2, axis=1))
                min_dists = np.minimum(min_dists, dists)

            merge_idx = int(np.argmin(min_dists))
            min_dist = float(min_dists[merge_idx])

            # Find nearest endpoint to merge point
            nearest_ep_idx = int(np.argmin(
                np.sqrt(np.sum((endpoint_coords - coords[merge_idx]) ** 2, axis=1))
            ))
            nearest_eid = endpoint_ids[nearest_ep_idx]

            corrected_merges.append({
                "rid": int(rid), "tid": int(tid),
                "merge_idx": merge_idx,
                "dist_to_endpoint": round(min_dist, 2),
                "nearest_edge": nearest_eid,
                "traj_len": T,
            })

        # Per-endpoint stats
        ep_groups = {}
        for cm in corrected_merges:
            eid = cm["nearest_edge"]
            ep_groups.setdefault(eid, []).append(cm["merge_idx"])

        results[str(loc_id)] = {
            "name": LOC_NAMES[loc_id],
            "onramp_edges": endpoints,
            "n_trajectories": len(corrected_merges),
            "per_endpoint_stats": {
                eid: {
                    "n": len(idxs),
                    "merge_idx_mean": float(np.mean(idxs)),
                    "merge_idx_std": float(np.std(idxs)),
                    "merge_idx_min": int(np.min(idxs)),
                    "merge_idx_max": int(np.max(idxs)),
                }
                for eid, idxs in ep_groups.items()
            },
        }

        for eid, stats_data in results[str(loc_id)]["per_endpoint_stats"].items():
            print(f"  {eid}: {stats_data['n']} trajs, "
                  f"merge_idx mean={stats_data['merge_idx_mean']:.0f}, "
                  f"range=[{stats_data['merge_idx_min']}, {stats_data['merge_idx_max']}]")

    # Save endpoint data
    out_path = os.path.join(os.path.dirname(__file__), "exid_merge_endpoints.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved endpoint data to {out_path}")

    # Build corrected merge cache
    all_corrected = []
    for loc_key in sorted(cache.keys(), key=int):
        loc_id = int(loc_key)
        if loc_key not in results:
            continue
        entries = cache[loc_key]
        eps = results[loc_key]["onramp_edges"]
        endpoint_coords = np.array([v["endpoint"] for v in eps.values()])
        for entry in entries:
            rid, tid = entry["rid"], entry["tid"]
            try:
                coords = load_trajectory_coords(rid, tid)
                min_dists = np.full(len(coords), np.inf)
                for ep in endpoint_coords:
                    dists = np.sqrt(np.sum((coords - ep) ** 2, axis=1))
                    min_dists = np.minimum(min_dists, dists)
                merge_idx = int(np.argmin(min_dists))
            except Exception:
                merge_idx = entry.get("merge_idx", -1)
            all_corrected.append({
                "rid": int(rid), "tid": int(tid),
                "merge_idx": merge_idx, "loc_id": loc_id,
            })

    cache_out = os.path.join(os.path.dirname(__file__), "exid_merge_cache_v2.json")
    with open(cache_out, "w") as f:
        json.dump(all_corrected, f, indent=2)
    print(f"Saved corrected merge cache ({len(all_corrected)} entries) to {cache_out}")


if __name__ == "__main__":
    main()
