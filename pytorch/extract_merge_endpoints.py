"""
Extract on-ramp edges and merge endpoints for all 7 locations.
Run on HPC (has CSV + lanelet2 data). Outputs JSON.
"""
import json, os, sys, math
import numpy as np, pandas as pd

# ── Paths ──
REPO = "/share/home/u23516/code/meta_dreamer-main"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "mirro_data_map"))
sys.path.insert(0, os.path.join(os.environ.get("SUMO_HOME","/share/apps/sumo-1.20"), "share/sumo/tools"))
os.environ["SUMO_HOME"] = os.environ.get("SUMO_HOME", "/share/apps/sumo-1.20")
import sumolib

from collect_merge_data import LOC_NAMES, DATA_DIR, MAP_DIR, load_lanelet2_onramp_ids

CACHE_PATH = os.path.join(REPO, "mirro_data_map", "exid_merge_cache.json")
OUT_PATH = os.path.join(REPO, "pytorch", "merge_endpoints.json")

LOC_NAMES_MAP = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}

_ONRAMP_EXTRA = {1: {"-23#2"}, 4: {"3#1"}}
_ONRAMP_EXCLUDE = {0: {"-3", "-16#0"}}


def get_onramp_edges(loc_id, net, trajs):
    """Same logic as plot_merge_scenes.py"""
    extra = _ONRAMP_EXTRA.get(loc_id, set())
    exclude = _ONRAMP_EXCLUDE.get(loc_id, set())
    
    ramp_pts = []
    for t in trajs:
        mi = t["merge_idx"]
        if mi > 0:
            step = max(1, mi // 20)
            for i in range(0, mi, step):
                ramp_pts.append((t["x"][i], t["y"][i]))
    if not ramp_pts:
        return extra
    ramp_pts = np.array(ramp_pts)
    
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
            mid = shape[len(shape)//2]
            dists = np.sqrt(np.sum((ramp_pts - mid)**2, axis=1))
            if dists.min() < 10:
                onramp_edges.add(eid)
                break
    return (onramp_edges | extra) - exclude


def find_merge_endpoints(net, onramp_edges):
    """
    For each on-ramp, find the geometric merge endpoint.
    
    Algorithm:
    1. Group connected on-ramp edges (through junctions) into ramp groups
    2. For each group, find the downstream-most edge
    3. The merge endpoint = last coordinate of that edge's lane shape
    4. Also find main road edges at the merge junction
    """
    # Build junction connectivity
    junc_out = {}
    junc_in = {}
    edge_from = {}
    edge_to = {}
    for e in net.getEdges():
        eid = e.getID()
        if eid.startswith(":"):
            continue
        fn = e.getFromNode()
        tn = e.getToNode()
        if fn:
            fid = fn.getID()
            edge_from[eid] = fid
            junc_out.setdefault(fid, set()).add(eid)
        if tn:
            tid = tn.getID()
            edge_to[eid] = tid
            junc_in.setdefault(tid, set()).add(eid)
    
    onramp_set = set(onramp_edges)
    visited = set()
    ramp_groups = []
    
    for eid in onramp_set:
        if eid in visited:
            continue
        group = set()
        queue = [eid]
        while queue:
            cur = queue.pop()
            if cur in visited or cur not in onramp_set:
                continue
            visited.add(cur)
            group.add(cur)
            # Connected through junctions
            for conn_eid in onramp_set:
                if conn_eid in visited:
                    continue
                # Same junction?
                if (edge_from.get(cur) and edge_from.get(conn_eid) and edge_from[cur] == edge_from[conn_eid]) or \
                   (edge_to.get(cur) and edge_to.get(conn_eid) and edge_to[cur] == edge_to[conn_eid]) or \
                   (edge_to.get(cur) and edge_from.get(conn_eid) and edge_to[cur] == edge_from[conn_eid]) or \
                   (edge_from.get(cur) and edge_to.get(conn_eid) and edge_from[cur] == edge_to[conn_eid]):
                    queue.append(conn_eid)
        if group:
            ramp_groups.append(group)
    
    # For each ramp group, find merge endpoint
    result = []
    for group in ramp_groups:
        merge_endpoints = []
        main_roads = set()
        
        for eid in group:
            edge = net.getEdge(eid)
            if edge is None:
                continue
            shape = edge.getLanes()[0].getShape()
            if len(shape) < 2:
                continue
            
            # Check TO junction for main road connection
            tid = edge_to.get(eid)
            if tid:
                outgoing = junc_out.get(tid, set())
                non_ramp = outgoing - onramp_set - {eid}
                # Filter internal edges
                non_ramp_real = {o for o in non_ramp if not o.startswith(':')}
                if non_ramp_real:
                    last = shape[-1]
                    merge_endpoints.append({"x": float(last[0]), "y": float(last[1]), "edge": eid})
                    main_roads.update(non_ramp_real)
        
        if not merge_endpoints:
            # Use most downstream edge's endpoint
            best_eid = None
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
                    best_eid = eid
            if best_end:
                merge_endpoints.append({"x": best_end[0], "y": best_end[1], "edge": best_eid})
        
        result.append({
            "edges": sorted(group),
            "merge_endpoints": merge_endpoints,
            "main_roads": sorted(main_roads),
        })
    
    # If only one ramp group but multiple endpoints, keep all
    # Sort ramp groups by x-coordinate of first endpoint
    result.sort(key=lambda r: r["merge_endpoints"][0]["x"] if r["merge_endpoints"] else 0)
    return result


def main():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    
    all_data = {}
    
    for loc_id in range(7):
        print(f"\nLocation {loc_id}: {LOC_NAMES_MAP[loc_id]}")
        
        # Load trajectories
        entries = cache.get(str(loc_id), [])
        if not entries:
            print(f"  No entries")
            continue
        
        # Build trajectories from CSV (like _build_trajs_from_cache)
        by_rec = {}
        for e in entries:
            by_rec.setdefault(e["rid"], []).append((e["tid"], e["merge_idx"]))
        
        trajs = []
        for rid, tracks in sorted(by_rec.items()):
            csv_path = os.path.join(DATA_DIR, f"{rid:02d}_tracks.csv")
            if not os.path.exists(csv_path):
                continue
            csv = pd.read_csv(csv_path, usecols=["trackId", "frame", "xCenter", "yCenter"])
            for tid, mi in tracks:
                sub = csv[csv["trackId"] == tid].sort_values("frame")
                if len(sub) < 20:
                    continue
                trajs.append({
                    "tid": tid, "rid": rid,
                    "x": sub["xCenter"].values,
                    "y": sub["yCenter"].values,
                    "merge_idx": mi,
                })
        
        print(f"  {len(trajs)} trajectories loaded")
        
        # Load SUMO net
        net_xml = MAP_DIR + f"/exid_loc{loc_id}_orig.net.xml"
        if not os.path.exists(net_xml):
            net_xml = MAP_DIR + f"/exid_loc{loc_id}.net.xml"
        net = sumolib.net.readNet(net_xml, withInternal=True)
        
        # Get on-ramp edges
        onramp_edges = get_onramp_edges(loc_id, net, trajs)
        print(f"  On-ramp edges: {sorted(onramp_edges)}")
        
        # Find merge endpoints
        ramp_info = find_merge_endpoints(net, onramp_edges)
        for i, ri in enumerate(ramp_info):
            eps = ri["merge_endpoints"]
            pts_str = ", ".join(f"({p['x']:.1f},{p['y']:.1f})" for p in eps)
            print(f"  Ramp {i+1}: {sorted(ri['edges'])} -> {pts_str}, main={ri['main_roads']}")
        
        # Get lanelet2 onramp info
        onramp_ll, highway_ll = load_lanelet2_onramp_ids(loc_id)
        
        all_data[str(loc_id)] = {
            "name": LOC_NAMES_MAP[loc_id],
            "onramp_edges": sorted(onramp_edges),
            "ramp_groups": ramp_info,
            "onramp_lanelet_ids": sorted(onramp_ll) if onramp_ll else [],
            "highway_lanelet_ids": sorted(highway_ll) if highway_ll else [],
            "n_trajectories": len(trajs),
        }
    
    with open(OUT_PATH, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
