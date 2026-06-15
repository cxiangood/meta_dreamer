#!/usr/bin/env python3
"""
Build merge cache using SUMO on-ramp edge ENDPOINTS as merge points.

For each on-ramp SUMO edge:
  - Its TO-NODE coordinate = the physical merge point (where ramp joins main road)
  
For each trajectory:
  - Check if it starts on an on-ramp lanelet (from Lanelet2 GT)
  - Find the on-ramp edge's endpoint
  - merge_idx = frame where ego is closest to this endpoint
"""
import json, os, sys
import numpy as np
import pandas as pd

os.environ['EXID_DATASET_DIR'] = '/share/home/u23516/data/exiD-dataset-v2.1'
SUMO_HOME = '/share/apps/sumo-1.20'
os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.insert(0, os.path.join(SUMO_HOME, 'share/sumo/tools'))
import sumolib

REPO = '/share/home/u23516/code/meta_dreamer-main'
DATA_DIR = '/share/home/u23516/data/exiD-dataset-v2.1/data'
MAP_DIR = os.path.join(REPO, 'mirro_data_map')
CACHE_PATH = os.path.join(MAP_DIR, 'exid_merge_cache.json')

LOC_NAMES = {
    0: 'cologne_butzweiler', 1: 'cologne_fortiib', 2: 'aachen_brand',
    3: 'bergheim_roemer', 4: 'cologne_klettenberg', 5: 'aachen_laurensberg',
    6: 'merzenich_rather',
}

# From plot_merge_scenes.py: location-specific corrections
_ONRAMP_EXTRA = {1: {"-23#2"}, 4: {"3#1"}}
_ONRAMP_EXCLUDE = {0: {"-3", "-16#0"}}

def get_map_file(loc_id):
    for name in [f'exid_loc{loc_id}_orig.net.xml', f'exid_loc{loc_id}.net.xml']:
        path = os.path.join(MAP_DIR, name)
        if os.path.exists(path):
            return path
    return None

def get_onramp_endpoints(loc_id, net, ramp_traj_coords):
    """Find on-ramp edges and their endpoint coordinates.
    
    Uses the same logic as plot_merge_scenes.get_onramp_sumo_edges,
    but additionally extracts the TO-NODE coordinate of each on-ramp edge.
    """
    extra = _ONRAMP_EXTRA.get(loc_id, set())
    exclude = _ONRAMP_EXCLUDE.get(loc_id, set())
    
    if len(ramp_traj_coords) == 0:
        return {}
    
    ramp_pts = np.array(ramp_traj_coords)
    
    endpoints = {}  # edge_id -> (x, y) endpoint coordinate
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(':'):
            continue  # skip internal edges
        if len(edge.getLanes()) > 2:
            continue  # skip multi-lane edges (likely main road)
        if eid in exclude:
            continue
        
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            mid = shape[len(shape) // 2]
            dists = np.sqrt(np.sum((ramp_pts - mid) ** 2, axis=1))
            if dists.min() < 10:
                # Found on-ramp edge
                to_node = edge.getToNode()
                tc = to_node.getCoord()  # (SUMO_x, SUMO_y)
                endpoints[eid] = (tc[0], tc[1])
                break
    
    return endpoints

MIN_SEG = 20

cache = {}
for loc_id in range(7):
    name = LOC_NAMES[loc_id]
    print(f'\nLocation {loc_id}: {name}')
    
    # Get recording IDs
    rec_ids = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('_recordingMeta.csv'):
            m = pd.read_csv(os.path.join(DATA_DIR, f))
            if int(m['locationId'].iloc[0]) == loc_id:
                rec_ids.append(int(m['recordingId'].iloc[0]))
    
    print(f'  {len(rec_ids)} recordings')
    
    # Load SUMO net
    net_xml = get_map_file(loc_id)
    if not net_xml:
        print(f'  No net.xml, skipping')
        cache[str(loc_id)] = []
        continue
    net = sumolib.net.readNet(net_xml, withInternal=False)
    
    # Step 1: do a first pass with lanelet classification to find ramp trajectories
    # (We need trajectory coords to locate on-ramp edges in SUMO)
    from collect_merge_data import classify_lanelets, find_merge_tracks
    
    sl = pd.read_csv(os.path.join(DATA_DIR, f'{rec_ids[0]:02d}_recordingMeta.csv'))['speedLimit'].iloc[0]
    ramp_ll, main_ll, main_speed = classify_lanelets(rec_ids, sl)
    
    # Collect ramp trajectory coordinates for SUMO edge matching
    ramp_coords = []
    first_pass_trajs = {}
    
    for rid in sorted(rec_ids):
        merges = find_merge_tracks(rid, ramp_ll, main_ll, net, loc_id=loc_id)
        if not merges:
            continue
        csv = pd.read_csv(os.path.join(DATA_DIR, f'{rid:02d}_tracks.csv'),
                         usecols=['trackId', 'frame', 'xCenter', 'yCenter', 'laneletId'],
                         low_memory=False)
        for m in merges:
            tid = m['track_id']
            sub = csv[csv['trackId'] == int(tid)].sort_values('frame')
            mi = m['merge_frame_idx']
            ll = sub['laneletId'].values
            
            # Refine: exact laneletId transition
            actual_mi = mi
            for j in range(1, len(ll)):
                if ll[j-1] in ramp_ll and ll[j] in main_ll:
                    actual_mi = j
                    break
            
            x = sub['xCenter'].values.astype(float)
            y = sub['yCenter'].values.astype(float)
            
            # Collect pre-merge coordinates for SUMO edge matching
            if actual_mi > 0 and actual_mi < len(x):
                step = max(1, actual_mi // 20)
                for i in range(0, actual_mi, step):
                    ramp_coords.append((x[i], y[i]))
            
            key = (rid, tid)
            first_pass_trajs[key] = {
                'x': x, 'y': y, 'll': ll, 'mi': actual_mi,
                'rid': rid, 'tid': tid,
            }
    
    print(f'  First pass: {len(first_pass_trajs)} trajectories, {len(ramp_coords)} ramp coords')
    
    # Step 2: find on-ramp edges and their endpoints from SUMO net
    endpoints = get_onramp_endpoints(loc_id, net, ramp_coords)
    print(f'  On-ramp endpoints from SUMO: {len(endpoints)}')
    for eid, ep in sorted(endpoints.items()):
        print(f'    edge {eid} -> endpoint ({ep[0]:.1f}, {ep[1]:.1f})')
    
    # Step 3: for each trajectory, find merge frame closest to any on-ramp endpoint
    all_trajs = []
    for key, t in first_pass_trajs.items():
        x, y, mi = t['x'], t['y'], t['mi']
        
        if not endpoints:
            # No SUMO endpoints found, use lanelet-transition merge_idx
            actual_mi = mi
        else:
            # Find closest approach to any on-ramp endpoint
            # Only consider frames where ll[j-1] in ramp_ll (still on ramp)
            ll = t['ll']
            best_dist = float('inf')
            best_frame = mi  # fallback to lanelet transition
            
            # Search within [mi*0.5, mi*1.5] window around expected merge
            search_start = max(0, int(mi * 0.5))
            search_end = min(len(x), int(mi * 1.5))
            
            for j in range(search_start, search_end):
                px, py = x[j], y[j]
                for eid, (epx, epy) in endpoints.items():
                    dist = np.sqrt((px - epx)**2 + (py - epy)**2)
                    # Prefer frames where vehicle is still on ramp
                    if j > 0:
                        prev_ll = int(ll[j-1]) if j > 0 else -1
                        curr_ll = int(ll[j])
                        if prev_ll in ramp_ll and curr_ll not in ramp_ll:
                            # This is the lanelet transition frame - weight it
                            dist *= 0.5
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_frame = j
            
            actual_mi = best_frame
        
        if actual_mi < MIN_SEG or (len(x) - actual_mi) < MIN_SEG:
            continue
        
        all_trajs.append({
            'rid': int(t['rid']), 'tid': int(t['tid']),
            'merge_idx': int(actual_mi), 'loc_id': loc_id,
        })
    
    cache[str(loc_id)] = all_trajs
    print(f'  Final: {len(all_trajs)} trajectories')

# Save
with open(CACHE_PATH, 'w') as f:
    json.dump(cache, f, indent=2)

print(f'\nSaved to {CACHE_PATH}')
print('Summary:')
for loc_id in range(7):
    split = 'Train' if loc_id in [0, 2, 4, 5, 6] else 'Test'
    print(f'  Loc {loc_id} [{split}]: {len(cache[str(loc_id)])}')
