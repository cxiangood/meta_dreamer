#!/usr/bin/env python3
"""
Build merge cache using SUMO on-ramp edge endpoints as merge points.
Merge point = on-ramp edge's "to" node (where it joins the main road).

For each trajectory:
  - Determine which on-ramp edge the vehicle starts on (first recorded lanelet)
  - Get that edge's endpoint coordinate (the physical merge point)
  - Find the frame where ego is closest to this point → merge_idx
"""
import json, os, sys, math
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

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

def load_onramp_lanelet_ids(loc_id):
    """Get onramp lanelet IDs from Lanelet2 .osm GT labels."""
    osm_path = os.path.join(
        os.environ['EXID_DATASET_DIR'], 'maps', 'lanelet2',
        f'{loc_id}_{LOC_NAMES[loc_id]}', f'location{loc_id}.osm'
    )
    if not os.path.exists(osm_path):
        return set()
    tree = ET.parse(osm_path)
    root = tree.getroot()
    onramp_ids = set()
    for rel in root.findall('relation'):
        tags = {t.get('k'): t.get('v') for t in rel.findall('tag')}
        if tags.get('type') == 'lanelet' and tags.get('onramp') == 'yes':
            onramp_ids.add(int(rel.get('id')))
    return onramp_ids

def get_map_file(loc_id):
    for name in [f'exid_loc{loc_id}_orig.net.xml', f'exid_loc{loc_id}.net.xml']:
        path = os.path.join(MAP_DIR, name)
        if os.path.exists(path):
            return path
    return None

def get_onramp_endpoints(net, onramp_ll_ids):
    """Find SUMO edges corresponding to onramp lanelets, return their 'to' node positions.
    
    SUMO edge IDs often contain lanelet IDs as suffixes.
    We find edges whose lane's 'param' contains the onramp lanelet ID,
    or edges near the onramp lanelet positions.
    """
    endpoints = {}  # lanelet_id -> (x, y) endpoint coordinate
    
    for edge in net.getEdges():
        eid = edge.getID()
        # Check if this edge corresponds to an onramp lanelet
        for lane in edge.getLanes():
            # Try matching lane params
            l_shape = lane.getShape()
            if len(l_shape) < 2:
                continue
            
            # Heuristic: check if edge name contains lanelet-like numbers
            # Lanelet2 IDs are typically integer lanelet numbers
            # In SUMO, edges from Lanelet2 often have format like ":junction_X_Y" or numbers
            for ll_id in onramp_ll_ids:
                str_id = str(ll_id)
                if str_id in eid or eid.endswith('_' + str_id) or eid.startswith(str_id + '_'):
                    # Found matching edge, get endpoint
                    to_node = edge.getToNode()
                    ep = to_node.getCoord()  # (lon, lat) = (SUMO_x, SUMO_y)
                    endpoints[ll_id] = (ep[0], ep[1])
                    break
    
    return endpoints

def get_onramp_endpoints_v2(net, onramp_ll_ids):
    """Alternative: find onramp edges by their position relative to known onramp areas.
    
    For each onramp lanelet ID, we look for edges that:
    1. Have lanelet IDs close to the onramp IDs
    2. OR are at positions where onramp trajectories cluster
    
    Since we may not have direct edge-lanelet mapping, use a different approach:
    Get the 'to' coordinates of ALL edges that could be onramps.
    """
    # Strategy: use edge naming conventions
    # From Lanelet2 conversion: edges often named like "ramp_N" or have lanelet IDs in them
    endpoints = {}
    
    # Collect all potential onramp edge candidates
    candidates = []
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(':'):
            continue  # skip internal edges
        
        # Try matching lanelet IDs in edge ID
        for ll_id in onramp_ll_ids:
            if str(ll_id) in eid:
                to_node = edge.getToNode()
                ep = to_node.getCoord()
                candidates.append((ll_id, ep, eid))
                break
    
    for ll_id, ep, eid in candidates:
        endpoints[ll_id] = (ep[0], ep[1])
    
    return endpoints

cache = {}
MIN_SEG = 20  # minimum 20 frames (0.8s) pre/post merge

for loc_id in range(7):
    name = LOC_NAMES[loc_id]
    print(f'\nLocation {loc_id}: {name}')
    
    # Load GT onramp lanelet IDs
    onramp_ll_ids = load_onramp_lanelet_ids(loc_id)
    print(f'  Onramp lanelet IDs: {len(onramp_ll_ids)}')
    
    if not onramp_ll_ids:
        print(f'  WARNING: No GT onramp IDs, skipping')
        cache[str(loc_id)] = []
        continue
    
    # Load SUMO net
    net_xml = get_map_file(loc_id)
    if not net_xml:
        print(f'  WARNING: No net.xml file')
        cache[str(loc_id)] = []
        continue
    
    net = sumolib.net.readNet(net_xml, withInternal=False)
    
    # Find onramp edge endpoints from SUMO net
    endpoints = get_onramp_endpoints_v2(net, onramp_ll_ids)
    print(f'  Onramp endpoints found: {len(endpoints)}')
    for ll_id, ep in sorted(endpoints.items()):
        print(f'    ll{ll_id} -> endpoint ({ep[0]:.1f}, {ep[1]:.1f})')
    
    # If no edge-lanelet mapping found, use alternative: 
    # find endpoints by analyzing onramp trajectories
    if not endpoints:
        print(f'  WARNING: No edge-lanelet mapping found, using trajectory-based approach')
        # Fall back to laneletId transition method (but with GT labels)
        # We'll collect trajectories and use the old method
        
        # Get recording IDs for this location
        rec_ids = []
        for f in sorted(os.listdir(DATA_DIR)):
            if f.endswith('_recordingMeta.csv'):
                m = pd.read_csv(os.path.join(DATA_DIR, f))
                if int(m['locationId'].iloc[0]) == loc_id:
                    rec_ids.append(int(m['recordingId'].iloc[0]))
        
        sl = pd.read_csv(os.path.join(DATA_DIR, f'{rec_ids[0]:02d}_recordingMeta.csv'))['speedLimit'].iloc[0]
        
        # Use old method: laneletId transition with GT labels
        from collect_merge_data import classify_lanelets, find_merge_tracks
        ramp_ll, main_ll, _ = classify_lanelets(rec_ids, sl)
        
        all_trajs = []
        for rid in sorted(rec_ids):
            merges = find_merge_tracks(rid, ramp_ll, main_ll, net, loc_id=loc_id)
            csv = pd.read_csv(os.path.join(DATA_DIR, f'{rid:02d}_tracks.csv'),
                            usecols=['trackId', 'frame', 'laneletId'], low_memory=False)
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
                
                if actual_mi < MIN_SEG or (len(sub) - actual_mi) < MIN_SEG:
                    continue
                
                all_trajs.append({
                    'rid': int(rid), 'tid': int(tid),
                    'merge_idx': int(actual_mi), 'loc_id': loc_id,
                })
        
        cache[str(loc_id)] = all_trajs
        print(f'  {len(all_trajs)} trajectories (lanelet-transition method)')
        continue
    
    # Use endpoint-based method
    # Get recording IDs
    rec_ids = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('_recordingMeta.csv'):
            m = pd.read_csv(os.path.join(DATA_DIR, f))
            if int(m['locationId'].iloc[0]) == loc_id:
                rec_ids.append(int(m['recordingId'].iloc[0]))
    
    all_trajs = []
    for rid in sorted(rec_ids):
        csv = pd.read_csv(os.path.join(DATA_DIR, f'{rid:02d}_tracks.csv'),
                         usecols=['trackId', 'frame', 'xCenter', 'yCenter', 'laneletId'],
                         low_memory=False)
        
        # Group by track
        for tid, sub in csv.groupby('trackId'):
            if len(sub) < MIN_SEG * 2:
                continue
            sub = sub.sort_values('frame')
            
            # Check if this track starts on an onramp lanelet
            first_ll = int(sub['laneletId'].iloc[0])
            if first_ll not in onramp_ll_ids and first_ll not in endpoints:
                continue
            
            # Get the merge endpoint for this onramp
            # Find which onramp this vehicle is on
            onramp_id = first_ll if first_ll in endpoints else None
            if onramp_id is None:
                # Find first onramp lanelet in the track
                for ll in sub['laneletId'].values:
                    if int(ll) in endpoints:
                        onramp_id = int(ll)
                        break
            
            if onramp_id is None or onramp_id not in endpoints:
                continue
            
            ep_x, ep_y = endpoints[onramp_id]
            
            # Find frame closest to endpoint (after passing it)
            x = sub['xCenter'].values.astype(float)
            y = sub['yCenter'].values.astype(float)
            
            # Compute distances
            dists = np.sqrt((x - ep_x)**2 + (y - ep_y)**2)
            
            # Find the point of closest approach
            closest_frame = int(np.argmin(dists))
            
            # Find the frame where ego EXITS the onramp (laneletId changes from onramp to non-onramp)
            # This is more precise than just closest distance
            ll_seq = sub['laneletId'].values.astype(int)
            merge_idx = closest_frame  # fallback
            
            for j in range(1, len(ll_seq)):
                if ll_seq[j-1] in onramp_ll_ids and ll_seq[j] not in onramp_ll_ids:
                    merge_idx = j
                    break
            
            if merge_idx < MIN_SEG or (len(sub) - merge_idx) < MIN_SEG:
                continue
            
            all_trajs.append({
                'rid': int(rid), 'tid': int(tid),
                'merge_idx': int(merge_idx), 'loc_id': loc_id,
            })
    
    cache[str(loc_id)] = all_trajs
    print(f'  {len(all_trajs)} trajectories (endpoint method)')

# Save cache
with open(CACHE_PATH, 'w') as f:
    json.dump(cache, f, indent=2)

print(f'\nCache saved to {CACHE_PATH}')
print('Summary:')
for loc_id in range(7):
    split = 'Train' if loc_id in [0, 2, 4, 5, 6] else 'Test'
    n = len(cache[str(loc_id)])
    print(f'  Loc {loc_id} [{split}]: {n}')

