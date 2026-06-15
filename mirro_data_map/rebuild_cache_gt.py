#!/usr/bin/env python3
"""Rebuild merge cache with Lanelet2 GT labels."""
import json, os, sys
import pandas as pd
import numpy as np

REPO = '/share/home/u23516/code/meta_dreamer-main'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'mirro_data_map'))

# CRITICAL: set env BEFORE importing collect_merge_data
os.environ['EXID_DATASET_DIR'] = '/share/home/u23516/data/exiD-dataset-v2.1'

SUMO_HOME = '/share/apps/sumo-1.20'
os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.insert(0, os.path.join(SUMO_HOME, 'share/sumo/tools'))

import sumolib
from collect_merge_data import (
    classify_lanelets, find_merge_tracks, get_map_file,
    load_lanelet2_onramp_ids, LOC_NAMES, DATA_DIR
)

CACHE_PATH = os.path.join(REPO, 'mirro_data_map', 'exid_merge_cache.json')

cache = {}
MIN_SEG = 20

for loc_id in range(7):
    name = LOC_NAMES[loc_id]
    print(f'\nLocation {loc_id}: {name}')
    
    # Use GT labels via EXID_DATASET_DIR
    onramp_ids, highway_ids = load_lanelet2_onramp_ids(loc_id)
    if onramp_ids and highway_ids:
        print(f'  GT: onramp={len(onramp_ids)}, highway={len(highway_ids)}')
    else:
        print(f'  WARNING: no GT labels, falling back to speed')
    
    # Get recording IDs for this location
    rec_ids = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('_recordingMeta.csv'):
            m = pd.read_csv(os.path.join(DATA_DIR, f))
            if int(m['locationId'].iloc[0]) == loc_id:
                rec_ids.append(int(m['recordingId'].iloc[0]))
    
    # Get speed limit
    sl = pd.read_csv(os.path.join(DATA_DIR, f'{rec_ids[0]:02d}_recordingMeta.csv'))['speedLimit'].iloc[0]
    
    # Classify lanelets with GT
    ramp, main, ms = classify_lanelets(rec_ids, sl)
    print(f'  classified: ramp={len(ramp)}, main={len(main)}, main_speed={ms:.0f}')
    
    # Load SUMO net for edge filtering
    net_xml = get_map_file(loc_id)
    try:
        sn = sumolib.net.readNet(net_xml, withInternal=False)
    except Exception:
        sn = None
    
    all_trajs = []
    for rid in sorted(rec_ids):
        merges = find_merge_tracks(rid, ramp, main, sn, loc_id=loc_id)
        csv = pd.read_csv(os.path.join(DATA_DIR, f'{rid:02d}_tracks.csv'),
                         usecols=['trackId', 'frame', 'xCenter', 'yCenter', 'laneletId'], low_memory=False)
        for m in merges:
            tid = m['track_id']
            sub = csv[csv['trackId'] == int(tid)].sort_values('frame')
            if len(sub) < MIN_SEG:
                continue
            
            mi = m['merge_frame_idx']
            ll = sub['laneletId'].values
            
            # Refine: find exact laneletId transition
            actual_mi = mi
            for j in range(1, len(ll)):
                if ll[j-1] in ramp and ll[j] in main:
                    actual_mi = j
                    break
            
            # Verify: enough pre/post merge frames
            if actual_mi < MIN_SEG or (len(sub) - actual_mi) < MIN_SEG:
                continue
            
            all_trajs.append({
                'rid': int(rid), 'tid': int(tid),
                'merge_idx': int(actual_mi), 'loc_id': loc_id,
            })
    
    cache[str(loc_id)] = all_trajs
    print(f'  {len(all_trajs)} trajectories')

# Save cache
with open(CACHE_PATH, 'w') as f:
    json.dump(cache, f, indent=2)

print(f'\nSaved to {CACHE_PATH}')
for loc_id in range(7):
    split = 'Train' if loc_id in [0, 2, 4, 5, 6] else 'Test'
    print(f'  Loc {loc_id} [{split}]: {len(cache[str(loc_id)])}')
