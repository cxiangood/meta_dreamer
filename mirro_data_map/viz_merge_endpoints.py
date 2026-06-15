#!/usr/bin/env python3
"""Visualize on-ramp endpoints and trajectory closest-approach points."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ['EXID_DATASET_DIR'] = '/share/home/u23516/data/exiD-dataset-v2.1'
SUMO_HOME = '/share/apps/sumo-1.20'
os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.insert(0, os.path.join(SUMO_HOME, 'share/sumo/tools'))
import sumolib

REPO = '/share/home/u23516/code/meta_dreamer-main'
sys.path.insert(0, os.path.join(REPO, 'mirro_data_map'))
DATA_DIR = '/share/home/u23516/data/exiD-dataset-v2.1/data'
MAP_DIR = os.path.join(REPO, 'mirro_data_map')
OUT_DIR = os.path.join(REPO, 'docs', 'thesis_figures', 'merge_verify')
os.makedirs(OUT_DIR, exist_ok=True)

LOC_NAMES = {0:'cologne_butzweiler',1:'cologne_fortiib',2:'aachen_brand',
             3:'bergheim_roemer',4:'cologne_klettenberg',5:'aachen_laurensberg',6:'merzenich_rather'}

_ONRAMP_EXTRA = {1:{"-23#2"},4:{"3#1"}}
_ONRAMP_EXCLUDE = {0:{"-3","-16#0"}}

def get_map_file(loc_id):
    for n in [f'exid_loc{loc_id}_orig.net.xml', f'exid_loc{loc_id}.net.xml']:
        p = os.path.join(MAP_DIR, n)
        if os.path.exists(p): return p
    return None

from collect_merge_data import classify_lanelets, find_merge_tracks

# Pick one loc to test
loc_id = 0
print(f'Location {loc_id}: {LOC_NAMES[loc_id]}')

# Load net
net = sumolib.net.readNet(get_map_file(loc_id), withInternal=False)

# Get recordings
rec_ids = []
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith('_recordingMeta.csv'):
        m = pd.read_csv(os.path.join(DATA_DIR, f))
        if int(m['locationId'].iloc[0]) == loc_id:
            rec_ids.append(int(m['recordingId'].iloc[0]))

sl = pd.read_csv(os.path.join(DATA_DIR, f'{rec_ids[0]:02d}_recordingMeta.csv'))['speedLimit'].iloc[0]
ramp_ll, main_ll, ms = classify_lanelets(rec_ids, sl)
print(f'ramp lanelets: {len(ramp_ll)}, main: {len(main_ll)}')

# Collect ramp trajectory coords for edge matching
ramp_coords = []
trajs_data = {}

for rid in sorted(rec_ids):
    merges = find_merge_tracks(rid, ramp_ll, main_ll, net, loc_id=loc_id)
    if not merges: continue
    csv = pd.read_csv(os.path.join(DATA_DIR, f'{rid:02d}_tracks.csv'),
                     usecols=['trackId','frame','xCenter','yCenter','laneletId'], low_memory=False)
    for m in merges:
        tid = m['track_id']
        sub = csv[csv['trackId']==int(tid)].sort_values('frame')
        mi = m['merge_frame_idx']
        ll = sub['laneletId'].values
        actual_mi = mi
        for j in range(1,len(ll)):
            if ll[j-1] in ramp_ll and ll[j] in main_ll:
                actual_mi = j; break
        x = sub['xCenter'].values.astype(float)
        y = sub['yCenter'].values.astype(float)
        if actual_mi>0:
            step = max(1,actual_mi//20)
            for i in range(0,actual_mi,step):
                ramp_coords.append((x[i],y[i]))
        trajs_data[(rid,tid)] = {'x':x,'y':y,'ll':ll,'mi':actual_mi,'rid':rid,'tid':tid}

ramp_pts = np.array(ramp_coords)
print(f'{len(trajs_data)} trajectories, {len(ramp_coords)} ramp coords')

# Find on-ramp edges and endpoints
extra = _ONRAMP_EXTRA.get(loc_id, set())
exclude = _ONRAMP_EXCLUDE.get(loc_id, set())
endpoints = {}  # edge_id -> (x, y)
onramp_edges = set()

for edge in net.getEdges():
    eid = edge.getID()
    if eid.startswith(':') or len(edge.getLanes())>2: continue
    if eid in exclude: continue
    for lane in edge.getLanes():
        shape = lane.getShape()
        if len(shape)<2: continue
        mid = shape[len(shape)//2]
        dists = np.sqrt(np.sum((ramp_pts-mid)**2, axis=1))
        if dists.min()<10:
            onramp_edges.add(eid)
            to_node = edge.getToNode()
            tc = to_node.getCoord()
            endpoints[eid] = (tc[0], tc[1])
            break

# Add extra edges
for eid in extra:
    if eid not in onramp_edges:
        onramp_edges.add(eid)
        for edge in net.getEdges():
            if edge.getID()==eid:
                to_node = edge.getToNode()
                tc = to_node.getCoord()
                endpoints[eid] = (tc[0], tc[1])

print(f'On-ramp edges: {len(onramp_edges)}')
for eid,ep in sorted(endpoints.items()):
    print(f'  {eid} -> ({ep[0]:.1f}, {ep[1]:.1f})')

# === PLOT ===
all_x = np.concatenate([t['x'] for t in trajs_data.values()])
all_y = np.concatenate([t['y'] for t in trajs_data.values()])
xlo,xhi = np.percentile(all_x,[2,98]); ylo,yhi = np.percentile(all_y,[2,98])
pad=80; xlo-=pad; xhi+=pad; ylo-=pad; yhi+=pad

aspect = (yhi-ylo)/(xhi-xlo)
figw=12; figh=figw*aspect
if figh>10: figw=10/aspect; figh=10

fig,ax = plt.subplots(1,1,figsize=(figw,figh))

# Draw road network
from matplotlib.patches import Polygon as MplPolygon
for edge in net.getEdges():
    for lane in edge.getLanes():
        shape = lane.getShape()
        if len(shape)<2: continue
        color = '#B8D4F0' if edge.getID() in onramp_edges else '#E8E8E8'
        alpha = 0.6 if edge.getID() in onramp_edges else 0.3
        poly = MplPolygon(shape, closed=False, facecolor=color, edgecolor='#AAAAAA', linewidth=0.3, alpha=alpha)
        ax.add_patch(poly)

# Draw on-ramp endpoints as BIG RED STARS
for eid,(ex,ey) in endpoints.items():
    ax.scatter(ex, ey, s=400, c='red', marker='*', edgecolors='darkred', linewidth=2, zorder=20)
    ax.annotate(f'endpoint\n{eid}', (ex,ey), fontsize=7, color='darkred', fontweight='bold',
               xytext=(10,10), textcoords='offset points', zorder=21)

# Draw 5 sample trajectories with closest-approach markers
rng = np.random.RandomState(42)
sample_keys = rng.choice(list(trajs_data.keys()), min(5,len(trajs_data)), replace=False)

for j,key in enumerate(sample_keys):
    t = trajs_data[key]
    x,y,mi = t['x'],t['y'],t['mi']
    
    # Full trajectory
    ax.plot(x, y, color='#333333', linewidth=0.5, alpha=0.3)
    
    # Find closest approach to each endpoint → best merge frame
    best_frame = mi
    best_dist = float('inf')
    best_ep = None
    
    search_s = max(0,int(mi*0.6))
    search_e = min(len(x),int(mi*1.4))
    
    for frame in range(search_s, search_e):
        px,py = x[frame],y[frame]
        for eid,(ex,ey) in endpoints.items():
            dist = np.sqrt((px-ex)**2+(py-ey)**2)
            if dist < best_dist:
                best_dist = dist
                best_frame = frame
                best_ep = eid
    
    # Draw trajectory segments
    if best_frame > 0:
        ax.plot(x[:best_frame+1], y[:best_frame+1], color='#4A7FD4', linewidth=1.5, alpha=0.8, zorder=5)
    if best_frame < len(x)-1:
        ax.plot(x[best_frame:], y[best_frame:], color='#C3DDF1', linewidth=1.5, alpha=0.8, zorder=5)
    
    # Closest point marker
    ax.scatter(x[best_frame], y[best_frame], s=120, c='orange', edgecolors='darkorange', linewidth=1.5, zorder=15, marker='D')
    ax.annotate(f'T{j+1} f={best_frame}', (x[best_frame],y[best_frame]), fontsize=7, fontweight='bold',
               xytext=(5,-10), textcoords='offset points', zorder=16)
    
    # Arrow from closest point to endpoint
    if best_ep in endpoints:
        ex,ey = endpoints[best_ep]
        ax.annotate('', xy=(ex,ey), xytext=(x[best_frame],y[best_frame]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
    
    print(f'  T{j+1} rid={t["rid"]} tid={t["tid"]}: lanelet_mi={mi}, closest_mi={best_frame}, dist_to_endpoint={best_dist:.1f}m, edge={best_ep}')

ax.set_xlim(xlo,xhi); ax.set_ylim(ylo,yhi)
ax.set_aspect('equal'); ax.axis('off')

legend = [
    mpatches.Patch(color='#B8D4F0', label='On-ramp edge'),
    mpatches.Patch(color='#E8E8E8', label='Other road'),
    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='red', markeredgecolor='darkred',
              markersize=12, label='On-ramp endpoint (merge point)'),
    plt.Line2D([0],[0], marker='D', color='w', markerfacecolor='orange', markeredgecolor='darkorange',
              markersize=8, label='Closest approach (merge_idx)'),
    plt.Line2D([0],[0], color='#4A7FD4', lw=2, label='Pre-merge'),
    plt.Line2D([0],[0], color='#C3DDF1', lw=2, label='Post-merge'),
]
ax.legend(handles=legend, loc='lower right', fontsize=7, framealpha=0.9)
ax.set_title(f'Location {loc_id}: {LOC_NAMES[loc_id]} — On-ramp Endpoints & Merge Points\n'
             f'Red stars = SUMO on-ramp edge endpoints (physical merge point)\n'
             f'Orange diamonds = trajectory closest approach to endpoint (= proposed merge_idx)',
             fontsize=10, pad=8)

out = os.path.join(OUT_DIR, f'loc{loc_id}_endpoints.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out}')
