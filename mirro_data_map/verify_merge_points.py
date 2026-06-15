#!/usr/bin/env python3
"""Merge point verification: plot sampled merge points on SUMO maps."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = '/share/home/u23516/code/meta_dreamer-main'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'mirro_data_map'))

SUMO_HOME = '/share/apps/sumo-1.20'
sys.path.insert(0, os.path.join(SUMO_HOME, 'share/sumo/tools'))
os.environ['SUMO_HOME'] = SUMO_HOME
import sumolib

DATA_DIR = '/share/home/u23516/data/exiD-dataset-v2.1/data'
CACHE_PATH = os.path.join(REPO, 'mirro_data_map', 'exid_merge_cache.json')
OUT_DIR = os.path.join(REPO, 'docs', 'thesis_figures', 'merge_verify')
os.makedirs(OUT_DIR, exist_ok=True)

MAP_DIR = os.path.join(REPO, 'mirro_data_map')
LOC_NAMES = {
    0: 'cologne_butzweiler', 1: 'cologne_fortiib', 2: 'aachen_brand',
    3: 'bergheim_roemer', 4: 'cologne_klettenberg', 5: 'aachen_laurensberg',
    6: 'merzenich_rather',
}

C_PRE = '#C3DDF1'    # pre-merge (light blue)
C_POST = '#4A7FD4'   # post-merge (blue)
C_MERGE = '#FF4444'  # merge point (red)
C_ARROW = '#228B22'  # merge zone arrow (green)

def get_map_file(loc_id):
    for name in [f'exid_loc{loc_id}_orig.net.xml', f'exid_loc{loc_id}.net.xml']:
        path = os.path.join(MAP_DIR, name)
        if os.path.exists(path):
            return path
    return None

def draw_road_network(ax, net):
    from matplotlib.patches import Polygon as MplPolygon
    for edge in net.getEdges():
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            poly = MplPolygon(shape, closed=False, facecolor='#E8E8E8',
                            edgecolor='#AAAAAA', linewidth=0.3, alpha=0.7)
            ax.add_patch(poly)

def get_crop_bounds(trajs, pad=80):
    all_x = np.concatenate([t['x'] for t in trajs])
    all_y = np.concatenate([t['y'] for t in trajs])
    xlo, xhi = np.percentile(all_x, [2, 98])
    ylo, yhi = np.percentile(all_y, [2, 98])
    return (xlo - pad, ylo - pad, xhi + pad, yhi + pad)

def load_trajs(entries):
    by_rec = {}
    for e in entries:
        by_rec.setdefault(e['rid'], []).append((e['tid'], e['merge_idx']))
    all_trajs = []
    for rid, tracks in sorted(by_rec.items()):
        csv = pd.read_csv(os.path.join(DATA_DIR, f'{rid:02d}_tracks.csv'),
                         usecols=['trackId', 'frame', 'xCenter', 'yCenter'], low_memory=False)
        for tid, mi in tracks:
            sub = csv[csv['trackId'] == tid].sort_values('frame')
            if len(sub) < 20:
                continue
            all_trajs.append({
                'tid': tid, 'rid': rid,
                'x': sub['xCenter'].values,
                'y': sub['yCenter'].values,
                'merge_idx': mi,
            })
    return all_trajs

with open(CACHE_PATH) as f:
    cache = json.load(f)

for loc_id_str in ['0', '1', '2', '3', '4', '5', '6']:
    loc_id = int(loc_id_str)
    entries = cache.get(loc_id_str, [])
    if not entries:
        continue
    print(f'Plotting Location {loc_id_str} ({len(entries)} trajectories) ...')
    
    net_xml = get_map_file(loc_id)
    net = sumolib.net.readNet(net_xml, withInternal=True)
    trajs = load_trajs(entries)
    bounds = get_crop_bounds(trajs)
    xlo, ylo, xhi, yhi = bounds
    
    aspect = (yhi - ylo) / (xhi - xlo)
    figw = 12
    figh = figw * aspect
    if figh > 10:
        figw = 10 / aspect
        figh = 10
    
    fig, ax = plt.subplots(1, 1, figsize=(figw, figh))
    draw_road_network(ax, net)
    
    rng = np.random.RandomState(42)
    sample_trajs = rng.choice(trajs, min(10, len(trajs)), replace=False)
    
    # Plot all trajectories faintly
    for t in trajs:
        mi = t['merge_idx']
        x, y = t['x'], t['y']
        if mi > 0:
            ax.plot(x[:mi+1], y[:mi+1], color=C_PRE, linewidth=0.2, alpha=0.15)
        if mi < len(x) - 1:
            ax.plot(x[mi:], y[mi:], color=C_POST, linewidth=0.2, alpha=0.15)
    
    # Highlight sample trajectories with numbered merge points
    for j, t in enumerate(sample_trajs):
        mi = t['merge_idx']
        x, y = t['x'], t['y']
        label = f'{j+1}'
        
        if mi > 0:
            ax.plot(x[:mi+1], y[:mi+1], color='#8BB8F5', linewidth=0.8, alpha=0.7)
        if mi < len(x) - 1:
            ax.plot(x[mi:], y[mi:], color='#6BA3E8', linewidth=0.8, alpha=0.7)
        
        if 0 <= mi < len(x):
            ax.scatter(x[mi], y[mi], s=150, c=C_MERGE, edgecolors='darkred',
                      linewidth=1.5, zorder=10, marker='o')
            ax.annotate(label, (x[mi], y[mi]), fontsize=9, fontweight='bold',
                       color='white', ha='center', va='center', zorder=11)
            
            post_idx = min(mi + 20, len(x) - 1)
            if post_idx > mi:
                ax.annotate('', xy=(x[post_idx], y[post_idx]), xytext=(x[mi], y[mi]),
                           arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2, alpha=0.8))
    
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect('equal')
    ax.axis('off')
    
    split = 'Train' if loc_id in [0, 2, 4, 5, 6] else 'Test'
    name = LOC_NAMES.get(loc_id, f'loc{loc_id}')
    
    legend_handles = [
        mpatches.Patch(color=C_PRE, label='Pre-merge (ramp)', alpha=0.7),
        mpatches.Patch(color=C_POST, label='Post-merge (highway)', alpha=0.7),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_MERGE,
                   markeredgecolor='darkred', markersize=8,
                   label='Merge point (laneletId transition)'),
        plt.Line2D([0], [0], color=C_ARROW, lw=2, label='Merge zone (+20 frames / 0.8s)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=7,
              framealpha=0.9, edgecolor='#ccc')
    
    ax.set_title(f'Location {loc_id}: {name} [{split}]\n'
                f'{len(trajs)} trajectories, {len(sample_trajs)} highlighted\n'
                f'Red circles = merge_idx (laneletId ramp->main transition)',
                fontsize=10, pad=8)
    
    out_path = os.path.join(OUT_DIR, f'loc{loc_id_str}_merge_verify.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path}')

print('\nAll done!')
