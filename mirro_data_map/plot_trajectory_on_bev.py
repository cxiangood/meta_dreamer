"""
Plot trajectory data on top of the BEV map to verify alignment.

Draws:
- Lane centerlines (Lane 1, 2, 3 from map_features)
- All trajectory points for ego (trackId=132) and nearby vehicles
- Colored by laneId

    python3 mirro_data_map/plot_trajectory_on_bev.py
"""
import math
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_DIR = "/Users/jiojio/Documents/课题组/毕设/mirro_dataset_on_ramp/Highway-merge-in"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

LANE_WIDTH_MAIN = 3.75
LANE_WIDTH_RAMP = 3.0


def to_sumo(localY, localX):
    return localY, -localX


def main():
    traj = pd.read_csv(os.path.join(DATASET_DIR, "Trajectory.csv"))
    traj.columns = [c.strip() for c in traj.columns]
    meta = pd.read_csv(os.path.join(DATASET_DIR, "TrackIDstate.csv"))

    # Ego = first ramp vehicle
    sdc_track_id = 132
    ego_row = meta.loc[meta["trackId"] == sdc_track_id].iloc[0]
    f0 = int(ego_row["InitialFrame"])
    f1 = f0 + int(ego_row["TotalFrame"]) - 1
    window = traj[(traj["frameId"] >= f0) & (traj["frameId"] <= f1)].copy()

    print(f"Ego: trackId={sdc_track_id}, frames=[{f0}, {f1}], {f1-f0+1} steps")
    print(f"Vehicles in window: {sorted(window['trackId'].unique())}")

    # ── Map geometry ──
    lane1_sy = -(LANE_WIDTH_MAIN / 2)           # 1.875
    lane2_sy = -(LANE_WIDTH_MAIN + LANE_WIDTH_MAIN / 2)  # 5.625
    road_x_start, road_x_end = 220.0, -10.0

    # Ramp centerline from data
    ramp_data = traj[traj["laneId"] == 3].dropna(subset=["localX", "localY"])
    lane2_data = traj[traj["laneId"] == 2].dropna(subset=["localX", "localY"])

    slab_step = 10.0
    ly_vals = ramp_data["localY"].values
    ly_min, ly_max = np.percentile(ly_vals, 1), np.percentile(ly_vals, 99)
    edges = np.arange(math.floor(ly_min / slab_step) * slab_step, ly_max + slab_step, slab_step)

    ramp_slabs = []
    for i in range(len(edges) - 1):
        m = (ramp_data["localY"] >= edges[i]) & (ramp_data["localY"] < edges[i + 1])
        if m.sum() < 5:
            continue
        ramp_slabs.append((0.5 * (edges[i] + edges[i + 1]),
                           float(ramp_data.loc[m, "localX"].median())))

    lane2_slabs = {}
    for i in range(len(edges) - 1):
        m = (lane2_data["localY"] >= edges[i]) & (lane2_data["localY"] < edges[i + 1])
        if m.sum() < 5:
            continue
        mid = 0.5 * (edges[i] + edges[i + 1])
        lane2_slabs[mid] = float(lane2_data.loc[m, "localX"].median())

    ramp_slabs = np.array(ramp_slabs)
    ramp_shape_pts = []
    for ly, lx in ramp_slabs[np.argsort(-ramp_slabs[:, 0])]:
        sx, sy = to_sumo(ly, lx)
        ramp_shape_pts.append((sx, sy))

    last_sx, last_sy = ramp_shape_pts[-1]
    merge_end_ly = 10.0
    merge_target_lx = lane2_slabs.get(merge_end_ly, -5.625)
    merge_sx, merge_sy = to_sumo(merge_end_ly, merge_target_lx)
    for i in range(1, 9):
        t = i / 8
        ts_v = 3 * t ** 2 - 2 * t ** 3
        ramp_shape_pts.append((last_sx + ts_v * (merge_sx - last_sx),
                               last_sy + ts_v * (merge_sy - last_sy)))

    ramp_pts = np.array(ramp_shape_pts)

    # ── Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(18, 8), dpi=150)

    # Lane centerlines
    ax.plot([road_x_start, road_x_end], [lane1_sy, lane1_sy], 'b-', lw=2, label='Lane 1 centerline')
    ax.plot([road_x_start, road_x_end], [lane2_sy, lane2_sy], 'g-', lw=2, label='Lane 2 centerline')
    ax.plot(ramp_pts[:, 0], ramp_pts[:, 1], 'orange', lw=2, label='Ramp (Lane 3) centerline')

    # Lane boundaries
    b12_y = (lane1_sy + lane2_sy) / 2
    ax.axhline(b12_y, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(lane1_sy - LANE_WIDTH_MAIN / 2, color='gray', ls='-', lw=0.8, alpha=0.5)
    ax.axhline(lane2_sy + LANE_WIDTH_MAIN / 2, color='gray', ls='-', lw=0.8, alpha=0.5)

    # Trajectory points (all in window, colored by laneId)
    colors = {1: '#1f77b4', 2: '#2ca02c', 3: '#ff7f0e'}
    for lid in [1, 2, 3]:
        sub = window[window["laneId"] == lid].dropna(subset=["localX", "localY"])
        if sub.empty:
            continue
        sx = sub["localY"].values
        sy = -sub["localX"].values
        ax.scatter(sx, sy, s=0.5, alpha=0.3, c=colors[lid], label=f'laneId={lid} trajectories')

    # Ego trajectory highlighted
    ego_trail = window[window["trackId"] == sdc_track_id].dropna(subset=["localX", "localY"]).sort_values("frameId")
    if not ego_trail.empty:
        ex = ego_trail["localY"].values
        ey = -ego_trail["localX"].values
        ax.plot(ex, ey, 'r-', lw=2, label=f'Ego (trackId={sdc_track_id})')
        ax.scatter(ex[0], ey[0], c='red', s=100, marker='*', zorder=10, label='Ego start')
        ax.scatter(ex[-1], ey[-1], c='darkred', s=100, marker='s', zorder=10, label='Ego end')

    # All ramp vehicle trajectories in this window
    ramp_ids = meta.loc[meta["RampVehicle"], "trackId"].astype(int).unique()
    for tid in ramp_ids:
        if tid == sdc_track_id:
            continue
        sub = window[window["trackId"] == tid].dropna(subset=["localX", "localY"]).sort_values("frameId")
        if sub.empty:
            continue
        sx = sub["localY"].values
        sy = -sub["localX"].values
        ax.plot(sx, sy, 'm-', lw=1.5, alpha=0.7)

    ax.set_xlabel("SUMO X (= localY, meters)")
    ax.set_ylabel("SUMO Y (= -localX, meters)")
    ax.set_title(f"Highway Merge: Trajectories on Map | Ego=trackId {sdc_track_id} (ramp) | {f1-f0+1} frames")
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # traffic flows right-to-left (high localY → low localY)

    out = os.path.join(OUTPUT_DIR, "trajectory_on_bev.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
