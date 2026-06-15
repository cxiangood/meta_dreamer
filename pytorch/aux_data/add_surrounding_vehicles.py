#!/usr/bin/env python3
"""
Generate ego-centric surrounding vehicle labels for each ego merge trajectory.

For every frame of ego trajectory in exid_dreamer_data .npz files:
  - Find all vehicles in the same frame from exiD CSV
  - Filter: same-direction (velocity heading within ±45° of ego), skip stationary (<0.1 m/s)
  - Compute ego-centric relative (dx, dy) and relative velocity (vx, vy)
  - Select top-N nearest vehicles
  - Save as companion _surrounding.npz alongside the original .npz

Zero-cost label (from existing exiD data, no re-collection needed).

Output per file:
  surrounding_positions: (T, N, 2) float32 — ego-centric (dx, dy) in meters
  surrounding_velocities:  (T, N, 2) float32 — ego-centric (vx, vy) in m/s

N = n_vehicles (default 5). Padded with zeros if fewer than N vehicles present.

Usage (HPC):
    python aux_data/add_surrounding_vehicles.py \
        --npz-dir /share/home/u23516/data/exid_dreamer_data \
        --exid-dir /share/home/u23516/data/exiD-dataset-v2.1 \
        --n-vehicles 5 --loc 0 --workers 4
"""

import os, sys, argparse, glob, re
import numpy as np
from collections import defaultdict
from multiprocessing import Pool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz-dir', required=True, help='Path to exid_dreamer_data/')
    p.add_argument('--exid-dir', required=True, help='Path to exiD-dataset-v2.1/')
    p.add_argument('--n-vehicles', type=int, default=5, help='Number of nearest vehicles to keep')
    p.add_argument('--loc', type=int, default=None, help='Process single location (0-6)')
    p.add_argument('--limit', type=int, default=0, help='Max files per location')
    p.add_argument('--workers', type=int, default=1, help='Parallel workers')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def load_csv_index(exid_dir, recording_ids):
    """Build per-recording frame→vehicles lookup from exiD tracks.csv files.

    Returns: dict[recording_id] = dict[frameId] = list of (trackId, posX, posY, vx, vy, heading)
    """
    data_dir = os.path.join(exid_dir, 'data')
    index = {}
    for rid in sorted(recording_ids):
        csv_path = os.path.join(data_dir, f'{rid:02d}_tracks.csv')
        if not os.path.exists(csv_path):
            print(f"  WARN: CSV not found: {csv_path}")
            continue
        frames = defaultdict(list)
        with open(csv_path) as f:
            header = [h.strip() for h in f.readline().strip().split(',')]
            col_idx = {h: i for i, h in enumerate(header)}
            for line in f:
                row = [r.strip() for r in line.strip().split(',')]
                if len(row) < max(col_idx.values()) + 1:
                    continue
                fid = int(row[col_idx['frame']])
                tid = int(row[col_idx['trackId']])
                px = float(row[col_idx['xCenter']])
                py = float(row[col_idx['yCenter']])
                vx = float(row[col_idx['xVelocity']])
                vy = float(row[col_idx['yVelocity']])
                hdg = float(row[col_idx['heading']])
                frames[fid].append((tid, px, py, vx, vy, hdg))
        index[rid] = dict(frames)
        print(f"  rec {rid:02d}: {len(frames)} frames, {sum(len(v) for v in frames.values())} rows")
    return index


def find_ego_frame_mapping(ego_positions, csv_index, rid, ego_tid):
    """Find the CSV start frame that matches the first npz frame."""
    frames = csv_index.get(rid, {})
    ego_start = ego_positions[0]

    for fid, vehicles in frames.items():
        for tid, px, py, vx, vy, hdg in vehicles:
            if tid == ego_tid:
                dist = np.sqrt((px - ego_start[0]) ** 2 + (py - ego_start[1]) ** 2)
                if dist < 0.5:
                    return fid
    return None


def extract_surroundings(ego_positions, csv_index, rid, ego_tid, n_vehicles):
    """Extract ego-centric surrounding vehicle info for one trajectory.

    Filters:
      - Same-direction: velocity heading within ±45° of ego heading
      - Stationary: skip vehicles with speed < 0.1 m/s
      - Top-N nearest by distance

    Returns:
        positions: (T, N, 2) ego-centric (dx, dy)
        velocities: (T, N, 2) ego-centric (vx, vy)
    """
    T = len(ego_positions)
    frames = csv_index.get(rid, {})
    start_fid = find_ego_frame_mapping(ego_positions, csv_index, rid, ego_tid)
    if start_fid is None:
        return (np.zeros((T, n_vehicles, 2), dtype=np.float32),
                np.zeros((T, n_vehicles, 2), dtype=np.float32))

    # Compute ego headings
    headings = np.zeros(T, dtype=np.float32)
    for i in range(T - 1):
        dx = ego_positions[i + 1, 0] - ego_positions[i, 0]
        dy = ego_positions[i + 1, 1] - ego_positions[i, 1]
        headings[i] = np.arctan2(dy, dx)
    headings[-1] = headings[-2] if T >= 2 else 0.0

    surr_pos = np.zeros((T, n_vehicles, 2), dtype=np.float32)
    surr_vel = np.zeros((T, n_vehicles, 2), dtype=np.float32)

    for t in range(T):
        fid = start_fid + t
        vehicles = frames.get(fid, [])
        if not vehicles:
            continue

        ego_x, ego_y = ego_positions[t, 0], ego_positions[t, 1]
        ego_h = headings[t]
        cos_h = np.cos(-ego_h)
        sin_h = np.sin(-ego_h)

        nearest = []
        for o_tid, o_px, o_py, o_vx, o_vy, o_hdg in vehicles:
            if o_tid == ego_tid:
                continue
            # Same-direction filter: velocity heading within ±45° of ego heading
            o_speed = np.sqrt(o_vx ** 2 + o_vy ** 2)
            if o_speed < 0.1:
                continue
            angle_diff = abs(o_hdg - ego_h)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            if angle_diff > np.pi / 4:  # > 45°
                continue
            # World → ego-centric
            dx_w = o_px - ego_x
            dy_w = o_py - ego_y
            dx_e = cos_h * dx_w - sin_h * dy_w
            dy_e = sin_h * dx_w + cos_h * dy_w
            vx_e = cos_h * o_vx - sin_h * o_vy
            vy_e = sin_h * o_vx + cos_h * o_vy
            d = np.sqrt(dx_e ** 2 + dy_e ** 2)
            nearest.append((d, dx_e, dy_e, vx_e, vy_e))

        nearest.sort(key=lambda x: x[0])
        for i, (_, dx_e, dy_e, vx_e, vy_e) in enumerate(nearest[:n_vehicles]):
            surr_pos[t, i] = [dx_e, dy_e]
            surr_vel[t, i] = [vx_e, vy_e]

    return surr_pos, surr_vel


def process_one_file(args_tuple):
    """Process a single npz file. Returns (ok, msg)."""
    fpath, csv_index, n_vehicles, dry_run = args_tuple
    fname = os.path.basename(fpath)
    try:
        data = dict(np.load(fpath, allow_pickle=True))
        rid = int(data.get('recording_id', -1))
        ego_tid = int(data.get('track_id', -1))
        positions = data.get('positions')
        if rid < 0 or ego_tid < 0 or positions is None:
            return False, f"{fname}: missing metadata"
        if rid not in csv_index:
            return False, f"{fname}: rec {rid} not in CSV index"

        surr_pos, surr_vel = extract_surroundings(
            positions, csv_index, rid, ego_tid, n_vehicles
        )

        if dry_run:
            n_nonzero = (np.abs(surr_pos).sum(axis=(1, 2)) > 1e-6).sum()
            avg_veh = (np.abs(surr_pos).sum(axis=2) > 1e-6).sum(axis=1).mean() if n_nonzero > 0 else 0
            return True, f"dry-run: T={len(positions)}, frames_w_veh={n_nonzero}/{len(positions)}, avg_veh={avg_veh:.1f}"

        out_path = fpath.replace('.npz', '_surrounding.npz')
        np.savez_compressed(out_path,
                            surrounding_positions=surr_pos,
                            surrounding_velocities=surr_vel)
        return True, f"ok: T={len(positions)}"

    except Exception as e:
        return False, f"error: {e}"


def process_location(loc_id, npz_dir, csv_index, n_vehicles, limit=0,
                     dry_run=False, workers=1):
    """Process all npz files for one location."""
    pattern = os.path.join(npz_dir, f'loc{loc_id}', '**', 'track*.npz')
    npz_files = sorted(glob.glob(pattern, recursive=True))
    # Filter out _surrounding.npz companion files
    npz_files = [f for f in npz_files if '_surrounding' not in f]
    print(f"loc{loc_id}: {len(npz_files)} npz files")
    if limit > 0:
        npz_files = npz_files[:limit]

    # Build task list
    tasks = [(fp, csv_index, n_vehicles, dry_run) for fp in npz_files]

    cnt, skipped, errors = 0, 0, 0
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(process_one_file, tasks)
    else:
        results = [process_one_file(t) for t in tasks]

    for ok, msg in results:
        if ok:
            cnt += 1
        else:
            if 'missing metadata' in msg:
                errors += 1
                print(f"  ERR {msg}" if errors <= 3 else "", end="")
            else:
                skipped += 1

    print(f"  loc{loc_id}: {cnt} ok, {skipped} skipped, {errors} errors")
    return cnt


def main():
    args = parse_args()

    # Collect unique recording IDs
    npz_files = sorted(glob.glob(
        os.path.join(args.npz_dir, 'loc*', '**', 'track*.npz'), recursive=True
    ))
    recording_ids = set()
    for fpath in npz_files:
        m = re.search(r'rec(\d+)', fpath)
        if m:
            recording_ids.add(int(m.group(1)))
    # Also try from npz metadata (faster for single-loc mode)
    if args.loc is not None and not recording_ids:
        for d in range(20):
            csv_path = os.path.join(args.exid_dir, 'data', f'{d:02d}_tracks.csv')
            if os.path.exists(csv_path):
                recording_ids.add(d)

    print(f"Found {len(recording_ids)} recording IDs from {len(npz_files)} npz files")
    if not recording_ids:
        print("No recordings found!")
        return

    # Build CSV index
    print("Loading exiD CSV data...")
    csv_index = load_csv_index(args.exid_dir, recording_ids)

    # Process
    if args.loc is not None:
        locs = [args.loc]
    else:
        locs = sorted(set(
            int(re.search(r'loc(\d+)', f).group(1))
            for f in glob.glob(os.path.join(args.npz_dir, 'loc*'))
        ))

    total = 0
    for lid in locs:
        total += process_location(lid, args.npz_dir, csv_index,
                                  args.n_vehicles, limit=args.limit,
                                  dry_run=args.dry_run, workers=args.workers)
    print(f"\nTotal: {total} files processed")


if __name__ == "__main__":
    main()
