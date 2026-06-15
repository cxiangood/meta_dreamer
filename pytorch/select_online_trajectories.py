"""
Select balanced trajectories for online DreamerV3 training by same-direction
traffic density, and visualize the data distribution.

Usage:
    python select_online_trajectories.py \
        --merge-cache mirro_data_map/exid_merge_cache.json \
        --data-dir /path/to/exiD-dataset-v2.1/data \
        --train-locs 0 2 4 5 6 \
        --max-per-loc 15 \
        --output mirro_data_map/exid_online_selection.json \
        --plot-dir ./plots_online_selection
"""

import argparse
import json
import math
import os
import sys
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Thesis plot style ─────────────────────────────────────────────────────
THESIS_COLORS = {
    "blue":   "#2166AC",
    "red":    "#B2182B",
    "green":  "#4DAF4A",
    "orange": "#FF7F00",
    "purple": "#984EA3",
    "grey":   "#999999",
}
THESIS_PALETTE = [THESIS_COLORS[c] for c in ["blue", "red", "green", "orange", "purple"]]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Trajectory statistics ─────────────────────────────────────────────────

def compute_traffic_density_from_df(tracks_csv, tid, merge_idx, window=50):
    """Mean number of same-direction vehicles within ±window frames of merge.
    Uses an already-loaded tracks dataframe (avoids per-trajectory CSV read)."""
    if merge_idx < 0:
        return 0.0, 0.0

    ego_sub = tracks_csv[tracks_csv["trackId"] == int(tid)].sort_values("frame")
    if len(ego_sub) == 0:
        return 0.0, 0.0

    f0, f1 = int(ego_sub["frame"].iloc[0]), int(ego_sub["frame"].iloc[-1])
    mf = f0 + merge_idx
    ws, we = max(f0, mf - window), min(f1, mf + window)

    if we <= ws:
        return 0.0, 0.0

    ego_win = ego_sub[(ego_sub["frame"] >= ws) & (ego_sub["frame"] <= we)]
    ego_h = ego_win["heading"].mean()
    ego_rad = math.radians(ego_h)
    ego_dir = np.array([math.cos(ego_rad), math.sin(ego_rad)])

    win_data = tracks_csv[(tracks_csv["frame"] >= ws) & (tracks_csv["frame"] <= we)]
    other = win_data[win_data["trackId"] != int(tid)]

    counts = []
    for frame in range(ws, we + 1):
        ft = other[other["frame"] == frame]
        c = 0
        headings = np.radians(ft["heading"].values)
        dots = np.dot(np.column_stack([np.cos(headings), np.sin(headings)]), ego_dir)
        c = int((dots > 0.5).sum())
        counts.append(c)
    density = float(np.mean(counts)) if counts else 0.0

    # Same-direction unique vehicles near merge
    same_dir_set = set()
    other_headings = np.radians(other["heading"].values)
    other_ids = other["trackId"].values
    dots_all = np.dot(np.column_stack([np.cos(other_headings), np.sin(other_headings)]), ego_dir)
    for j, dot in enumerate(dots_all):
        if dot > 0.5:
            same_dir_set.add(int(other_ids[j]))

    return density, float(len(same_dir_set))


def compute_stats_for_recording(rec_id, trajectories, data_dir):
    """Compute stats for all trajectories in one recording (single CSV read)."""
    csv_path = os.path.join(data_dir, f"{rec_id:02d}_tracks.csv")
    try:
        tracks_csv = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return [None] * len(trajectories)

    results = []
    for traj in trajectories:
        tid = traj["tid"]
        merge_idx = traj.get("merge_idx", -1)

        ego = tracks_csv[tracks_csv["trackId"] == int(tid)].sort_values("frame")
        if len(ego) == 0:
            results.append(None)
            continue

        f0 = int(ego["frame"].iloc[0])
        total_frames = len(ego)
        pre_frames = merge_idx
        post_frames = total_frames - merge_idx

        pre_speed = ego.iloc[:max(1, merge_idx)]["lonVelocity"].mean()
        post_speed = ego.iloc[merge_idx:]["lonVelocity"].mean() if post_frames > 0 else 0.0

        density, same_dir_count = compute_traffic_density_from_df(
            tracks_csv, tid, merge_idx)

        mf = f0 + merge_idx
        merge_row = ego[ego["frame"] == mf]
        lead_dhw = float(merge_row["leadDHW"].iloc[0]) if len(merge_row) > 0 else -1.0

        results.append({
            "rid": rec_id,
            "tid": tid,
            "merge_idx": merge_idx,
            "loc_id": traj.get("loc_id", -1),
            "total_frames": total_frames,
            "pre_frames": pre_frames,
            "post_frames": post_frames,
            "pre_speed_ms": float(pre_speed),
            "post_speed_ms": float(post_speed),
            "density": density,
            "same_dir_vehicles_near_merge": same_dir_count,
            "lead_dhw_at_merge": lead_dhw,
        })
    return results


# ── Selection ─────────────────────────────────────────────────────────────

def select_balanced(all_stats, train_locs, max_per_loc=15, seed=42):
    """Select balanced trajectories per location by density bins."""
    rng = np.random.RandomState(seed)
    selected = []

    for loc_id in train_locs:
        loc_stats = [s for s in all_stats if s["loc_id"] == loc_id]
        if not loc_stats:
            continue

        loc_stats.sort(key=lambda x: x["density"])
        n = len(loc_stats)

        # Split into 3 density bins
        third = n // 3
        bins = [loc_stats[:third], loc_stats[third:2 * third], loc_stats[2 * third:]]
        labels = ["Low", "Mid", "High"]

        per_bin = max(1, max_per_loc // 3)
        for label, bd in zip(labels, bins):
            if not bd:
                continue
            n_sel = min(per_bin, len(bd))
            indices = rng.choice(len(bd), n_sel, replace=False)
            for idx in indices:
                entry = bd[idx].copy()
                entry["density_bin"] = label
                selected.append(entry)

    return selected


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_density_distribution(all_stats, selected, train_locs, save_dir):
    """Histogram of traffic density: all vs selected."""
    densities_all = [s["density"] for s in all_stats if s["loc_id"] in train_locs]
    densities_sel = [s["density"] for s in selected]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(densities_all, bins=40, color=THESIS_COLORS["blue"], alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(densities_all), color=THESIS_COLORS["red"],
               linestyle="--", linewidth=1.5, label=f'Median={np.median(densities_all):.1f}')
    ax.set_xlabel("Same-direction traffic density (vehicles)")
    ax.set_ylabel("Count")
    ax.set_title("All Training Trajectories")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(densities_sel, bins=20, color=THESIS_COLORS["green"], alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(densities_sel), color=THESIS_COLORS["red"],
               linestyle="--", linewidth=1.5, label=f'Median={np.median(densities_sel):.1f}')
    ax.set_xlabel("Same-direction traffic density (vehicles)")
    ax.set_ylabel("Count")
    ax.set_title("Selected (Balanced)")
    ax.legend(frameon=False)

    fig.suptitle("Traffic Density Distribution: Before vs After Selection", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "density_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_density_by_location(all_stats, selected, train_locs, save_dir):
    """Box plot of density per location, before and after selection."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    loc_labels = [str(l) for l in train_locs]

    # Before selection
    data_before = []
    for loc_id in train_locs:
        data_before.append([s["density"] for s in all_stats if s["loc_id"] == loc_id])

    ax = axes[0]
    bp = ax.boxplot(data_before, labels=loc_labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], THESIS_PALETTE[:len(train_locs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Location")
    ax.set_ylabel("Traffic density (vehicles)")
    ax.set_title("All Trajectories")

    # After selection
    data_after = []
    for loc_id in train_locs:
        data_after.append([s["density"] for s in selected if s["loc_id"] == loc_id])

    ax = axes[1]
    bp = ax.boxplot(data_after, labels=loc_labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], THESIS_PALETTE[:len(train_locs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Location")
    ax.set_ylabel("Traffic density (vehicles)")
    ax.set_title("Selected (Balanced)")

    fig.suptitle("Traffic Density by Location", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "density_by_location.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_density_bins(selected, save_dir):
    """Bar chart: count per density bin."""
    bin_counts = defaultdict(int)
    for s in selected:
        bin_counts[s.get("density_bin", "?")] += 1

    bins_order = ["Low", "Mid", "High"]
    counts = [bin_counts.get(b, 0) for b in bins_order]
    colors = [THESIS_COLORS["green"], THESIS_COLORS["orange"], THESIS_COLORS["red"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(bins_order, counts, color=colors, alpha=0.8, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Traffic Density Bin")
    ax.set_ylabel("Number of Trajectories")
    ax.set_title(f"Selected Trajectories by Density Bin (N={sum(counts)})")
    plt.tight_layout()
    path = os.path.join(save_dir, "density_bins.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_merge_duration(all_stats, selected, save_dir):
    """Histogram of merge duration (pre-merge frames)."""
    pre_all = [s["pre_frames"] for s in all_stats]
    pre_sel = [s["pre_frames"] for s in selected]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(np.array(pre_all) / 25.0, bins=50, color=THESIS_COLORS["blue"],
            alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(pre_all) / 25.0, color=THESIS_COLORS["red"],
               linestyle="--", linewidth=1.5)
    ax.set_xlabel("Pre-merge duration (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"All Trajectories (median={np.median(pre_all) / 25:.1f}s)")

    ax = axes[1]
    ax.hist(np.array(pre_sel) / 25.0, bins=20, color=THESIS_COLORS["green"],
            alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(pre_sel) / 25.0, color=THESIS_COLORS["red"],
               linestyle="--", linewidth=1.5)
    ax.set_xlabel("Pre-merge duration (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Selected (median={np.median(pre_sel) / 25:.1f}s)")

    fig.suptitle("Pre-Merge Duration Distribution", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "merge_duration.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_speed_profile(selected, save_dir):
    """Speed comparison: pre-merge vs post-merge, colored by density bin."""
    fig, ax = plt.subplots(figsize=(7, 6))

    bin_colors = {
        "Low": THESIS_COLORS["green"],
        "Mid": THESIS_COLORS["orange"],
        "High": THESIS_COLORS["red"],
    }

    for s in selected:
        c = bin_colors.get(s.get("density_bin", "?"), THESIS_COLORS["grey"])
        ax.scatter(s["pre_speed_ms"], s["post_speed_ms"], c=c, alpha=0.5, s=20)

    # Diagonal
    lims = [0, max(max(s["pre_speed_ms"] for s in selected),
                   max(s["post_speed_ms"] for s in selected)) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Pre-merge speed (m/s)")
    ax.set_ylabel("Post-merge speed (m/s)")
    ax.set_title("Speed: Pre-merge vs Post-merge")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=bin_colors[b], alpha=0.5, label=b)
                       for b in ["Low", "Mid", "High"]]
    ax.legend(handles=legend_elements, frameon=False)

    plt.tight_layout()
    path = os.path.join(save_dir, "speed_profile.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_location_summary(all_stats, selected, train_locs, save_dir):
    """Summary panel per location."""
    n_locs = len(train_locs)
    fig, axes = plt.subplots(2, n_locs, figsize=(4 * n_locs, 8))
    if n_locs == 1:
        axes = axes.reshape(2, 1)

    for i, loc_id in enumerate(train_locs):
        loc_all = [s for s in all_stats if s["loc_id"] == loc_id]
        loc_sel = [s for s in selected if s["loc_id"] == loc_id]

        # Density
        ax = axes[0, i]
        ax.hist([s["density"] for s in loc_all], bins=20, alpha=0.6,
                color=THESIS_COLORS["blue"], label=f"All ({len(loc_all)})")
        ax.hist([s["density"] for s in loc_sel], bins=10, alpha=0.8,
                color=THESIS_COLORS["green"], label=f"Sel ({len(loc_sel)})")
        ax.set_xlabel("Density")
        ax.set_ylabel("Count")
        ax.set_title(f"Location {loc_id}")
        ax.legend(frameon=False, fontsize=8)

        # Pre-merge duration
        ax = axes[1, i]
        ax.hist(np.array([s["pre_frames"] for s in loc_all]) / 25.0, bins=20,
                alpha=0.6, color=THESIS_COLORS["blue"], label=f"All")
        ax.hist(np.array([s["pre_frames"] for s in loc_sel]) / 25.0, bins=10,
                alpha=0.8, color=THESIS_COLORS["green"], label=f"Sel")
        ax.set_xlabel("Pre-merge duration (s)")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Per-Location Summary", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "per_location_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Select balanced online training trajectories + visualize")
    parser.add_argument("--merge-cache", required=True,
                        help="Path to exid_merge_cache.json")
    parser.add_argument("--data-dir", required=True,
                        help="Path to exiD dataset (with NN_tracks.csv files)")
    parser.add_argument("--train-locs", type=int, nargs="+", default=[0, 2, 4, 5, 6])
    parser.add_argument("--max-per-loc", type=int, default=15)
    parser.add_argument("--output", required=True,
                        help="Output JSON path for selected trajectories")
    parser.add_argument("--plot-dir", default="./plots_online_selection")
    parser.add_argument("--cache-stats", default=None,
                        help="Cache computed stats to JSON (avoids re-computing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    # Load merge cache
    with open(args.merge_cache) as f:
        merge_cache = json.load(f)
    print(f"Loaded merge cache: {sum(len(v) for v in merge_cache.values())} trajectories")

    # Load or compute stats
    if args.cache_stats and os.path.exists(args.cache_stats):
        with open(args.cache_stats) as f:
            stats_list = json.load(f)
        print(f"Loaded cached stats: {len(stats_list)} trajectories")
    else:
        # Group all training trajectories by recording ID
        all_items = []
        for loc_id in args.train_locs:
            items = merge_cache.get(str(loc_id), [])
            all_items.extend(items)
            print(f"  Loc {loc_id}: {len(items)} trajectories")

        # Group by recording
        by_recording = defaultdict(list)
        for traj in all_items:
            by_recording[traj["rid"]].append(traj)

        unique_recs = sorted(by_recording.keys())
        print(f"  {len(unique_recs)} unique recordings, {len(all_items)} total trajectories")

        stats_list = []
        for ri, rec_id in enumerate(unique_recs):
            trajs = by_recording[rec_id]
            print(f"  Rec {rec_id:02d}: {len(trajs)} traj ... ", end="", flush=True)
            results = compute_stats_for_recording(rec_id, trajs, args.data_dir)
            valid = [r for r in results if r is not None]
            stats_list.extend(valid)
            print(f"{len(valid)} valid")

        print(f"  Total: {len(stats_list)}/{len(all_items)} trajectories with stats")

        if args.cache_stats:
            with open(args.cache_stats, "w") as f:
                json.dump(stats_list, f)
            print(f"  Cached stats: {args.cache_stats}")

    # Select
    selected = select_balanced(stats_list, args.train_locs, args.max_per_loc, args.seed)

    # Print summary
    print(f"\nSelected {len(selected)} trajectories:")
    for loc_id in args.train_locs:
        loc_sel = [s for s in selected if s["loc_id"] == loc_id]
        bins = defaultdict(int)
        for s in loc_sel:
            bins[s["density_bin"]] += 1
        print(f"  Loc {loc_id}: {len(loc_sel)} ({dict(bins)})")

    # Save
    output_data = {str(k): [] for k in args.train_locs}
    for s in selected:
        output_data[str(s["loc_id"])].append({
            "rid": s["rid"],
            "tid": s["tid"],
            "merge_idx": s["merge_idx"],
            "loc_id": s["loc_id"],
            "density_bin": s["density_bin"],
            "density": s["density"],
        })

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved selection to: {args.output}")

    # Plots
    print("\nGenerating plots...")
    plot_density_distribution(stats_list, selected, args.train_locs, args.plot_dir)
    plot_density_by_location(stats_list, selected, args.train_locs, args.plot_dir)
    plot_density_bins(selected, args.plot_dir)
    plot_merge_duration(stats_list, selected, args.plot_dir)
    plot_speed_profile(selected, args.plot_dir)
    plot_per_location_summary(stats_list, selected, args.train_locs, args.plot_dir)

    print(f"\nAll plots saved to: {args.plot_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
