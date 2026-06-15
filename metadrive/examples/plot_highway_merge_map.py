"""
Plot ``map_features`` from Highway-merge-in (default: slab mode, same as replay).

Output: ``docs/map_slab_preview.png`` (under repo root containing ``metadrive/``).

    python3 -m metadrive.examples.plot_highway_merge_map
    python3 -m metadrive.examples.plot_highway_merge_map --map legacy --out /tmp/x.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metadrive.scenario.highway_merge_in import (
    OMIT_MAP_LANE_IDS,
    build_clean_merge_map_features,
    build_map_features_from_trajectory_window,
    build_map_features_slab_lane_statistics,
    default_dataset_dir,
)
from metadrive.type import MetaDriveType


def _load_window(dataset_dir: str, flip_y: bool):
    meta_path = os.path.join(dataset_dir, "TrackIDstate.csv")
    traj_path = os.path.join(dataset_dir, "Trajectory.csv")
    if not os.path.isfile(meta_path) or not os.path.isfile(traj_path):
        print(f"Missing CSV under {dataset_dir}", file=sys.stderr)
        sys.exit(1)
    meta = pd.read_csv(meta_path)
    sdc = int(meta.loc[meta["RampVehicle"], "trackId"].iloc[0]) if meta["RampVehicle"].any() else int(
        meta["trackId"].iloc[0]
    )
    row = meta.loc[meta["trackId"] == sdc].iloc[0]
    f0, nf = int(row["InitialFrame"]), int(row["TotalFrame"])
    f1 = f0 + nf - 1
    traj = pd.read_csv(traj_path)
    traj.columns = [c.strip() for c in traj.columns]
    window = traj[(traj["frameId"] >= f0) & (traj["frameId"] <= f1)].copy()
    if flip_y:
        window["posY"] = -window["posY"]
    return window, meta, sdc, (f0, f1)


def _features(window, meta, map_mode: str):
    if map_mode == "clean":
        return build_clean_merge_map_features(window, meta)
    if map_mode == "legacy":
        return build_map_features_from_trajectory_window(window)
    return build_map_features_slab_lane_statistics(window)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--map", choices=["slab", "legacy", "clean"], default="slab")
    parser.add_argument("--no-flip-y", action="store_true")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: <repo>/docs/map_preview_<map>.png)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or default_dataset_dir()
    window, meta, sdc, fr = _load_window(dataset_dir, flip_y=not args.no_flip_y)
    feats = _features(window, meta, args.map)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = args.out or os.path.join(repo_root, "docs", f"map_preview_{args.map}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=120)
    # faint background: subsample trajectory cloud
    xy = window[["posX", "posY"]].dropna().to_numpy()
    if xy.shape[0] > 8000:
        rng = np.random.default_rng(0)
        xy = xy[rng.choice(xy.shape[0], 8000, replace=False)]
    ax.scatter(xy[:, 0], xy[:, 1], s=1, c="#cccccc", alpha=0.25, label="trajectory samples")

    lane_colors = {"1": "#1f77b4", "2": "#2ca02c", "3": "#ff7f0e", "ramp": "#ff7f0e"}
    for fid, data in sorted(feats.items(), key=lambda x: x[0]):
        t = data.get("type", "")
        pl = np.asarray(data.get("polyline", []), dtype=np.float64)
        if pl.shape[0] < 2:
            continue
        is_edge = t == MetaDriveType.BOUNDARY_LINE
        c = lane_colors.get(fid, "#9467bd" if is_edge else "#333333")
        lw = 1.2 if is_edge else 2.4
        ls = "--" if is_edge else "-"
        label = f"{fid} ({'edge' if is_edge else 'lane'})"
        if fid in lane_colors and not is_edge:
            label = f"lane {fid}"
        ax.plot(pl[:, 0], pl[:, 1], color=c, linewidth=lw, linestyle=ls, label=label)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("posX (CSV, after optional flip_y)")
    ax.set_ylabel("posY (CSV)")
    omit_txt = ", ".join(str(x) for x in sorted(OMIT_MAP_LANE_IDS))
    ax.set_title(
        f"Highway-merge-in map preview  |  mode={args.map}  |  ego track {sdc}  |  frames {fr[0]}–{fr[1]}\n"
        f"omitted from map: laneId in {{{omit_txt}}}"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
