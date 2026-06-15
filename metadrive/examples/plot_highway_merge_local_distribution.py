"""
Distribution of ``localY`` (and ``localX`` for context) for laneId 1–3 inside **one**
``localY`` band — uses **full** ``Trajectory.csv``, no ego time window.
**说明文档** / ``Road.jpg`` custom frame: **horizontal = localY**, **vertical = localX** (meters).

Default band: sliding window of ``--band-width`` (m) along ``localY`` that contains
the most samples (lanes 1–3 only).

    python3 -m metadrive.examples.plot_highway_merge_local_distribution
    python3 -m metadrive.examples.plot_highway_merge_local_distribution \\
        --band-center 70 --band-width 20 --out docs/local_dist.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metadrive.scenario.highway_merge_in import default_dataset_dir

LANE_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e"}


def _load_trajectory(dataset_dir: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir, "Trajectory.csv")
    if not os.path.isfile(path):
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)
    use = [
        "localX",
        "localY",
        "laneId",
    ]
    df = pd.read_csv(path, usecols=use)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["localX", "localY", "laneId"])
    df["laneId"] = df["laneId"].astype(int)
    df = df[df["laneId"].isin((1, 2, 3))]
    return df


def _auto_band(ly: np.ndarray, width: float, step: float) -> tuple[float, float]:
    lo, hi = float(np.percentile(ly, 1.0)), float(np.percentile(ly, 99.0))
    if hi - lo < width * 1.5:
        return lo, min(hi, lo + width)
    best_n, y0b, y1b = -1, lo, lo + width
    y = lo
    while y + width <= hi:
        y0, y1 = y, y + width
        n = int(np.sum((ly >= y0) & (ly < y1)))
        if n > best_n:
            best_n, y0b, y1b = n, y0, y1
        y += step
    return y0b, y1b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--band-center", type=float, default=None, help="Fixed band center (localY); uses --band-width")
    p.add_argument("--band-width", type=float, default=15.0, help="Band width along localY (same units as CSV)")
    p.add_argument("--auto-step", type=float, default=2.0, help="Search step when auto-picking band")
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG (default: <repo>/docs/localY_lane_distribution.png)",
    )
    args = p.parse_args()

    dataset_dir = args.dataset_dir or default_dataset_dir()
    df = _load_trajectory(dataset_dir)
    ly_all = df["localY"].to_numpy(dtype=np.float64)

    if args.band_center is not None:
        half = 0.5 * args.band_width
        y0, y1 = float(args.band_center - half), float(args.band_center + half)
    else:
        y0, y1 = _auto_band(ly_all, args.band_width, args.auto_step)

    sub = df[(df["localY"] >= y0) & (df["localY"] < y1)].copy()
    if len(sub) < 50:
        print(f"Too few samples in band [{y0:.2f}, {y1:.2f}): {len(sub)}", file=sys.stderr)
        sys.exit(1)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = args.out or os.path.join(repo_root, "docs", "localY_lane_distribution.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # (1) User-asked: localY distribution per lane inside the same localY band
    ax = axes[0]
    bins = max(12, int(args.band_width))
    for lid in (1, 2, 3):
        v = sub.loc[sub["laneId"] == lid, "localY"].to_numpy(dtype=np.float64)
        if v.size == 0:
            continue
        ax.hist(
            v,
            bins=bins,
            range=(y0, y1),
            alpha=0.5,
            label=f"lane {lid} (n={v.size})",
            density=True,
            color=LANE_COLORS[lid],
        )
    ax.set_xlabel("localY")
    ax.set_ylabel("density")
    ax.set_title(f"localY density | same band [{y0:.1f}, {y1:.1f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) Same band: localX (lateral) — usually clearer separation between lanes
    ax = axes[1]
    for lid in (1, 2, 3):
        v = sub.loc[sub["laneId"] == lid, "localX"].to_numpy(dtype=np.float64)
        if v.size == 0:
            continue
        ax.hist(
            v,
            bins=40,
            alpha=0.5,
            label=f"lane {lid} (n={v.size})",
            density=True,
            color=LANE_COLORS[lid],
        )
    ax.set_xlabel("localX")
    ax.set_ylabel("density")
    ax.set_title("localX density in same band (lateral)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    mode = f"center={args.band_center}" if args.band_center is not None else "auto band (max count)"
    fig.suptitle(
        f"Highway-merge-in | full Trajectory (no time window) | laneId 1–3 | band: {mode}, width={args.band_width:g}\n"
        f"Samples in band: {len(sub)} / {len(df)} total rows",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
