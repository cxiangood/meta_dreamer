#!/usr/bin/env python3
"""
Visualize road mask, ramp status, and road centerline overlay on BEV frames.

Usage:
    python aux_data/visualize_labels.py --npz mirro_data_map/exid_dreamer_data/rec61/track143.npz \
        --frames 0,50,100,200 --output docs/analysis/label_viz
"""

import os, sys, math, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def visualize_frame(ax, bev_crop, road_mask, ramp_mask, ramp_phase, frame_idx, merge_idx):
    """Show BEV + road mask overlay + ramp indicator."""
    h, w = bev_crop.shape[:2]
    # BEV
    ax.imshow(bev_crop)

    # Road mask overlay (green = road, red = ramp)
    road_u = np.array(Image.fromarray(road_mask).resize((w, h), Image.NEAREST))
    ramp_u = np.array(Image.fromarray(ramp_mask).resize((w, h), Image.NEAREST))

    # Road overlay: green for all road
    road_overlay = np.zeros((h, w, 3), dtype=np.float32)
    road_overlay[road_u > 0, 1] = 0.3  # green for road
    road_overlay[ramp_u > 0, 0] = 0.4  # red for ramp
    road_overlay[ramp_u > 0, 1] = 0.0

    ax.imshow(road_overlay, alpha=0.4)

    # Frame info
    phase = "RAMP" if ramp_phase else "MAIN"
    title = f"Frame {frame_idx} [{phase}]"
    if merge_idx > 0 and frame_idx == merge_idx:
        title += " ← MERGE"
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True, help='Path to .npz file')
    parser.add_argument('--frames', default='0,50,100,200,300,400',
                        help='Comma-separated frame indices to visualize')
    parser.add_argument('--output', default='docs/analysis/label_viz', help='Output prefix')
    args = parser.parse_args()

    data = dict(np.load(args.npz, allow_pickle=True))
    bev_images = data.get('bev_images')
    road_mask = data.get('road_mask')
    ramp_mask = data.get('ramp_mask')
    merge_idx = int(data.get('merge_frame_idx', -1))

    if bev_images is None:
        print("ERROR: no bev_images in npz")
        return
    if road_mask is None:
        print("ERROR: no road_mask in npz (run add_aux_labels.py first)")
        return

    frames = [int(x) for x in args.frames.split(',')]
    T = len(bev_images)
    frames = [f for f in frames if 0 <= f < T]
    n = len(frames)

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, fi in enumerate(frames):
        # Center crop BEV
        H, W = bev_images[fi].shape[:2]
        size = min(H, W)
        dh, dw = (H - size) // 2, (W - size) // 2
        crop = bev_images[fi, dh:dh + size, dw:dw + size]

        ramp_phase = bool(data.get('ramp_phase', np.zeros(T, dtype=bool))[fi])

        rm = road_mask[fi] if road_mask is not None else np.zeros((64, 64), dtype=np.uint8)
        ram = ramp_mask[fi] if ramp_mask is not None else np.zeros((64, 64), dtype=np.uint8)

        visualize_frame(axes[i], crop, rm, ram, ramp_phase, fi, merge_idx)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    out_path = f"{args.output}_{os.path.basename(args.npz).replace('.npz','')}.png"
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
