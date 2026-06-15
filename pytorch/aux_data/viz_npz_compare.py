#!/usr/bin/env python3
"""Debug: compare manual vs camera model projections for BEV overlay alignment."""

import os, sys, math, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BEV_HEIGHT = 50.0
CAMERA_FOV = 65.0
BEV_W, BEV_H = 400, 300
CROP_SIZE = 300

# Camera model intrinsics
_FY = (BEV_H / 2.0) / math.tan(math.radians(CAMERA_FOV) / 2.0)
_CP = math.cos(math.radians(89.0))
_SP = math.sin(math.radians(89.0))

_VERT_EXTENT = 2.0 * BEV_HEIGHT * math.tan(math.radians(CAMERA_FOV / 2.0))


def center_crop_bev(bev_img):
    H, W = bev_img.shape[:2]
    size = min(H, W)
    dh, dw = (H - size) // 2, (W - size) // 2
    return bev_img[dh:dh + size, dw:dw + size]


def compute_headings(positions):
    T = len(positions)
    headings = np.zeros(T, dtype=np.float32)
    for i in range(T - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        headings[i] = math.atan2(dy, dx)
    headings[-1] = headings[-2] if T >= 2 else 0.0
    return headings


def project_v1(local_x, local_y, size=300, mpp=None):
    """Original: r = size/2 - x (what add_aux_labels.py used)"""
    if mpp is None:
        mpp = _VERT_EXTENT / size
    r = size / 2.0 - local_x / mpp
    c = size / 2.0 + local_y / mpp   # original add_aux_labels convention
    return c, r


def project_v2(local_x, local_y, size=300, mpp=None):
    """Fixed: r = size/2 - x (forward=UP), c = size/2 - y (L-R flip corrected)"""
    if mpp is None:
        mpp = _VERT_EXTENT / size
    r = size / 2.0 - local_x / mpp   # forward = UP (smaller row)
    c = size / 2.0 - local_y / mpp   # flipped: left → smaller col
    return c, r


def project_camera(local_x, local_y, size=300):
    """Camera model: Panda3D perspective projection (exact match to MetaDrive)."""
    X_cam = -local_y
    Y_cam = _CP * local_x + _SP * BEV_HEIGHT
    Z_cam = _SP * local_x - _CP * BEV_HEIGHT

    u_full = _FY * X_cam / Y_cam + BEV_W / 2.0
    v_full = _FY * Z_cam / Y_cam + BEV_H / 2.0

    v_flipped = (BEV_H - 1) - v_full
    u_crop = u_full - (BEV_W - CROP_SIZE) / 2.0

    if size != CROP_SIZE:
        scale = size / CROP_SIZE
        return u_crop * scale, v_flipped * scale
    return u_crop, v_flipped


def world_to_local(wx, wy, ego_x, ego_y, ego_heading):
    dx = wx - ego_x
    dy = wy - ego_y
    cos_h = math.cos(-ego_heading)
    sin_h = math.sin(-ego_heading)
    local_x = cos_h * dx - sin_h * dy
    local_y = sin_h * dx + cos_h * dy
    return local_x, local_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    data = dict(np.load(args.npz, allow_pickle=True))
    bev_images = data['bev_images']
    positions = data['positions']
    road_mask = data.get('road_mask')
    loc_id = int(data.get('location_id', -1))

    headings = compute_headings(positions)
    fi = args.frame
    bev_crop = center_crop_bev(bev_images[fi])
    mpp = _VERT_EXTENT / 300

    # Compute map offset
    import sumolib
    map_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'mirro_data_map')
    net_xml = os.path.join(map_dir, f'exid_loc{loc_id}_orig.net.xml')
    if not os.path.exists(net_xml):
        net_xml = os.path.join(map_dir, f'exid_loc{loc_id}.net.xml')
    net = sumolib.net.readNet(net_xml, withInternal=True)
    xmin, ymin, xmax, ymax = net.getBoundary()
    off_x = -(xmax + xmin) / 2
    off_y = -(ymax + ymin) / 2

    ego_x_meta = positions[fi, 0] + off_x
    ego_y_meta = positions[fi, 1] + off_y
    heading = headings[fi]

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    # (0): Raw BEV
    axes[0].imshow(bev_crop)
    axes[0].set_title("BEV (raw)", fontsize=10)
    axes[0].plot(300, 300, 'rx', markersize=12, markeredgewidth=2)  # off-screen, hide

    # (1): Road mask overlay (upscaled 64->300, as-is from .npz)
    axes[1].imshow(bev_crop)
    if road_mask is not None:
        rm = np.array(Image.fromarray(road_mask[fi]).resize((300, 300), Image.NEAREST))
        overlay = np.zeros((300, 300, 3), dtype=np.float32)
        overlay[rm > 0, 1] = 0.4
        axes[1].imshow(overlay, alpha=0.4)
    axes[1].set_title("BEV + road_mask (from .npz)", fontsize=10)

    # (2): Manual projection v2 (corrected L-R, orthographic)
    axes[2].imshow(bev_crop)
    for d in range(-200, 201, 25):
        lx, ly = d, 0
        c, r = project_v2(lx, ly, 300, mpp)
        if 0 <= c < 300 and 0 <= r < 300:
            axes[2].axhline(r, color='yellow', alpha=0.2, linewidth=0.5)
        lx, ly = 0, d
        c, r = project_v2(0, d, 300, mpp)
        if 0 <= c < 300 and 0 <= r < 300:
            axes[2].axvline(c, color='cyan', alpha=0.2, linewidth=0.5)

    for label, lx, ly in [("FWD 50m", 50, 0), ("BACK 50m", -50, 0),
                           ("L 50m", 0, 50), ("R 50m", 0, -50)]:
        c, r = project_v2(lx, ly, 300, mpp)
        if 0 <= c < 300 and 0 <= r < 300:
            color = 'yellow' if ly == 0 else 'magenta'
            axes[2].plot(c, r, 's', color=color, markersize=6)
            axes[2].text(c + 3, r - 3, label, color=color, fontsize=7)

    # Ego marker at (0,0) in local coords
    c0, r0 = project_v2(0, 0, 300, mpp)
    axes[2].plot(c0, r0, 'r*', markersize=10, markeredgewidth=1.5)
    axes[2].set_title(f"v2 (orthographic, r=150-lx/mpp)\nEgo=({c0:.0f},{r0:.0f})", fontsize=9)

    # (3): Camera model projection
    axes[3].imshow(bev_crop)
    for d in range(-200, 201, 25):
        lx, ly = d, 0
        c, r = project_camera(lx, ly, 300)
        if 0 <= c < 300 and 0 <= r < 300:
            axes[3].axhline(r, color='yellow', alpha=0.2, linewidth=0.5)
        lx, ly = 0, d
        c, r = project_camera(0, d, 300)
        if 0 <= c < 300 and 0 <= r < 300:
            axes[3].axvline(c, color='cyan', alpha=0.2, linewidth=0.5)

    for label, lx, ly in [("FWD 50m", 50, 0), ("BACK 50m", -50, 0),
                           ("L 50m", 0, 50), ("R 50m", 0, -50)]:
        c, r = project_camera(lx, ly, 300)
        if 0 <= c < 300 and 0 <= r < 300:
            color = 'yellow' if ly == 0 else 'magenta'
            axes[3].plot(c, r, 's', color=color, markersize=6)
            axes[3].text(c + 3, r - 3, label, color=color, fontsize=7)

    c0, r0 = project_camera(0, 0, 300)
    axes[3].plot(c0, r0, 'r*', markersize=10, markeredgewidth=1.5)
    axes[3].set_title(f"Camera model (perspective)\nEgo=({c0:.0f},{r0:.0f})", fontsize=9)

    # (4): Camera-vs-v2 difference
    axes[4].imshow(bev_crop)
    for d in range(-200, 201, 25):
        for ly in [0]:  # horizontal grid lines
            lx = d
            c2, r2 = project_v2(lx, ly, 300, mpp)
            cc, rc = project_camera(lx, ly, 300)
            if 0 <= c2 < 300 and 0 <= r2 < 300:
                axes[4].axhline(r2, color='yellow', alpha=0.15, linewidth=0.5)
            if 0 <= cc < 300 and 0 <= rc < 300:
                axes[4].axhline(rc, color='lime', alpha=0.15, linewidth=0.5)
        for ly in [0]:  # vertical grid lines, but iterate on ly
            pass
    for d in range(-200, 201, 25):
        lx, ly = 0, d
        c2, r2 = project_v2(lx, ly, 300, mpp)
        cc, rc = project_camera(0, d, 300)
        if 0 <= c2 < 300 and 0 <= r2 < 300:
            axes[4].axvline(c2, color='cyan', alpha=0.15, linewidth=0.5)
        if 0 <= cc < 300 and 0 <= rc < 300:
            axes[4].axvline(cc, color='lime', alpha=0.15, linewidth=0.5)

    # Mark key points with both models
    for label, lx, ly in [("F50", 50, 0), ("B50", -50, 0), ("L50", 0, 50), ("R50", 0, -50)]:
        c2, r2 = project_v2(lx, ly, 300, mpp)
        cc, rc = project_camera(lx, ly, 300)
        if 0 <= c2 < 300 and 0 <= r2 < 300:
            axes[4].plot(c2, r2, 's', color='cyan', markersize=5, alpha=0.7)
            if abs(c2-cc) > 0.5 or abs(r2-rc) > 0.5:
                axes[4].annotate('', xy=(cc, rc), xytext=(c2, r2),
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    axes[4].legend([plt.Line2D([0],[0],color='cyan'), plt.Line2D([0],[0],color='lime'),
                    plt.Line2D([0],[0],color='red')],
                   ['v2 (ortho)', 'camera (persp)', 'Δ'], fontsize=7, loc='upper right')
    axes[4].set_title(f"Overlay: cyan=v2, green=camera\nRed arrows = perspective-vs-ortho Δ", fontsize=9)

    for ax in axes:
        ax.axis('off')

    suptitle = (f"Frame {fi} | loc{loc_id} | ego_meta=({ego_x_meta:.0f},{ego_y_meta:.0f}) | "
                f"heading={math.degrees(heading):.1f}deg | offset=({off_x:.0f},{off_y:.0f})")
    plt.suptitle(suptitle, fontsize=11)
    plt.tight_layout()

    out = args.output or f'/tmp/viz_compare_f{fi}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
