"""
论文 Fig 1: 各 Location 汇入场景图（分文件夹生成）。

生成两个文件夹:
  fig1_scenes/       — 7 个 Location BEV 场景，标注匝道段和汇入区域
  fig1_trajectories/ — 同上 + 叠加所有汇入轨迹（匝道=蓝, 汇入点=橙, 汇入后=深蓝）

Location 0 特殊处理: 只画通过 lane 过滤的轨迹（主路 _0 可接受, _1/_2 排除）。

用法:
    python3 mirro_data_map/plot_merge_scenes.py
    python3 mirro_data_map/plot_merge_scenes.py --loc 0
    python3 mirro_data_map/plot_merge_scenes.py --trajectories-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimSun", "STSong"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
})

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo",
)
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
os.environ["SUMO_HOME"] = SUMO_HOME

import sumolib
from mirro_data_map.collect_merge_data import (
    classify_lanelets, find_merge_tracks, get_map_file,
    load_lanelet2_onramp_ids,
    LOC_NAMES, DATA_DIR, MAP_DIR,
)

OUT_BASE = os.path.join(REPO_ROOT, "docs", "thesis_figures")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "exid_merge_cache.json")

# ── 配色 (thesis-plot-style skill) ──
C_RAMP = "#4A7FD4"         # 明亮蓝（比浅蓝紫更显眼）
C_MERGE_POINT = "#F8CA9C"  # 浅橙
C_POST_MERGE = "#C3DDF1"   # 雾霾浅蓝
C_HIGHWAY = "#DCEAF7"      # 淡青蓝
C_ROAD_EDGE = "#576E90"    # 深雾霾蓝
C_ROAD_FILL = "#EDF3F4"    # 极浅青白


def _lane_polygon(shape, width):
    """从车道中心线和宽度计算左右边界多边形。"""
    n = len(shape)
    if n < 2:
        return None, None
    hw = max(width, 1.5) / 2.0
    left, right = [], []
    for i in range(n):
        if i == 0:
            dx = shape[1][0] - shape[0][0]
            dy = shape[1][1] - shape[0][1]
        elif i == n - 1:
            dx = shape[-1][0] - shape[-2][0]
            dy = shape[-1][1] - shape[-2][1]
        else:
            dx = shape[i + 1][0] - shape[i - 1][0]
            dy = shape[i + 1][1] - shape[i - 1][1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            nx, ny = 0.0, 1.0
        else:
            nx, ny = -dy / length, dx / length
        left.append((shape[i][0] + nx * hw, shape[i][1] + ny * hw))
        right.append((shape[i][0] - nx * hw, shape[i][1] - ny * hw))
    poly_x = [p[0] for p in left] + [p[0] for p in reversed(right)]
    poly_y = [p[1] for p in left] + [p[1] for p in reversed(right)]
    return poly_x, poly_y


def draw_road_network(ax, net):
    """绘制 SUMO 道路网络，使用车道宽度构建真实多边形。"""
    for edge in net.getEdges():
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            width = lane.getWidth()
            px, py = _lane_polygon(shape, width)
            if px is None:
                continue
            ax.fill(px, py, color=C_ROAD_FILL, alpha=0.6, linewidth=0)
            ax.plot(px + [px[0]], py + [py[0]],
                    color=C_ROAD_EDGE, linewidth=0.3, alpha=0.4)


def draw_ramp_highlight(ax, net, onramp_edge_ids):
    """高亮匝道 SUMO edge（仅限 on-ramp，不含 off-ramp）。"""
    for edge in net.getEdges():
        if edge.getID() not in onramp_edge_ids:
            continue
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            width = lane.getWidth()
            px, py = _lane_polygon(shape, width)
            if px is None:
                continue
            ax.fill(px, py, color=C_RAMP, alpha=0.45, linewidth=0)
            ax.plot(px + [px[0]], py + [py[0]],
                    color=C_RAMP, linewidth=1.2, alpha=0.7)


# Location-specific ramp edge corrections
_ONRAMP_EXTRA = {
    1: {"-23#2"},
    4: {"3#1"},
}
_ONRAMP_EXCLUDE = {
    0: {"-3", "-16#0"},
}


def get_onramp_sumo_edges(loc_id, net, trajs):
    """根据汇入轨迹匝道阶段的坐标定位 on-ramp SUMO edge。"""
    extra = _ONRAMP_EXTRA.get(loc_id, set())
    exclude = _ONRAMP_EXCLUDE.get(loc_id, set())

    ramp_pts = []
    for t in trajs:
        mi = t["merge_idx"]
        if mi > 0:
            step = max(1, mi // 20)
            for i in range(0, mi, step):
                ramp_pts.append((t["x"][i], t["y"][i]))
    if not ramp_pts:
        return extra
    ramp_pts = np.array(ramp_pts)

    onramp_edges = set()
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(":") or len(edge.getLanes()) > 2:
            continue
        if eid in exclude:
            continue
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            mid = shape[len(shape) // 2]
            dists = np.sqrt(np.sum((ramp_pts - mid) ** 2, axis=1))
            if dists.min() < 10:
                onramp_edges.add(eid)
                break
    return (onramp_edges | extra) - exclude


def get_merge_crop_bounds(trajs, pad=80):
    """从轨迹坐标计算汇入区域的裁剪范围（用百分位排除离群点）。"""
    all_x = np.concatenate([t["x"] for t in trajs])
    all_y = np.concatenate([t["y"] for t in trajs])
    xlo, xhi = np.percentile(all_x, [2, 98])
    ylo, yhi = np.percentile(all_y, [2, 98])
    return (xlo - pad, ylo - pad, xhi + pad, yhi + pad)


def _load_from_cache(loc_id):
    """从缓存文件加载轨迹元数据，返回 [(rid, tid, merge_idx), ...] 或 None。"""
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH) as f:
        data = json.load(f)
    entries = data.get(str(loc_id))
    if entries is None:
        return None
    return [(e["rid"], e["tid"], e["merge_idx"]) for e in entries]


def _build_trajs_from_cache(entries):
    """从缓存条目 + 原始 CSV 重建轨迹坐标。"""
    by_rec = {}
    for rid, tid, mi in entries:
        by_rec.setdefault(rid, []).append((tid, mi))

    all_trajs = []
    for rid, tracks in sorted(by_rec.items()):
        csv = pd.read_csv(
            os.path.join(DATA_DIR, f"{rid:02d}_tracks.csv"),
            usecols=["trackId", "frame", "xCenter", "yCenter"],
        )
        for tid, mi in tracks:
            sub = csv[csv["trackId"] == tid].sort_values("frame")
            if len(sub) < 20:
                continue
            all_trajs.append({
                "tid": tid, "rid": rid,
                "x": sub["xCenter"].values,
                "y": sub["yCenter"].values,
                "merge_idx": mi,
            })
    return all_trajs


def find_merge_trajectories(loc_id, use_cache=True):
    """获取某个 location 的所有汇入轨迹。优先使用缓存。"""
    # 尝试从缓存加载
    if use_cache:
        entries = _load_from_cache(loc_id)
        if entries is not None:
            return _build_trajs_from_cache(entries)

    # 完整流水线（无缓存）
    rec_ids = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith("_recordingMeta.csv"):
            m = pd.read_csv(os.path.join(DATA_DIR, f))
            if int(m["locationId"].iloc[0]) == loc_id:
                rec_ids.append(int(m["recordingId"].iloc[0]))

    sl = pd.read_csv(
        os.path.join(DATA_DIR, f"{rec_ids[0]:02d}_recordingMeta.csv")
    )["speedLimit"].iloc[0]
    ramp, main, ms = classify_lanelets(rec_ids, sl)

    try:
        net_xml = get_map_file(loc_id)
        sn = sumolib.net.readNet(net_xml, withInternal=False)
    except Exception:
        sn = None

    all_trajs = []
    for rid in sorted(rec_ids):
        merges = find_merge_tracks(rid, ramp, main, sn, loc_id=loc_id)
        csv = pd.read_csv(
            os.path.join(DATA_DIR, f"{rid:02d}_tracks.csv"),
            usecols=["trackId", "frame", "xCenter", "yCenter", "laneletId"],
        )
        for m in merges:
            tid = m["track_id"]
            sub = csv[csv["trackId"] == tid].sort_values("frame")
            if len(sub) < 20:
                continue
            mi = m["merge_frame_idx"]
            ll = sub["laneletId"].values
            x = sub["xCenter"].values
            y = sub["yCenter"].values

            # 找准确的 merge frame (lanelet 转换帧)
            actual_mi = mi
            for j in range(1, len(ll)):
                if ll[j - 1] in ramp and ll[j] in main:
                    actual_mi = j
                    break

            all_trajs.append({
                "tid": tid, "rid": rid,
                "x": x, "y": y, "ll": ll,
                "merge_idx": actual_mi,
            })
    return all_trajs


def plot_scene(loc_id, trajectories=False):
    """绘制单个 Location 的汇入场景图（裁剪至汇入区域）。"""
    name = LOC_NAMES[loc_id]
    net_xml = get_map_file(loc_id)
    net = sumolib.net.readNet(net_xml, withInternal=True)

    # 始终计算轨迹（用于裁剪范围和匝道定位）
    trajs = find_merge_trajectories(loc_id)
    print(f"  {len(trajs)} trajectories")

    # 裁剪范围
    if trajs:
        bounds = get_merge_crop_bounds(trajs)
    else:
        xmin, ymin, xmax, ymax = net.getBoundary()
        bounds = (xmin - 50, ymin - 50, xmax + 50, ymax + 50)

    xlo, ylo, xhi, yhi = bounds
    aspect = (yhi - ylo) / (xhi - xlo)
    figw = 10
    figh = figw * aspect
    if figh > 8:
        figw = 8 / aspect
        figh = 8

    fig, ax = plt.subplots(1, 1, figsize=(figw, figh))

    # 画道路网络（统一风格，真实车道宽度多边形）
    draw_road_network(ax, net)

    # 场景图: 高亮匝道（仅 on-ramp），图例只标注 Ramp
    if not trajectories and trajs:
        onramp_edges = get_onramp_sumo_edges(loc_id, net, trajs)
        if onramp_edges:
            draw_ramp_highlight(ax, net, onramp_edges)
        legend_handles = [
            mpatches.Patch(facecolor=C_ROAD_FILL, label="Highway",
                           alpha=0.7, edgecolor=C_ROAD_EDGE),
            mpatches.Patch(color=C_RAMP, label="On-ramp", alpha=0.7),
        ]
    else:
        legend_handles = [
            mpatches.Patch(color=C_POST_MERGE, label="Ramp", alpha=0.7),
            mpatches.Patch(color=C_MERGE_POINT, label="Merge Point", alpha=0.7),
            mpatches.Patch(color=C_RAMP, label="Post-merge", alpha=0.7),
        ]

    # 轨迹图: 画汇入轨迹（匝道=浅蓝, 汇入后=亮蓝）
    if trajectories:
        for t in trajs:
            mi = t["merge_idx"]
            x, y = t["x"], t["y"]

            # 匝道段（浅蓝）
            if mi > 0:
                ax.plot(x[:mi + 1], y[:mi + 1], color=C_POST_MERGE, linewidth=0.3, alpha=0.4)
            # 汇入点
            if mi < len(x):
                ax.plot(x[mi], y[mi], ".", color=C_MERGE_POINT, markersize=2, alpha=0.6)
            # 汇入后（亮蓝）
            if mi < len(x) - 1:
                ax.plot(x[mi:], y[mi:], color=C_RAMP, linewidth=0.3, alpha=0.4)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.axis("off")

    split = "Train" if loc_id in [0, 2, 4, 5, 6] else "Test"

    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#ccc")

    ax.set_title(f"Location {loc_id}: {name} [{split}]", fontsize=12, pad=10)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=int, default=None, help="Only plot one location")
    parser.add_argument("--scenes-only", action="store_true")
    parser.add_argument("--trajectories-only", action="store_true")
    parser.add_argument("--build-cache", action="store_true",
                        help="Run full pipeline and save trajectory cache")
    args = parser.parse_args()

    locs = [args.loc] if args.loc is not None else list(range(7))

    # 构建缓存模式: 运行完整流水线并保存
    if args.build_cache:
        cache = {}
        MIN_SEG = 20  # 匝道/汇入后各至少 20 帧 (0.8s)
        for loc_id in locs:
            name = LOC_NAMES[loc_id]
            print(f"\nBuilding cache for Location {loc_id}: {name}")
            trajs = find_merge_trajectories(loc_id, use_cache=False)
            filtered = [
                t for t in trajs
                if t["merge_idx"] >= MIN_SEG
                and (len(t["x"]) - t["merge_idx"]) >= MIN_SEG
            ]
            cache[str(loc_id)] = [
                {"rid": int(t["rid"]), "tid": int(t["tid"]),
                 "merge_idx": int(t["merge_idx"]), "loc_id": loc_id}
                for t in filtered
            ]
            print(f"  {len(filtered)}/{len(trajs)} trajectories (removed {len(trajs) - len(filtered)} short)")
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"\nCache saved to {CACHE_PATH}")
        print("Summary:")
        for loc_id in locs:
            split = "Train" if loc_id in [0, 2, 4, 5, 6] else "Test"
            n = len(cache[str(loc_id)])
            print(f"  Loc {loc_id} [{split}]: {n}")
        return

    for loc_id in locs:
        name = LOC_NAMES[loc_id]
        print(f"\nLocation {loc_id}: {name}")

        if not args.trajectories_only:
            print("  Drawing scene ...")
            fig = plot_scene(loc_id, trajectories=False)
            out_dir = os.path.join(OUT_BASE, "fig1_scenes")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"loc{loc_id}_merge_scene.png")
            fig.savefig(path)
            plt.close(fig)
            print(f"  -> {path}")

        if not args.scenes_only:
            print("  Drawing trajectories ...")
            fig = plot_scene(loc_id, trajectories=True)
            out_dir = os.path.join(OUT_BASE, "fig1_trajectories")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"loc{loc_id}_merge_trajectories.png")
            fig.savefig(path)
            plt.close(fig)
            print(f"  -> {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
