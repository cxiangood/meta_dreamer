"""
论文图表生成脚本。

生成毕设论文所需的数据分析图表：
  1. 各 Location BEV 场景总览 + 汇入轨迹叠加
  2. 汇入数据统计（按 location / recording 的条数柱状图）
  3. 轨迹特征分布（速度、长度、汇入时间）
  4. 奖励分布分析（各组件贡献、return 分布）
  5. 单条汇入轨迹详细可视化（位置 + 速度 + lanelet + reward）

用法:
    python3 mirro_data_map/plot_thesis_figures.py              # 全部图表
    python3 mirro_data_map/plot_thesis_figures.py --fig 2      # 只画图 2
    python3 mirro_data_map/plot_thesis_figures.py --fig 1,3    # 图 1 和 3
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ── 配色常量 (thesis-plot-style skill) ──
C_TRAIN = "#A3C2F4"         # 浅蓝紫
C_TEST = "#F8CA9C"          # 浅橙
C_POSITIVE = "#C5DFB4"      # 嫩草绿
C_NEGATIVE = "#F8CAAC"      # 暖浅橙
C_NEUTRAL = "#F9EBF8"       # 淡粉紫
C_HIGHLIGHT = "#FFE699"     # 亮奶黄
C_VELOCITY = "#DCEFFB"      # 淡青蓝
C_POLICY = "#AE87CA"        # 深薰衣草紫
C_BASELINE = "#D5C48B"      # 暖土黄
C_AXIS = "#576E90"          # 深雾霾蓝
C_ACCEL = "#C5DFB4"         # 嫩草绿
C_DECEL = "#F8CAAC"         # 暖浅橙
C_EMPHASIS = "#F87E0F"      # 强调橙

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
    load_lanelet2_onramp_ids, extract_actions_and_rewards,
    LOC_NAMES, DATA_DIR, MAP_DIR, OUT_DIR,
    W_BASE_COST, W_SPEED_FRAC, W_ACCEL, W_DECEL, W_SPEED_POST,
    W_PROGRESS, W_SUCCESS,
)

DATASET_DIR = os.path.dirname(DATA_DIR)
OUT_FIG = os.path.join(REPO_ROOT, "docs", "thesis_figures")
os.makedirs(OUT_FIG, exist_ok=True)

TRAIN_LOCS = [0, 2, 4, 5, 6]
TEST_LOCS = [1, 3]


# ── 工具函数 ──

def _load_all_stats():
    """加载所有 location 的汇入统计数据。"""
    stats = {}
    for loc_id in range(7):
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

        details = []
        for rid in sorted(rec_ids):
            merges = find_merge_tracks(rid, ramp, main, sn, loc_id=loc_id)
            details.append({"rec_id": rid, "n_merges": len(merges), "tracks": merges})

        total = sum(d["n_merges"] for d in details)
        stats[loc_id] = {
            "name": LOC_NAMES[loc_id],
            "rec_ids": rec_ids,
            "n_recs": len(rec_ids),
            "n_merges": total,
            "split": "train" if loc_id in TRAIN_LOCS else "test",
            "details": details,
            "main_speed": ms,
        }
    return stats


def _compute_reward_stats(max_tracks=50):
    """计算 rec 00 的详细 reward 统计。"""
    loc_id = 0
    rec_id = 0
    sl = pd.read_csv(os.path.join(DATA_DIR, "00_recordingMeta.csv"))["speedLimit"].iloc[0]
    ramp, main, main_speed = classify_lanelets([0], sl)
    csv = pd.read_csv(os.path.join(DATA_DIR, "00_tracks.csv"), low_memory=False)

    net_xml = get_map_file(0)
    sn = sumolib.net.readNet(net_xml, withInternal=False)
    merges = find_merge_tracks(0, ramp, main, sn, loc_id=0)

    records = []
    for m in merges[:max_tracks]:
        tid = m["track_id"]
        ego_df = csv[csv["trackId"] == tid].sort_values("frame")
        if len(ego_df) < 20:
            continue
        actions, rewards, dones, ll_ids, lon_vel, positions = \
            extract_actions_and_rewards(ego_df, ramp, main, main_speed)
        ms = m["merge_frame_idx"]
        T = len(rewards)
        records.append({
            "tid": tid, "T": T, "ret": float(rewards.sum()),
            "v0": float(lon_vel[0]), "v_merge": float(lon_vel[min(ms, T - 1)]),
            "v1": float(lon_vel[-1]),
            "merge_step": ms, "merge_frac": ms / T,
            "rewards": rewards, "lon_vel": lon_vel,
            "pre_mean": float(rewards[:ms].mean()) if ms < T else 0,
            "post_mean": float(rewards[ms:].mean()) if ms < T else 0,
        })
    return records


# ── Figure 1: 场景 BEV 总览 ──

def fig1_scene_overview(stats):
    """7 个 Location 的 BEV 地图 + 匝道汇入轨迹叠加。"""
    print("[Fig 1] Scene BEV overview ...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, loc_id in enumerate(range(7)):
        ax = axes[idx]
        s = stats[loc_id]
        # 加载已有的 BEV 预览图
        preview = os.path.join(OUT_DIR, f"loc{loc_id}_ramp_merge.png")
        if os.path.exists(preview):
            img = cv2.imread(preview)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Loc {loc_id}\n{s['name']}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
        split_label = "Train" if loc_id in TRAIN_LOCS else "Test"
        ax.set_title(f"Loc {loc_id}: {s['name']}\n{s['n_merges']} merges ({split_label})",
                     fontsize=10)
        ax.axis("off")

    # 隐藏多余子图
    axes[-1].axis("off")
    fig.suptitle("exiD Dataset: 7 Locations with On-Ramp Merge Trajectories", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_FIG, "fig1_scene_overview.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── Figure 2: 汇入数据统计 ──

def fig2_merge_statistics(stats):
    """按 location 和 recording 的汇入数量统计。"""
    print("[Fig 2] Merge statistics ...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) 按 location 柱状图
    ax = axes[0]
    loc_ids = list(range(7))
    names = [f"L{loc}\n{stats[loc]['name'][:12]}" for loc in loc_ids]
    counts = [stats[loc]["n_merges"] for loc in loc_ids]
    colors = [C_TRAIN if loc in TRAIN_LOCS else C_TEST for loc in loc_ids]
    bars = ax.bar(range(7), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Number of Merge Trajectories")
    ax.set_title("(a) Merges per Location")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(c), ha="center", va="bottom", fontsize=9)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc=C_TRAIN, label="Train"),
        plt.Rectangle((0, 0), 1, 1, fc=C_TEST, label="Test"),
    ])

    # (b) 每个 recording 的汇入数热力图风格
    ax = axes[1]
    all_counts = []
    all_labels = []
    all_colors = []
    for loc in range(7):
        for d in stats[loc]["details"]:
            all_counts.append(d["n_merges"])
            all_labels.append(f"L{loc}")
            all_colors.append(C_TRAIN if loc in TRAIN_LOCS else C_TEST)

    ax.bar(range(len(all_counts)), all_counts, color=all_colors, width=1.0, linewidth=0)
    ax.set_xlabel("Recording Index")
    ax.set_ylabel("Merges per Recording")
    ax.set_title("(b) Merges per Recording (93 recordings)")

    # 添加 location 分割线
    cum = 0
    for loc in range(7):
        n = len(stats[loc]["details"])
        if loc > 0:
            ax.axvline(cum - 0.5, color="gray", linestyle="--", linewidth=0.5)
        mid = cum + n / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95, f"L{loc}", ha="center", fontsize=7, color="red")
        cum += n

    plt.tight_layout()
    path = os.path.join(OUT_FIG, "fig2_merge_statistics.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── Figure 3: 轨迹特征分布 ──

def fig3_trajectory_distributions(stats, reward_records):
    """轨迹特征分布：初始速度、汇入速度、轨迹长度、汇入时刻。"""
    print("[Fig 3] Trajectory distributions ...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    if not reward_records:
        print("  No reward records, skipping")
        return

    v0 = np.array([r["v0"] for r in reward_records])
    vm = np.array([r["v_merge"] for r in reward_records])
    v1 = np.array([r["v1"] for r in reward_records])
    T = np.array([r["T"] for r in reward_records])
    mf = np.array([r["merge_frac"] for r in reward_records])

    ax = axes[0, 0]
    ax.hist(v0, bins=20, color=C_POSITIVE, alpha=0.7, label="Start (ramp)")
    ax.hist(vm, bins=20, color=C_TRAIN, alpha=0.7, label="At merge")
    ax.hist(v1, bins=20, color=C_EMPHASIS, alpha=0.7, label="End (highway)")
    ax.set_xlabel("Longitudinal Velocity (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Velocity Distribution")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.hist(T / 25.0, bins=25, color=C_POLICY, alpha=0.8)
    ax.set_xlabel("Trajectory Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Trajectory Length Distribution")
    ax.axvline(np.mean(T / 25.0), color="red", linestyle="--", label=f"mean={np.mean(T/25.0):.1f}s")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(mf, bins=25, color=C_VELOCITY, alpha=0.8)
    ax.set_xlabel("Merge Frame Fraction (merge_step / T)")
    ax.set_ylabel("Count")
    ax.set_title("(c) When Does Merge Happen?")
    ax.axvline(np.mean(mf), color="red", linestyle="--", label=f"mean={np.mean(mf):.2f}")
    ax.legend()

    ax = axes[1, 1]
    # 速度变化：v0 -> vm -> v1 箱线图
    bp = ax.boxplot([v0, vm, v1], tick_labels=["Start\n(ramp)", "Merge\npoint", "End\n(highway)"],
                    patch_artist=True)
    colors = [C_POSITIVE, C_TRAIN, C_EMPHASIS]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("(d) Velocity at Key Moments")

    fig.suptitle("Merge Trajectory Characteristics (rec 00, n={})".format(len(reward_records)),
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_FIG, "fig3_trajectory_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── Figure 4: 奖励分析 ──

def fig4_reward_analysis(reward_records):
    """奖励分布分析：return 分布、各阶段均值、单条轨迹 reward 曲线。"""
    print("[Fig 4] Reward analysis ...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    rets = np.array([r["ret"] for r in reward_records])

    # (a) Return 分布
    ax = axes[0]
    ax.hist(rets, bins=20, color=C_TRAIN, alpha=0.8, edgecolor="white")
    ax.axvline(rets.mean(), color="red", linestyle="--", label=f"mean={rets.mean():.1f}")
    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Count")
    ax.set_title("(a) Return Distribution")
    ax.legend()

    # (b) pre vs post merge reward
    ax = axes[1]
    pre = [r["pre_mean"] for r in reward_records]
    post = [r["post_mean"] for r in reward_records]
    x = np.arange(len(reward_records))
    width = 0.35
    ax.bar(x - width / 2, pre, width, label="Pre-merge", color=C_TEST, alpha=0.8)
    ax.bar(x + width / 2, post, width, label="Post-merge", color=C_POSITIVE, alpha=0.8)
    ax.set_xlabel("Trajectory (sorted by return)")
    ax.set_ylabel("Mean Reward per Frame")
    ax.set_title("(b) Pre vs Post Merge Reward")
    ax.legend(fontsize=9)
    ax.set_xticks([])

    # (c) 单条轨迹 reward + velocity 曲线
    ax = axes[2]
    best = max(reward_records, key=lambda r: r["ret"])
    worst = min(reward_records, key=lambda r: r["ret"])
    mid = reward_records[len(reward_records) // 2]

    for label, rec, color in [("best", best, C_POSITIVE), ("median", mid, C_TRAIN),
                               ("worst", worst, C_EMPHASIS)]:
        r = rec["rewards"]
        ms = rec["merge_step"]
        ax.plot(r, color=color, alpha=0.7, linewidth=0.8,
                label=f"{label} (ret={rec['ret']:.0f})")
        ax.axvline(ms, color=color, linestyle=":", alpha=0.4)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Reward")
    ax.set_title("(c) Reward Curves (3 samples)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_FIG, "fig4_reward_analysis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── Figure 5: 单条轨迹详细可视化 ──

def _compute_lateral_offset(positions, merge_step):
    """计算横向偏移：相对主路中心线的垂直距离。

    用 PCA 拟合汇入后轨迹得到主路方向，然后投影所有点到垂直轴。
    """
    post_pos = positions[merge_step:]
    center = post_pos.mean(axis=0)
    cov = np.cov(post_pos[:, 0], post_pos[:, 1])
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = eigvecs[:, -1]  # 最大特征值方向 = 主路方向
    lat_dir = np.array([-main_dir[1], main_dir[0]])  # 垂直方向
    lateral = (positions - center) @ lat_dir
    # 归一化：汇入点 = 0
    return lateral - lateral[merge_step]


def fig5_trajectory_detail(reward_records):
    """单条汇入轨迹的详细可视化：速度、加速度、横向偏移、累计reward。"""
    print("[Fig 5] Trajectory detail ...")

    # 使用 rec39 track39（匝道起步，曲线汇入）
    rec_id = 39
    tid = 39

    csv = pd.read_csv(os.path.join(DATA_DIR, f"{rec_id:02d}_tracks.csv"), low_memory=False)
    ego_df = csv[csv["trackId"] == tid].sort_values("frame")

    sl = pd.read_csv(os.path.join(DATA_DIR, f"{rec_id:02d}_recordingMeta.csv"))["speedLimit"].iloc[0]
    ramp, main, main_speed = classify_lanelets([rec_id], sl)

    actions, rewards, dones, ll_ids, lon_vel, positions = \
        extract_actions_and_rewards(ego_df, ramp, main, main_speed)

    T = len(rewards)
    ts = np.arange(T) / 25.0

    # 找到汇入帧
    ms = None
    for i in range(1, T):
        if ll_ids[i - 1] in ramp and ll_ids[i] in main:
            ms = i
            break
    if ms is None:
        print("  WARNING: no merge detected, using cache")
        ms = 289

    # 加速度
    lon_accel = ego_df["lonAcceleration"].values.astype(np.float32)

    # 横向偏移（相对主路中心线）
    lateral = _compute_lateral_offset(positions, ms)

    # 累计 reward
    cum_reward = np.cumsum(rewards)

    merge_t = ms / 25.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                              gridspec_kw={"hspace": 0.25})
    merge_kw = dict(color="red", linestyle=":", alpha=0.5, linewidth=1)

    # (a) 速度
    ax = axes[0]
    ax.plot(ts[:ms], lon_vel[:ms], color=C_TEST, linewidth=1.5, label="Ramp")
    ax.plot(ts[ms:], lon_vel[ms:], color="#333", linewidth=1.5, label="Post-merge")
    ax.axhline(main_speed, color="gray", linestyle="--", alpha=0.4,
               label=f"Main road ({main_speed:.1f} m/s)")
    ax.axvline(merge_t, **merge_kw)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("(a) Longitudinal Speed", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    # (b) 加速度
    ax = axes[1]
    ax.fill_between(ts[:ms], lon_accel[:ms], 0, alpha=0.3, color=C_ACCEL)
    ax.fill_between(ts[ms:], lon_accel[ms:], 0, alpha=0.3, color=C_POSITIVE)
    ax.plot(ts[:ms], lon_accel[:ms], color=C_TEST, linewidth=1, label="Ramp")
    ax.plot(ts[ms:], lon_accel[ms:], color="#333", linewidth=1, label="Post-merge")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(merge_t, **merge_kw)
    ax.set_ylabel("Accel. (m/s²)")
    ax.set_title("(b) Longitudinal Acceleration", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    # (c) 横向偏移
    ax = axes[2]
    ax.plot(ts[:ms], lateral[:ms], color=C_TEST, linewidth=1.5, label="Ramp")
    ax.plot(ts[ms:], lateral[ms:], color="#333", linewidth=1.5, label="Post-merge")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4,
               label="Main road center")
    ax.axvline(merge_t, **merge_kw)
    ax.scatter(ts[ms], lateral[ms], c="red", s=60, zorder=5, marker="*")
    ax.set_ylabel("Lateral Offset (m)")
    ax.set_title("(c) Lateral Offset from Main Road Centerline", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    # (d) 累计 Reward
    ax = axes[3]
    ax.plot(ts[:ms], cum_reward[:ms], color=C_TEST, linewidth=1.5, label="Ramp")
    ax.plot(ts[ms:], cum_reward[ms:], color="#333", linewidth=1.5, label="Post-merge")
    ax.axvline(merge_t, **merge_kw)
    ax.scatter(ts[ms], cum_reward[ms], c="red", s=60, zorder=5, marker="*")
    ax.scatter(ts[-1], cum_reward[-1], c="blue", s=50, zorder=5, marker="s")
    ax.annotate(f"+{rewards[ms]:.0f} merge bonus",
                xy=(ts[ms], cum_reward[ms]),
                xytext=(ts[ms] + 1, cum_reward[ms] - 5),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax.annotate(f"+{rewards[-1]:.0f} success",
                xy=(ts[-1], cum_reward[-1]),
                xytext=(ts[-1] - 3, cum_reward[-1] - 8),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="blue", lw=0.8))
    ax.set_ylabel("Cumulative Reward")
    ax.set_xlabel("Time (s)")
    ax.set_title("(d) Cumulative Reward", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Merge Trajectory Detail — Track {tid} rec{rec_id:02d} "
        f"(T={T}, merge@{merge_t:.1f}s, return={cum_reward[-1]:.1f})",
        fontsize=13, fontweight="bold",
    )
    path = os.path.join(OUT_FIG, "fig5_trajectory_detail.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── Figure 6: Action 分布 ──

def fig6_action_distribution(reward_records):
    """Steering 和 Throttle 的分布。"""
    print("[Fig 6] Action distribution ...")

    loc_id = 0
    sl = pd.read_csv(os.path.join(DATA_DIR, "00_recordingMeta.csv"))["speedLimit"].iloc[0]
    ramp, main, main_speed = classify_lanelets([0], sl)
    csv = pd.read_csv(os.path.join(DATA_DIR, "00_tracks.csv"), low_memory=False)

    all_steer = []
    all_throttle = []
    for rec in reward_records:
        tid = rec["tid"]
        ego_df = csv[csv["trackId"] == tid].sort_values("frame")
        if len(ego_df) < 20:
            continue
        actions, _, _, _, _, _ = extract_actions_and_rewards(ego_df, ramp, main, main_speed)
        all_steer.append(actions[:, 0])
        all_throttle.append(actions[:, 1])

    steer = np.concatenate(all_steer)
    throttle = np.concatenate(all_throttle)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(steer, bins=50, color=C_POLICY, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Steering (normalized)")
    ax.set_ylabel("Count")
    ax.set_title(f"(a) Steering: mean={steer.mean():.4f}, std={steer.std():.4f}")

    ax = axes[1]
    ax.hist(throttle, bins=50, color=C_VELOCITY, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Throttle (normalized)")
    ax.set_ylabel("Count")
    ax.set_title(f"(b) Throttle: mean={throttle.mean():.4f}, std={throttle.std():.4f}")

    fig.suptitle("Action Distribution (rec 00)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_FIG, "fig6_action_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", default=None, help="Figure numbers, e.g. '1,3' or 'all'")
    args = parser.parse_args()

    if args.fig:
        figs = [int(x.strip()) for x in args.fig.split(",")]
    else:
        figs = [1, 2, 3, 4, 5, 6]

    print(f"Generating figures: {figs}")
    print(f"Output dir: {OUT_FIG}")
    print()

    # Load stats (shared across figures)
    if any(f in figs for f in [1, 2]):
        stats = _load_all_stats()
    else:
        stats = None

    # Load reward stats (shared)
    if any(f in figs for f in [3, 4, 5, 6]):
        reward_records = _compute_reward_stats()
    else:
        reward_records = None

    if 1 in figs and stats:
        fig1_scene_overview(stats)
    if 2 in figs and stats:
        fig2_merge_statistics(stats)
    if 3 in figs and reward_records:
        fig3_trajectory_distributions(stats, reward_records)
    if 4 in figs and reward_records:
        fig4_reward_analysis(reward_records)
    if 5 in figs and reward_records:
        fig5_trajectory_detail(reward_records)
    if 6 in figs and reward_records:
        fig6_action_distribution(reward_records)

    print(f"\nDone! Figures saved to {OUT_FIG}/")


if __name__ == "__main__":
    main()
