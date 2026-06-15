"""
Online DreamerV3 Training on exiD Highway Merge Scenarios.

Closed-loop training: interact with MetaDrive exiD scenarios (loc 0/2/4/5/6),
collect episodes into replay buffer, train WM on mixed offline+online data,
train AC in imagination. Evaluates zero-shot on loc 1/3.

Usage:
    python train_online_dreamer.py \
        --wm-ckpt logs/df_sig_cnn_jepa_traj/checkpoint_step2000.pt \
        --ac-ckpt logs/bc_jepa_traj/checkpoint_bc_best.pt \
        --data-dir /path/to/exiD-dataset-v2.1/data \
        --npz-dir /path/to/exid_dreamer_data \
        --train-locs 0 2 4 5 6 --eval-locs 1 3 \
        --max-traj-per-loc 20
"""

import argparse
import json
import math
import os
import sys
import time
import numpy as np
import torch
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# SUMO setup
if "SUMO_HOME" not in os.environ:
    for candidate in [
        "/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo",
        "/usr/share/sumo",
        "/share/apps/sumo-1.20",           # HPC module load sumo/1.20
    ]:
        if os.path.isdir(candidate):
            os.environ["SUMO_HOME"] = candidate
            break
SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
# HPC: sumolib at $SUMO_HOME/share/sumo/tools, local: $SUMO_HOME/tools
for tools_candidate in [
    os.path.join(SUMO_HOME, "share/sumo/tools"),
    os.path.join(SUMO_HOME, "tools"),
]:
    if os.path.isdir(tools_candidate):
        sys.path.insert(0, tools_candidate)
        break
os.environ.setdefault("METADRIVE_HEADLESS", "1")

import sumolib
from metadrive.type import MetaDriveType as MT
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.scenario_env import ScenarioOnlineEnv

from config import Config
from models import WorldModel, Actor, Critic
from envs.world_model_policy import WorldModelPolicy, setup_wm_policy
from training.replay_buffer import ReplayBuffer
from training.offline_buffer import OfflineDataset
from training.trainer import Trainer

# ── Constants ──────────────────────────────────────────────────────────────
LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}
TRAIN_LOCS = {0, 2, 4, 5, 6}
VAL_LOCS = {1, 3}
BEV_W, BEV_H = 400, 300


# ═══════════════════════════════════════════════════════════════════════════
#  Trajectory Selection by Traffic Density
# ═══════════════════════════════════════════════════════════════════════════

def compute_traffic_density(traj, data_dir, window=50):
    """Compute same-direction traffic density during the merge window.

    For each frame in [merge_idx-window, merge_idx+window], counts vehicles
    on the main road going roughly the same direction as ego.

    Args:
        traj: dict with rid, tid, merge_idx, loc_id
        data_dir: path to exiD data/ directory
        window: half-width of merge window in frames

    Returns:
        float: average number of same-direction vehicles per frame
    """
    import pandas as pd

    rec_id = traj["rid"]
    tid = traj["tid"]
    merge_idx = traj.get("merge_idx", -1)
    if merge_idx < 0:
        return 0.0

    try:
        tracks_csv = pd.read_csv(
            os.path.join(data_dir, f"{rec_id:02d}_tracks.csv"), low_memory=False
        )
    except Exception:
        return 0.0

    ego_sub = tracks_csv[tracks_csv["trackId"] == int(tid)].sort_values("frame")
    if len(ego_sub) == 0:
        return 0.0

    f0 = int(ego_sub["frame"].iloc[0])
    f1 = int(ego_sub["frame"].iloc[-1])

    # Merge window
    mf = f0 + merge_idx
    win_start = max(f0, mf - window)
    win_end = min(f1, mf + window)
    if win_end <= win_start:
        return 0.0

    # Ego heading (average over window)
    ego_win = ego_sub[(ego_sub["frame"] >= win_start) & (ego_sub["frame"] <= win_end)]
    ego_heading = ego_win["heading"].mean()

    # Other vehicles in the same frames
    win_data = tracks_csv[(tracks_csv["frame"] >= win_start) & (tracks_csv["frame"] <= win_end)]
    other_vehicles = win_data[win_data["trackId"] != int(tid)]

    # Count vehicles going the same direction (±45° of ego heading)
    ego_rad = math.radians(ego_heading)
    ego_dir = np.array([math.cos(ego_rad), math.sin(ego_rad)])

    counts_per_frame = []
    for frame in range(win_start, win_end + 1):
        frame_tracks = other_vehicles[other_vehicles["frame"] == frame]
        count = 0
        for _, row in frame_tracks.iterrows():
            h = math.radians(row["heading"])
            vdir = np.array([math.cos(h), math.sin(h)])
            dot = np.dot(ego_dir, vdir)
            if dot > 0.5:  # within ±60°
                count += 1
        counts_per_frame.append(count)

    return float(np.mean(counts_per_frame)) if counts_per_frame else 0.0


def select_trajectories(merge_cache_path, data_dir, train_locs, max_per_loc=20):
    """Select balanced trajectories stratified by traffic density.

    For each training location:
      1. Compute traffic density for all trajectories
      2. Sort and split into 3 equal bins (low / medium / high)
      3. Select max_per_loc // 3 from each bin

    Args:
        merge_cache_path: path to exid_merge_cache.json
        data_dir: path to exiD data/
        train_locs: list of location IDs
        max_per_loc: max trajectories per location

    Returns:
        list of selected trajectory dicts
    """
    with open(merge_cache_path) as f:
        all_items = json.load(f)

    selected = []

    for loc_id in train_locs:
        items = all_items.get(str(loc_id), [])
        if not items:
            print(f"  [loc {loc_id}] No trajectories in cache")
            continue

        print(f"  [loc {loc_id}] Computing traffic density for {len(items)} trajectories...")
        scored = []
        for item in items:
            density = compute_traffic_density(item, data_dir)
            scored.append((density, item))

        scored.sort(key=lambda x: x[0])

        # Split into 3 bins
        n = len(scored)
        per_bin = max(1, (max_per_loc or n) // 3)
        bin_size = n // 3
        low = scored[:bin_size]
        mid = scored[bin_size:2 * bin_size]
        high = scored[2 * bin_size:]

        # Sample from each bin
        rng = np.random.RandomState(42)
        for bin_name, bin_data in [("low", low), ("mid", mid), ("high", high)]:
            if not bin_data:
                continue
            n_select = min(per_bin, len(bin_data))
            indices = rng.choice(len(bin_data), n_select, replace=False)
            densities = [bin_data[i][0] for i in indices]
            for i in indices:
                selected.append(bin_data[i][1])

        densities_all = [s[0] for s in scored]
        print(f"    density range: {densities_all[0]:.1f} - {densities_all[-1]:.1f} "
              f"| selected {max_per_loc} (low≤{low[-1][0]:.1f} "
              f"mid≤{mid[-1][0]:.1f} high≤{high[-1][0]:.1f})")

    print(f"  Total selected: {len(selected)} trajectories")
    return selected


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario Setup (reuses eval_exid_phase4.py patterns)
# ═══════════════════════════════════════════════════════════════════════════

def get_map_file(loc_id):
    map_dir = os.path.join(os.path.dirname(__file__), "../mirro_data_map")
    for name in [f"exid_loc{loc_id}_orig.net.xml", f"exid_loc{loc_id}.net.xml"]:
        path = os.path.join(map_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No SUMO map for location {loc_id} in {map_dir}")


def build_map_features(loc_id):
    net_xml = get_map_file(loc_id)
    raw_net = sumolib.net.readNet(net_xml, withInternal=True)
    xmin, ymin, xmax, ymax = raw_net.getBoundary()
    off_x = -(xmax + xmin) / 2
    off_y = -(ymax + ymin) / 2

    graph = RoadLaneJunctionGraph(net_xml)
    features = extract_map_features(graph)

    # Resample short broken lines
    SPACING = 2.0
    for k, v in features.items():
        if v.get("type") in (MT.LINE_BROKEN_SINGLE_WHITE, MT.LINE_BROKEN_SINGLE_YELLOW):
            pl = v.get("polyline", [])
            if len(pl) < 4:
                pl = np.array(pl, dtype=np.float32)
                d = pl[-1] - pl[0]
                length = float(np.hypot(d[0], d[1]))
                n = max(int(length / SPACING), 4)
                pts = [pl[0]] + [pl[0] + (i / n) * d for i in range(1, n)] + [pl[-1]]
                v["polyline"] = np.array(pts, dtype=np.float32)
    return features, off_x, off_y


def build_scenario_dict(rec_id, track_id, loc_id, map_features, off_x, off_y,
                        data_dir, max_vehicles=100):
    import pandas as pd

    tracks_csv = pd.read_csv(os.path.join(data_dir, f"{rec_id:02d}_tracks.csv"),
                             low_memory=False)
    tracks_meta = pd.read_csv(os.path.join(data_dir, f"{rec_id:02d}_tracksMeta.csv"))

    sdc_id = int(track_id)
    ego_sub = tracks_csv[tracks_csv["trackId"] == sdc_id].sort_values("frame")
    f0 = int(ego_sub["frame"].iloc[0])
    f1 = int(ego_sub["frame"].iloc[-1])
    t_len = f1 - f0 + 1

    window = tracks_csv[(tracks_csv["frame"] >= f0) & (tracks_csv["frame"] <= f1)]
    candidates = [int(x) for x in window["trackId"].unique() if int(x) != sdc_id]
    selected = [sdc_id] + candidates[:max_vehicles - 1]

    meta_by_id = tracks_meta.set_index("trackId")
    ts_arr = np.arange(t_len, dtype=np.float32) / 25.0
    _CLASS_HEIGHT = {"car": 1.5, "van": 1.85, "truck": 2.8, "motorcycle": 1.37}
    tracks = {}

    for tid in selected:
        try:
            mrow = meta_by_id.loc[tid]
            if isinstance(mrow, pd.DataFrame):
                mrow = mrow.iloc[0]
            default_len = float(mrow["length"])
            default_wid = float(mrow["width"])
            default_height = _CLASS_HEIGHT.get(str(mrow.get("class", "car")), 1.5)
        except (KeyError, TypeError):
            default_len, default_wid, default_height = 4.5, 2.0, 1.5

        sub = window[window["trackId"] == tid].sort_values("frame")
        if len(sub) == 0:
            continue

        pos = np.zeros((t_len, 3), dtype=np.float32)
        heading = np.zeros(t_len, dtype=np.float32)
        vel = np.zeros((t_len, 2), dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)

        for _, r in sub.iterrows():
            fi = int(r["frame"]) - f0
            if 0 <= fi < t_len:
                pos[fi, 0] = float(r["xCenter"]) + off_x
                pos[fi, 1] = float(r["yCenter"]) + off_y
                heading[fi] = math.radians(float(r["heading"]))
                vel[fi, 0] = float(r["xVelocity"])
                vel[fi, 1] = float(r["yVelocity"])
                valid[fi] = True

        if not valid.any():
            continue

        tracks[str(tid)] = {
            "type": MT.VEHICLE,
            "state": {
                "position": pos, "velocity": vel, "heading": heading, "valid": valid,
                "length": np.full(t_len, default_len, dtype=np.float32),
                "width": np.full(t_len, default_wid, dtype=np.float32),
                "height": np.full(t_len, default_height, dtype=np.float32),
            },
            "metadata": {"type": MT.VEHICLE, "object_id": str(tid), "dataset": "exiD"},
        }

    return {
        "id": f"exid-{rec_id:02d}-track{sdc_id}",
        "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": MT.COORDINATE_METADRIVE,
            "ts": ts_arr,
            "sdc_id": str(sdc_id),
            "scenario_id": f"exid_{rec_id:02d}_{sdc_id}",
            "dataset": "exiD",
            "source_file": f"recording_{rec_id:02d}",
            "ego_vehicle_class": "car",
            "frame_range": (f0, f1),
            "location_id": loc_id,
        },
        "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": map_features,
    }, t_len


def create_env(scenario, t_len):
    """Create ScenarioOnlineEnv matching Phase 1 data collection config."""
    return ScenarioOnlineEnv(config=dict(
        use_render=False,
        image_observation=True,
        agent_policy=WorldModelPolicy,
        horizon=t_len + 200,
        store_map=False,
        set_static=True,
        camera_smooth=False,
        vehicle_config=dict(
            no_wheel_friction=True,
            show_navi_mark=False,
            image_source="rgb_camera",
        ),
        sensors={"rgb_camera": (RGBCamera, BEV_W, BEV_H)},
        norm_pixel=False,
        height_scale=0.01,
    ))


# ═══════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_models_for_online(wm_ckpt_path, ac_ckpt_path, device):
    """Load WM + Actor + Critic and optimizers for online training.

    WM is set to train mode (requires_grad=True).
    Actor and Critic are also trainable.
    """
    print(f"Loading world model from: {wm_ckpt_path}")
    ckpt = torch.load(wm_ckpt_path, map_location=device)

    # Reconstruct config from checkpoint
    saved_cfg = ckpt.get("config", {})
    cfg = Config()
    for k, v in saved_cfg.items():
        if hasattr(cfg, k) and k not in ("train_phase", "logdir", "total_steps"):
            setattr(cfg, k, v)

    # Override for online training: shorter sequences (no JEPA needed),
    # disable auxiliary heads (no labels during online collection)
    cfg.batch_length = 50
    cfg.batch_size = 16
    cfg.use_jepa = False
    cfg.use_speed_head = False
    cfg.use_traj_head = False
    cfg.use_phase_head = False
    cfg.rssm_phase_conditional = False
    cfg.rssm_moe = False

    print(f"  Config: reg={cfg.regularization} decoder={cfg.use_decoder} "
          f"bev={cfg.bev_size} downsample={cfg.bev_downsample} "
          f"batch={cfg.batch_size} batch_len={cfg.batch_length}"
          f" {'(online: JEPA/Speed/Traj/Phase DISABLED)'}")

    # Create models
    wm = WorldModel(cfg).to(device)
    wm.load_state_dict(ckpt["world_model"])
    wm.train()
    for p in wm.parameters():
        p.requires_grad = True

    feat_dim = wm.feature_dim()
    actor = Actor(feat_dim, action_dim=cfg.action_dim,
                  hidden_dim=cfg.actor_hidden, layers=cfg.actor_layers).to(device)
    critic = Critic(feat_dim, cfg.critic_hidden, cfg.critic_layers).to(device)
    slow_critic = Critic(feat_dim, cfg.critic_hidden, cfg.critic_layers).to(device)

    # Load actor
    actor_loaded = False
    if ac_ckpt_path and os.path.exists(ac_ckpt_path):
        print(f"Loading actor from: {ac_ckpt_path}")
        ac_ckpt = torch.load(ac_ckpt_path, map_location=device)
        if "actor" in ac_ckpt:
            actor.load_state_dict(ac_ckpt["actor"])
            actor_loaded = True
            print(f"  BC step: {ac_ckpt.get('global_step', '?')}")
            if "critic" in ac_ckpt:
                critic.load_state_dict(ac_ckpt["critic"])
    elif "actor" in ckpt:
        print("Loading actor from WM checkpoint")
        actor.load_state_dict(ckpt["actor"])
        actor_loaded = True
    if not actor_loaded:
        print("WARNING: No actor checkpoint found, using random weights!")

    # Load or init critic
    if "critic" in ckpt:
        critic.load_state_dict(ckpt["critic"])

    actor.train()
    critic.train()
    slow_critic.load_state_dict(critic.state_dict())
    for p in slow_critic.parameters():
        p.requires_grad = False

    # Optimizers
    wm_optimizer = torch.optim.Adam(wm.parameters(), lr=cfg.world_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    # Restore optimizer states if available
    if "wm_optimizer" in ckpt:
        wm_optimizer.load_state_dict(ckpt["wm_optimizer"])
    if "actor_optimizer" in ckpt:
        actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    if "critic_optimizer" in ckpt:
        critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

    param_count = sum(p.numel() for p in wm.parameters())
    print(f"  WM params: {param_count / 1e6:.1f}M | "
          f"Actor: {sum(p.numel() for p in actor.parameters()) / 1e3:.0f}K | "
          f"Critic: {sum(p.numel() for p in critic.parameters()) / 1e3:.0f}K")
    return wm, actor, critic, slow_critic, wm_optimizer, actor_optimizer, critic_optimizer, cfg


# ═══════════════════════════════════════════════════════════════════════════
#  Episode Collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_online_episode(env, t_len, wm, actor, device, buffer,
                           bev_size=300, max_extra_steps=150):
    """Collect one closed-loop episode and add transitions to replay buffer.

    WorldModelPolicy.act() is called internally by MetaDrive agent_manager on
    each env.step(). We capture BEV BEFORE step and the REAL action AFTER step.

    Transition stored: (obs[t], action[t], reward[t], done[t])
    """
    obs_dict, info = env.reset()

    # Fix ego z-height
    ego = env.agent
    if ego is not None:
        p = ego.position
        ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)

    total_reward = 0.0
    max_steps = t_len + max_extra_steps

    for i in range(max_steps):
        # Capture BEV BEFORE step (same as what WorldModelPolicy will see)
        bev = _capture_bev_from_env(env, bev_size)
        if bev is None:
            bev = np.zeros((3, bev_size, bev_size), dtype=np.uint8)

        # env.step() calls WorldModelPolicy.act() internally
        obs_dict, reward, done, truncated, info = env.step([0.0, 0.0])

        total_reward += float(reward) if reward else 0.0

        # Get the REAL action that WorldModelPolicy produced
        action = np.array([0.0, 0.0], dtype=np.float32)
        try:
            if hasattr(ego, 'policy') and ego.policy is not None:
                action = ego.policy._prev_action.copy()
        except Exception:
            pass

        # Store transition
        buffer.add(
            bev.astype(np.uint8),
            action,
            float(reward) if reward else 0.0,
            done
        )

        # Check termination
        if info.get("crash_vehicle", False) or info.get("crash", False):
            break
        if info.get("arrive_destination", False):
            break
        if done or truncated:
            break

    return {"reward": total_reward, "steps": i + 1}


def _capture_bev_from_env(env, bev_size=300):
    """Capture top-down BEV from the env's RGBCamera sensor.

    Replicates WorldModelPolicy._capture_bev() logic for offline buffer storage.
    """
    try:
        engine = env.engine
        rgb_cam = engine.sensors.get("rgb_camera")
        if rgb_cam is None:
            return None

        ego = env.agent
        if ego is None:
            return None

        ego_pos = ego.position
        ego_hpr = ego.origin.getHpr()
        heading = ego_hpr.getX()
        engine_origin = engine.origin

        bev_img = rgb_cam.perceive(
            to_float=False,
            new_parent_node=engine_origin,
            position=(float(ego_pos[0]), float(ego_pos[1]), 50.0),
            hpr=(heading, -89, 0),
        )

        if bev_img is None or bev_img.ndim != 3:
            return None

        # Center-crop 400x300 → square, then resize to bev_size
        H, W = bev_img.shape[:2]
        size = min(H, W)
        dh = (H - size) // 2
        dw = (W - size) // 2
        bev_img = bev_img[dh:dh + size, dw:dw + size]

        if size != bev_size:
            from PIL import Image
            bev_img = np.array(
                Image.fromarray(bev_img).resize(
                    (bev_size, bev_size), Image.BILINEAR
                )
            )

        # (H, W, C) uint8 → (C, H, W) uint8
        return bev_img.transpose(2, 0, 1)

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Training Helpers
# ═══════════════════════════════════════════════════════════════════════════

def train_wm_mixed(wm, offline_dataset, buffer, wm_optimizer, cfg, device):
    """Train WM on mixed offline + online batch. Returns metrics dict."""
    half = max(1, cfg.batch_size // 2)

    # Offline batch (torch tensors, float [0,1])
    with torch.no_grad():
        off_result = offline_dataset.sample(half)
        off_obs = off_result[0].permute(1, 0, 2, 3, 4).to(device)
        off_actions = off_result[2].permute(1, 0, 2).to(device)
        off_rewards = off_result[3].permute(1, 0).to(device)
        off_continues = off_result[4].permute(1, 0).to(device)

    # Online batch (numpy uint8 [0,255])
    on_obs_np, on_actions_np, on_rewards_np, on_continues_np = buffer.sample(
        half, cfg.batch_length
    )
    on_obs = torch.FloatTensor(on_obs_np).to(device).permute(1, 0, 2, 3, 4) / 255.0
    on_actions = torch.FloatTensor(on_actions_np).to(device).permute(1, 0, 2)
    on_rewards = torch.FloatTensor(on_rewards_np).to(device).permute(1, 0)
    on_continues = torch.FloatTensor(on_continues_np).to(device).permute(1, 0)

    # Concatenate
    obs = torch.cat([off_obs, on_obs], dim=1)
    actions = torch.cat([off_actions, on_actions], dim=1)
    rewards = torch.cat([off_rewards, on_rewards], dim=1)
    continues = torch.cat([off_continues, on_continues], dim=1)

    prev_state = wm.get_initial_state(obs.shape[1], device)

    wm_optimizer.zero_grad()
    loss, metrics = wm.compute_world_loss(obs, actions, rewards, continues, prev_state)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
    wm_optimizer.step()

    return metrics


def train_ac_imagination(wm, actor, critic, slow_critic,
                         actor_optimizer, critic_optimizer,
                         buffer, cfg, device):
    """Train actor-critic in imagination from ReplayBuffer start states."""
    # Sample start observations
    obs_np, _, _, _ = buffer.sample(cfg.batch_size, 1)
    obs = torch.FloatTensor(obs_np).to(device).squeeze(1) / 255.0  # (B, C, H, W)

    with torch.no_grad():
        embed = wm.encode(obs)
        prev_state = wm.get_initial_state(cfg.batch_size, device)
        dummy_action = torch.zeros(cfg.batch_size, cfg.action_dim, device=device)
        states, _ = wm.rssm.observe(
            embed.unsqueeze(0), dummy_action.unsqueeze(0), prev_state
        )
        start_state = states[-1]

    # Imagine rollout
    imag_rewards = []
    imag_continues = []
    imag_log_probs = []
    state = start_state
    for t in range(cfg.imagination_horizon):
        feature = wm.get_feature(state)
        action, log_prob = actor(feature.detach())
        with torch.no_grad():
            reward = wm.reward_head(feature).squeeze()
            cont = torch.sigmoid(wm.continue_head(feature).squeeze())
        with torch.no_grad():
            states_imag = wm.rssm.imagine(action.unsqueeze(0), state)
            state = states_imag[0]
        imag_rewards.append(reward)
        imag_continues.append(cont)
        imag_log_probs.append(log_prob)

    imag_rewards = torch.stack(imag_rewards)
    imag_continues = torch.stack(imag_continues)
    imag_log_probs = torch.stack(imag_log_probs)

    # Features for value estimation
    with torch.no_grad():
        imag_features = []
        state = start_state
        for t in range(cfg.imagination_horizon):
            feature = wm.get_feature(state)
            action, _ = actor(feature)
            imag_features.append(feature)
            states_imag = wm.rssm.imagine(action.unsqueeze(0), state)
            state = states_imag[0]
        imag_features = torch.stack(imag_features)

    # Lambda returns
    with torch.no_grad():
        slow_values, _ = slow_critic(
            imag_features.reshape(-1, imag_features.shape[-1])
        )
        slow_values = slow_values.reshape(imag_features.shape[:2])

    returns = compute_lambda_returns(
        imag_rewards, imag_continues, slow_values, cfg.gamma, cfg.lambda_gae
    )

    # Critic loss
    values, value_logits = critic(
        imag_features.detach().reshape(-1, imag_features.shape[-1])
    )
    values = values.reshape(imag_features.shape[:2])
    value_logits = value_logits.reshape(-1, 255)
    value_targets = critic.compute_target(returns.detach().reshape(-1))
    critic_loss = torch.nn.functional.cross_entropy(value_logits, value_targets)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
    critic_optimizer.step()

    # Actor loss
    advantages = (returns - values.detach()).reshape(imag_log_probs.shape)
    actor_loss = -(imag_log_probs * advantages).mean()
    actor_loss -= cfg.entropy_weight * (-imag_log_probs).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
    actor_optimizer.step()

    # Update slow critic (EMA)
    for p_slow, p_fast in zip(slow_critic.parameters(), critic.parameters()):
        p_slow.data.lerp_(p_fast.data, 0.005)

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "imag_reward_mean": imag_rewards.mean().item(),
        "value_mean": values.mean().item(),
    }


def compute_lambda_returns(rewards, continues, values, gamma=0.997, lam=0.95):
    H, B = rewards.shape
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1] + gamma * continues[-1] * values[-1]
    for t in reversed(range(H - 1)):
        next_val = rewards[t] + gamma * continues[t] * values[t + 1]
        returns[t] = next_val + lam * continues[t] * (returns[t + 1] - values[t + 1])
    return returns


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run_eval_episode(env, t_len, max_extra_steps=150):
    """Run one eval episode with WorldModelPolicy. Returns result dict."""
    obs, info = env.reset()
    ego = env.agent
    start_pos = None
    if ego is not None:
        p = ego.position
        ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)
        start_pos = (float(p[0]), float(p[1]))

    total_reward = 0.0
    end_reason = "timeout"
    collision = False
    crash_type = ""
    max_steps = t_len + max_extra_steps

    for i in range(max_steps):
        obs, reward, done, truncated, info = env.step([0.0, 0.0])
        total_reward += float(reward) if reward else 0.0

        if ego is not None and start_pos is not None:
            dx = float(ego.position[0]) - start_pos[0]
            dy = float(ego.position[1]) - start_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > 500:
                end_reason = "out_of_bounds"
                collision = True
                crash_type = "out_of_bounds"
                break

        if info.get("crash_vehicle", False):
            collision = True
            crash_type = "vehicle"
            end_reason = "crash_vehicle"
            break
        if info.get("crash", False):
            collision = True
            crash_type = info.get("crash_type", "unknown")
            end_reason = "crash"
            break
        if info.get("arrive_destination", False):
            end_reason = "arrive_destination"
            break
        if done:
            end_reason = "env_done"
            break
        if truncated:
            end_reason = "truncated"
            break

    return {
        "reward": total_reward, "steps": i + 1,
        "collision": collision, "crash_type": crash_type,
        "survived": not collision, "end_reason": end_reason,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Online DreamerV3 training on exiD merge scenarios")
    parser.add_argument("--wm-ckpt", required=True,
                        help="Path to Phase 2 WM checkpoint (.pt)")
    parser.add_argument("--ac-ckpt", default=None,
                        help="Path to Phase 3 BC checkpoint (.pt)")
    parser.add_argument("--data-dir", required=True,
                        help="Path to exiD data/ directory")
    parser.add_argument("--npz-dir", required=True,
                        help="Path to offline npz data (exid_dreamer_data/)")
    parser.add_argument("--train-locs", type=int, nargs="+", default=[0, 2, 4, 5, 6])
    parser.add_argument("--eval-locs", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--max-traj-per-loc", type=int, default=20,
                        help="Max online trajectories per location (default: 20)")
    parser.add_argument("--total-episodes", type=int, default=500,
                        help="Total online episodes to collect (default: 500)")
    parser.add_argument("--train-steps-per-collect", type=int, default=50,
                        help="WM+AC training steps per collection round")
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Episodes between eval runs")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Max eval episodes per location")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logdir", default="./logs/online_dreamer")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--wandb-run-name", default="online_dreamer")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 episode without training")
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    wm, actor, critic, slow_critic, wm_optimizer, actor_optimizer, critic_optimizer, cfg = \
        load_models_for_online(args.wm_ckpt, args.ac_ckpt, device)

    # Setup WorldModelPolicy (must be done before creating env)
    setup_wm_policy(wm, actor, bev_size=cfg.bev_size, device=device)

    # Load offline dataset for mixed training
    print(f"\nLoading offline dataset from: {args.npz_dir}")
    offline_dataset = OfflineDataset(
        args.npz_dir, bev_size=cfg.bev_size,
        seq_len=cfg.batch_length, device=str(device),
        cache_size=cfg.offline_cache_size,
        preload=False,  # Use mmap for memory efficiency
        skip_resize=True,  # Data already at target resolution
    )

    # Replay buffer for online data
    obs_shape = (cfg.input_channels, cfg.bev_size, cfg.bev_size)
    buffer = ReplayBuffer(cfg.online_buffer_capacity, obs_shape, cfg.action_dim)
    print(f"ReplayBuffer: capacity={cfg.online_buffer_capacity} "
          f"obs_shape={obs_shape}")

    # Select trajectories
    merge_cache_path = os.path.join(
        os.path.dirname(__file__), "../mirro_data_map/exid_merge_cache.json"
    )
    print(f"\nSelecting trajectories from: {merge_cache_path}")
    selected_trajs = select_trajectories(
        merge_cache_path, args.data_dir, args.train_locs, args.max_traj_per_loc
    )
    if not selected_trajs:
        print("ERROR: No trajectories selected!")
        sys.exit(1)

    # Group by location
    traj_by_loc = defaultdict(list)
    for traj in selected_trajs:
        traj_by_loc[traj["loc_id"]].append(traj)

    # Pre-build map features for all training locations
    print("\nBuilding map features...")
    map_cache = {}
    for loc_id in sorted(traj_by_loc.keys()):
        features, off_x, off_y = build_map_features(loc_id)
        map_cache[loc_id] = (features, off_x, off_y)
        print(f"  loc {loc_id} ({LOC_NAMES[loc_id]}): {len(features)} features")

    # ── Dry run mode ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n=== DRY RUN: Testing 1 episode ===")
        traj = selected_trajs[0]
        loc_id = traj["loc_id"]
        rec_id = traj["rid"]
        tid = traj["tid"]
        print(f"  loc={loc_id} rec={rec_id} track={tid} merge_idx={traj['merge_idx']}")

        features, off_x, off_y = map_cache[loc_id]
        scenario_dict, t_len = build_scenario_dict(
            rec_id, tid, loc_id, features, off_x, off_y, args.data_dir)
        scenario = ScenarioDescription(scenario_dict)
        env = create_env(scenario, t_len)
        env.set_scenario(scenario)

        result = collect_online_episode(env, t_len, wm, actor, device, buffer,
                                        bev_size=cfg.bev_size)
        print(f"  Reward={result['reward']:.1f} Steps={result['steps']} "
              f"Buffer size={len(buffer)}")
        env.close()
        print("Dry run passed!")
        sys.exit(0)

    # ── Training loop ─────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Online DreamerV3 Training")
    print(f"  Train locations: {args.train_locs}")
    print(f"  Eval locations: {args.eval_locs}")
    print(f"  Trajectories: {len(selected_trajs)} ({args.max_traj_per_loc}/loc)")
    print(f"  Total episodes: {args.total_episodes}")
    print(f"  Train steps/collect: {args.train_steps_per_collect}")
    print(f"  Batch: {cfg.batch_size} x {cfg.batch_length}")
    print(f"  WM: {args.wm_ckpt}")
    print(f"  AC: {args.ac_ckpt or '(from WM ckpt)'}")
    print(f"{'='*60}")

    # WandB setup
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project="online-dreamer",
            name=args.wandb_run_name,
            config=cfg.to_dict(),
            dir=logdir,
        )
    except Exception:
        pass

    env = None
    total_episodes = 0
    total_train_steps = 0
    best_eval_survival = 0.0
    episode_idx = 0
    t_start = time.time()

    # Flatten trajectory list and shuffle
    flat_trajs = selected_trajs.copy()

    while total_episodes < args.total_episodes:
        np.random.shuffle(flat_trajs)

        for traj in flat_trajs:
            if total_episodes >= args.total_episodes:
                break

            loc_id = traj["loc_id"]
            rec_id = traj["rid"]
            tid = traj["tid"]
            merge_idx = traj.get("merge_idx", -1)

            try:
                # Build scenario
                features, off_x, off_y = map_cache[loc_id]
                scenario_dict, t_len = build_scenario_dict(
                    rec_id, tid, loc_id, features, off_x, off_y, args.data_dir)
                scenario = ScenarioDescription(scenario_dict)

                # Fresh env per scenario
                if env is not None:
                    env.close()
                env = create_env(scenario, t_len)
                env.set_scenario(scenario)

                # Collect episode
                result = collect_online_episode(
                    env, t_len, wm, actor, device, buffer, bev_size=cfg.bev_size
                )
                total_episodes += 1
                episode_idx += 1

                # Train after collecting enough data
                if len(buffer) >= cfg.batch_length * 2:
                    wm_metrics_list = []
                    ac_metrics_list = []
                    for _ in range(args.train_steps_per_collect):
                        wm_m = train_wm_mixed(
                            wm, offline_dataset, buffer, wm_optimizer, cfg, device
                        )
                        ac_m = train_ac_imagination(
                            wm, actor, critic, slow_critic,
                            actor_optimizer, critic_optimizer,
                            buffer, cfg, device
                        )
                        wm_metrics_list.append(wm_m)
                        ac_metrics_list.append(ac_m)
                        total_train_steps += 1

                    # Average metrics
                    reg_key = f"loss_{cfg.regularization}"
                    reg_val = np.mean([m.get(reg_key, m.get("loss_dyn", 0))
                                       for m in wm_metrics_list])
                    if cfg.use_decoder:
                        rec_label, rec_val = "rec", np.mean(
                            [m.get("loss_rec", 0) for m in wm_metrics_list])
                    else:
                        rec_label, rec_val = "barlow", np.mean(
                            [m.get("loss_barlow", m.get("loss_rec", 0))
                             for m in wm_metrics_list])
                    actor_loss = np.mean([m["actor_loss"] for m in ac_metrics_list])

                    print(f"[ep {total_episodes}] R={result['reward']:.1f} "
                          f"steps={result['steps']} "
                          f"| buf={len(buffer)} "
                          f"| {rec_label}={rec_val:.4f} "
                          f"dyn={reg_val:.4f} "
                          f"| actor={actor_loss:.4f}")

                    if wandb_run is not None:
                        wandb_log = {
                            "episode": total_episodes,
                            "train_steps": total_train_steps,
                            "ep_reward": result["reward"],
                            "ep_steps": result["steps"],
                            "buffer_size": len(buffer),
                            "wm/loss_barlow": rec_val if not cfg.use_decoder else 0,
                            f"wm/{reg_key}": reg_val,
                            "ac/actor_loss": actor_loss,
                        }
                        wandb_run.log(wandb_log, step=total_episodes)
                else:
                    print(f"[ep {total_episodes}] R={result['reward']:.1f} "
                          f"steps={result['steps']} | buf={len(buffer)} "
                          f"(building buffer...)")

            except Exception as e:
                print(f"[ep {total_episodes}] ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

            # ── Evaluation ─────────────────────────────────────────────────
            if episode_idx % args.eval_interval == 0:
                print(f"\n--- Eval at episode {total_episodes} ---")
                wm.eval()
                actor.eval()

                eval_results = []
                for eval_loc in args.eval_locs:
                    eval_features, eval_off_x, eval_off_y = build_map_features(eval_loc)
                    # Get eval trajectories from cache
                    eval_trajs = select_trajectories(
                        merge_cache_path, args.data_dir, [eval_loc],
                        max_per_loc=args.eval_episodes
                    )
                    for et in eval_trajs[:args.eval_episodes]:
                        try:
                            es_dict, et_len = build_scenario_dict(
                                et["rid"], et["tid"], eval_loc,
                                eval_features, eval_off_x, eval_off_y, args.data_dir
                            )
                            e_scenario = ScenarioDescription(es_dict)
                            if env is not None:
                                env.close()
                            env = create_env(e_scenario, et_len)
                            env.set_scenario(e_scenario)
                            er = run_eval_episode(env, et_len)
                            er["loc_id"] = eval_loc
                            eval_results.append(er)
                        except Exception:
                            continue

                if eval_results:
                    n = len(eval_results)
                    survived = sum(1 for r in eval_results if r["survived"])
                    avg_r = np.mean([r["reward"] for r in eval_results])
                    print(f"  Eval ({n} episodes): survival={survived}/{n} "
                          f"({survived / n * 100:.1f}%) avgR={avg_r:.1f}")

                    if survived / n > best_eval_survival:
                        best_eval_survival = survived / n
                        save_path = os.path.join(logdir, "checkpoint_best.pt")
                        torch.save({
                            "world_model": wm.state_dict(),
                            "actor": actor.state_dict(),
                            "critic": critic.state_dict(),
                            "wm_optimizer": wm_optimizer.state_dict(),
                            "actor_optimizer": actor_optimizer.state_dict(),
                            "critic_optimizer": critic_optimizer.state_dict(),
                            "total_episodes": total_episodes,
                            "total_train_steps": total_train_steps,
                            "eval_survival": best_eval_survival,
                            "config": cfg.to_dict(),
                        }, save_path)
                        print(f"  Saved best checkpoint (survival={best_eval_survival:.3f})")

                    if wandb_run is not None:
                        wandb_run.log({
                            "eval/survival": survived / n,
                            "eval/avg_reward": avg_r,
                        }, step=total_episodes)

                wm.train()
                actor.train()

            # ── Save checkpoint ────────────────────────────────────────────
            if episode_idx % args.save_every == 0:
                save_path = os.path.join(logdir, f"checkpoint_ep{total_episodes}.pt")
                torch.save({
                    "world_model": wm.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "wm_optimizer": wm_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "total_episodes": total_episodes,
                    "total_train_steps": total_train_steps,
                    "config": cfg.to_dict(),
                }, save_path)
                print(f"  Saved checkpoint: ep{total_episodes}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    if env is not None:
        env.close()

    # Final save
    final_path = os.path.join(logdir, "checkpoint_final.pt")
    torch.save({
        "world_model": wm.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "total_episodes": total_episodes,
        "total_train_steps": total_train_steps,
        "config": cfg.to_dict(),
    }, final_path)

    elapsed = time.time() - t_start
    print(f"\nDone! {total_episodes} episodes in {elapsed:.0f}s "
          f"({elapsed / total_episodes:.1f}s/ep) "
          f"{total_train_steps} train steps")
    print(f"Best eval survival: {best_eval_survival:.3f}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
