"""
P4 Online DAPO: Fine-tune actor in real MetaDrive env with group-based PG.

For each scenario, run K episodes with stochastic actor → group advantage →
policy gradient update. Custom reward penalizes collision, no critic needed.

Usage:
    python train_p4_online.py \
        --wm-ckpt logs/df_sigreg_p2/checkpoint_latest.pt \
        --ac-ckpt logs/df_sig_p3/checkpoint_latest.pt \
        --data-dir /path/to/exiD-dataset-v2.1/data \
        --loc 1 3 --max-episodes 100 --k 8 --lr 1e-5
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

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
    ]:
        if os.path.isdir(candidate):
            os.environ["SUMO_HOME"] = candidate
            break
SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))
os.environ.setdefault("METADRIVE_HEADLESS", "1")

import sumolib
from metadrive.policy.base_policy import BasePolicy
from metadrive.type import MetaDriveType as MT
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.scenario_env import ScenarioOnlineEnv

from config import Config
from models import WorldModel, Actor


# ── External action policy ─────────────────────────────────────────────────
# MetaDrive calls policy.act() internally to get the vehicle action.
# We store the action globally and have the policy return it.

_GLOBAL_ACTION = np.array([0.0, 0.0], dtype=np.float32)


class IdlePolicy(BasePolicy):
    """Policy that returns the last externally-set action.

    Set _GLOBAL_ACTION before each env.step() call.
    """
    @classmethod
    def get_input_space(cls):
        import gymnasium as gym
        return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    def act(self, agent_id=None):
        return _GLOBAL_ACTION.copy()


# ── Constants ──────────────────────────────────────────────────────────────

LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}
VAL_LOCS = {1, 3}
BEV_W, BEV_H = 400, 300


# ── Map / scenario / cache ─────────────────────────────────────────────────

def get_map_file(loc_id):
    map_dir = os.path.join(SCRIPT_DIR, "../mirro_data_map")
    for name in [f"exid_loc{loc_id}_orig.net.xml", f"exid_loc{loc_id}.net.xml"]:
        path = os.path.join(map_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No SUMO map for location {loc_id}")


def build_map_features(loc_id):
    net_xml = get_map_file(loc_id)
    raw_net = sumolib.net.readNet(net_xml, withInternal=True)
    xmin, ymin, xmax, ymax = raw_net.getBoundary()
    off_x = -(xmax + xmin) / 2
    off_y = -(ymax + ymin) / 2
    graph = RoadLaneJunctionGraph(net_xml)
    features = extract_map_features(graph)
    SPACING = 2.0
    for v in features.values():
        if v.get("type") in (MT.LINE_BROKEN_SINGLE_WHITE, MT.LINE_BROKEN_SINGLE_YELLOW):
            pl = v.get("polyline", [])
            if len(pl) < 4:
                pl = np.array(pl, dtype=np.float32)
                d = pl[-1] - pl[0]
                length = float(np.hypot(d[0], d[1]))
                n = max(int(length / SPACING), 4)
                v["polyline"] = np.array(
                    [pl[0]] + [pl[0] + (i / n) * d for i in range(1, n)] + [pl[-1]],
                    dtype=np.float32,
                )
    return features, off_x, off_y


def load_merge_cache(loc_ids):
    meta_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    for name in ["exid_merge_cache_selected.json", "exid_merge_cache.json"]:
        cache_path = os.path.join(meta_root, "mirro_data_map", name)
        if os.path.exists(cache_path):
            break
    else:
        raise FileNotFoundError("No merge cache found")
    with open(cache_path) as f:
        all_items = json.load(f)
    trajectories = []
    for loc_id in loc_ids:
        for item in all_items.get(str(loc_id), []):
            trajectories.append(item)
    return trajectories


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
            default_len, default_wid = float(mrow["length"]), float(mrow["width"])
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
            "metadrive_processed": True, "coordinate": MT.COORDINATE_METADRIVE,
            "ts": ts_arr, "sdc_id": str(sdc_id),
            "scenario_id": f"exid_{rec_id:02d}_{sdc_id}",
            "dataset": "exiD", "source_file": f"recording_{rec_id:02d}",
            "ego_vehicle_class": "car",
            "frame_range": (f0, f1), "location_id": loc_id,
        },
        "tracks": tracks, "dynamic_map_states": {},
        "map_features": map_features,
    }, t_len


# ── Env ────────────────────────────────────────────────────────────────────

def _create_env(scenario, t_len):
    return ScenarioOnlineEnv(config=dict(
        use_render=False, image_observation=True,
        agent_policy=IdlePolicy,
        horizon=t_len + 200, store_map=False, set_static=True,
        camera_smooth=False,
        vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False,
                            image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, BEV_W, BEV_H)},
        norm_pixel=False, height_scale=0.01,
    ))


# ── BEV capture ────────────────────────────────────────────────────────────

def capture_bev(env, bev_size=64):
    try:
        engine = env.engine
        rgb_cam = engine.sensors.get("rgb_camera")
        if rgb_cam is None or env.agent is None:
            return np.zeros((3, bev_size, bev_size), dtype=np.float32)
        ego = env.agent
        ego_pos = ego.position
        ego_hpr = ego.origin.getHpr()
        engine_origin = engine.origin
        bev_img = rgb_cam.perceive(
            to_float=False,
            new_parent_node=engine_origin,
            position=(float(ego_pos[0]), float(ego_pos[1]), 50.0),
            hpr=(ego_hpr.getX(), -89, 0),
        )
        if bev_img is None or bev_img.ndim != 3:
            return np.zeros((3, bev_size, bev_size), dtype=np.float32)
        H, W = bev_img.shape[:2]
        size = min(H, W)
        dh, dw = (H - size) // 2, (W - size) // 2
        bev_img = bev_img[dh:dh + size, dw:dw + size]
        if size != bev_size:
            from PIL import Image
            bev_img = np.array(Image.fromarray(bev_img).resize(
                (bev_size, bev_size), Image.BILINEAR))
        return bev_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    except Exception:
        return np.zeros((3, bev_size, bev_size), dtype=np.float32)


# ── Video recording ────────────────────────────────────────────────────────

def capture_bev_video(env, video_size=300):
    """Capture full-res BEV for video recording (square, no normalize)."""
    try:
        engine = env.engine
        rgb_cam = engine.sensors.get("rgb_camera")
        if rgb_cam is None or env.agent is None:
            return None
        ego = env.agent
        ego_pos = ego.position
        ego_hpr = ego.origin.getHpr()
        engine_origin = engine.origin
        bev_img = rgb_cam.perceive(
            to_float=False,
            new_parent_node=engine_origin,
            position=(float(ego_pos[0]), float(ego_pos[1]), 50.0),
            hpr=(ego_hpr.getX(), -89, 0),
        )
        if bev_img is None or bev_img.ndim != 3:
            return None
        H, W = bev_img.shape[:2]
        size = min(H, W)
        dh, dw = (H - size) // 2, (W - size) // 2
        bev_img = bev_img[dh:dh + size, dw:dw + size]
        if size != video_size:
            from PIL import Image
            bev_img = np.array(Image.fromarray(bev_img).resize(
                (video_size, video_size), Image.BILINEAR))
        return bev_img  # (H, W, 3) uint8 RGB
    except Exception:
        return None


def _find_ffmpeg():
    """Find ffmpeg binary, checking conda base + env first."""
    import shutil
    # Check CONDA_PREFIX env (e.g. miniforge3/envs/metadrive)
    for env_var in ["CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "CONDA_ROOT"]:
        prefix = os.environ.get(env_var, "")
        if prefix:
            for sub in ["bin/ffmpeg", "../bin/ffmpeg"]:
                path = os.path.join(prefix, sub)
                if os.path.isfile(path):
                    return path
    # Check hardcoded conda paths
    for base in [
        os.path.expanduser("~/miniforge3/bin/ffmpeg"),
        os.path.expanduser("~/miniconda3/bin/ffmpeg"),
        "/share/home/u23516/miniforge3/bin/ffmpeg",
    ]:
        if os.path.isfile(base):
            return base
    found = shutil.which("ffmpeg")
    if found:
        return found
    return "ffmpeg"


def save_video_ffmpeg(frames, output_path, fps=10):
    """Encode frames (list of HxWx3 uint8 RGB arrays) to mp4 via ffmpeg pipe."""
    if not frames:
        return
    H, W = frames[0].shape[:2]
    ffmpeg_bin = _find_ffmpeg()
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast", "-crf", "28",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


# ── Online DAPO training ───────────────────────────────────────────────────

def compute_custom_reward(env_reward, info, collision_weight=50.0, success_bonus=20.0):
    """Custom reward with collision penalty and success bonus.

    MetaDrive's default reward is driving-progress based (~0.1-0.5 per step).
    We add:
    - Collision penalty: -collision_weight
    - Arrive destination bonus: +success_bonus
    """
    r = float(env_reward) if env_reward else 0.0
    if info.get("crash_vehicle", False) or info.get("crash", False):
        r -= collision_weight
    if info.get("arrive_destination", False):
        r += success_bonus
    return r


def run_k_episodes(env_builder, scenario, t_len, wm, actor, device, bev_size, K,
                   video_dir=None, video_name_prefix="", exploration_noise=0.0,
                   steer_scale=1.0, debug=False):
    """Run K episodes on the SAME scenario. Actor samples stochastically.

    Args:
        exploration_noise: if > 0, add Gaussian noise to actions (std=noise)
                           to ensure diversity across K runs.
    Returns:
        log_probs_list: list of K tensors, each (T_k,) summed log_prob per step
        custom_rewards: list of K floats (total custom reward per episode)
        results: list of K episode result dicts
    """
    log_probs_list = []
    custom_rewards = []
    results = []
    record = video_dir is not None
    debug_actions = [] if debug else None

    # Create ONE env, reset K times (avoids MetaDrive engine init issues)
    env = env_builder(scenario, t_len)
    env.set_scenario(scenario)
    try:
        for k in range(K):
            obs, info = env.reset()
            ego = env.agent
            if ego is not None:
                p = ego.position
                ego.set_position(
                    [float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)

            prev_state = wm.get_initial_state(1, device)
            prev_action = torch.zeros(1, 2, device=device)
            max_steps = t_len + 150

            ep_log_probs = []
            ep_reward = 0.0
            collision = False
            video_frames = [] if record else None

            for i in range(max_steps):
                if record:
                    frame = capture_bev_video(env)
                    if frame is not None:
                        video_frames.append(frame)

                bev = capture_bev(env, bev_size)
                bev_t = torch.FloatTensor(bev).unsqueeze(0).to(device)

                with torch.no_grad():
                    embed = wm.encode(bev_t)
                    states, _ = wm.rssm.observe(
                        embed.unsqueeze(0), prev_action.unsqueeze(0), prev_state)
                    prev_state = states[-1]
                    feature = wm.get_feature(prev_state)

                # Actor: stochastic forward WITH log_prob (actor.train())
                action, log_prob = actor(feature)
                # Add noise directly to squashed action for diversity
                # (atanh-space noise fails when actions are saturated near ±1)
                if exploration_noise > 0:
                    action = action + torch.randn_like(action) * exploration_noise
                    action = action.clamp(-1, 1)
                # Scale steering to prevent immediate off-road (clone to avoid inplace)
                if steer_scale != 1.0:
                    action = torch.cat([
                        action[:, 0:1] * steer_scale,
                        action[:, 1:2],
                    ], dim=1)
                action_np = action[0].detach().cpu().numpy().astype(np.float32)
                lp = log_prob.sum()  # total log_prob for this step

                # Debug first step of first 2 K-runs
                if debug and i == 0 and k < 2:
                    print(f"    [DEBUG k={k}] action=[{action_np[0]:+.4f},{action_np[1]:+.4f}] "
                          f"lp={lp.item():.1f}", flush=True)

                # MetaDrive calls policy.act() internally → must set global action
                _GLOBAL_ACTION[:] = action_np
                obs, reward, done, truncated, info = env.step(action_np)
                ep_log_probs.append(lp)
                ep_reward += compute_custom_reward(reward, info)

                prev_action = torch.FloatTensor(action_np).unsqueeze(0).to(device)

                if info.get("crash_vehicle", False) or info.get("crash", False):
                    collision = True
                    break
                if info.get("arrive_destination", False):
                    break
                if done or truncated:
                    break

            log_probs_list.append(torch.stack(ep_log_probs))  # (T_k,)
            custom_rewards.append(ep_reward)
            results.append({
                "collision": collision,
                "reward": ep_reward,
                "steps": len(ep_log_probs),
            })

            if record and video_frames:
                video_path = os.path.join(
                    video_dir, f"{video_name_prefix}_k{k}.mp4")
                save_video_ffmpeg(video_frames, video_path)
    finally:
        env.close()

    if debug:
        print(f"  [DEBUG] K rewards: {[f'{r:.1f}' for r in custom_rewards]} "
              f"steps: {[r['steps'] for r in results]} "
              f"collisions: {[r['collision'] for r in results]}", flush=True)
        if len(log_probs_list) >= 2:
            lp0 = log_probs_list[0].sum().item()
            lp1 = log_probs_list[1].sum().item()
            print(f"  [DEBUG] lp[0]={lp0:.1f} lp[1]={lp1:.1f}", flush=True)

    return log_probs_list, custom_rewards, results


def dapo_update(actor, optimizer, log_probs_list, custom_rewards, clip_high=3.0, clip_low=-1.0):
    """DAPO group-based policy gradient update.

    Args:
        log_probs_list: list of K tensors, each (T_k,) summed per step
        custom_rewards: list of K floats
    Returns:
        actor_loss: float
        advantage_stats: (mean, std, max, min) of advantages
    """
    K = len(log_probs_list)
    rewards_t = torch.tensor(custom_rewards, device=next(actor.parameters()).device)  # (K,)

    # Group advantage
    mean_r = rewards_t.mean()
    std_r = rewards_t.std().clamp(min=1e-4)
    advantage = (rewards_t - mean_r) / std_r

    # Asymmetric clipping (DAPO Clip-Higher)
    advantage_clipped = advantage.clamp(min=clip_low, max=clip_high)

    # Sum log_probs across steps for each run → (K,)
    total_log_probs = torch.stack([lp.sum() for lp in log_probs_list])  # (K,)

    # Policy gradient: -log_prob * advantage
    loss = -(total_log_probs * advantage_clipped.detach()).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
    optimizer.step()

    return loss.item(), {
        "adv_mean": advantage.mean().item(),
        "adv_std": advantage.std().item(),
        "adv_max": advantage.max().item(),
        "adv_min": advantage.min().item(),
    }


# ── Model loading ──────────────────────────────────────────────────────────

def load_models(wm_ckpt_path, ac_ckpt_path, device):
    print(f"Loading WM from: {wm_ckpt_path}")
    ckpt = torch.load(wm_ckpt_path, map_location=device)
    saved_cfg = ckpt.get("config", {})
    cfg = Config()
    for k, v in saved_cfg.items():
        if hasattr(cfg, k) and k not in ("train_phase", "logdir", "total_steps"):
            setattr(cfg, k, v)
    wm = WorldModel(cfg).to(device)
    wm.load_state_dict(ckpt["world_model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    feat_dim = wm.feature_dim()
    actor = Actor(feat_dim, action_dim=cfg.action_dim,
                  hidden_dim=cfg.actor_hidden, layers=cfg.actor_layers).to(device)

    ac_ckpt = torch.load(ac_ckpt_path, map_location=device)
    if "actor" in ac_ckpt:
        actor.load_state_dict(ac_ckpt["actor"])
        print(f"  Loaded actor from step {ac_ckpt.get('global_step', '?')}")
    else:
        print("  WARNING: No actor in AC ckpt, using random weights")
    actor.train()

    return wm, actor, cfg


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P4 Online DAPO in real env")
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--ac-ckpt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--loc", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--max-episodes", type=int, default=200,
                        help="Max scenarios to train on")
    parser.add_argument("--k", type=int, default=8, help="DAPO group size")
    parser.add_argument("--k-eval", type=int, default=1,
                        help="K for eval-only scenarios (1 = deterministic)")
    parser.add_argument("--collision-weight", type=float, default=50.0)
    parser.add_argument("--success-bonus", type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Run eval (no update) every N training scenarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logdir", default="./logs/p4_online_dapo")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--record-video", action="store_true",
                        help="Record BEV top-down video for first N training scenarios")
    parser.add_argument("--record-first-n", type=int, default=5,
                        help="Record videos for first N training scenarios (default: 5)")
    parser.add_argument("--video-dir", default=None,
                        help="Directory for recorded videos (default: logdir/videos)")
    parser.add_argument("--exploration-noise", type=float, default=0.3,
                        help="Action noise std for diversity across K runs (0=off)")
    parser.add_argument("--scratch", action="store_true",
                        help="Reinitialize actor from scratch (ignore AC checkpoint)")
    parser.add_argument("--init-log-std", type=float, default=0.0,
                        help="log_std for scratch actor (0.0 → std=0.69)")
    parser.add_argument("--steer-scale", type=float, default=0.5,
                        help="Scale steering by this factor (0.5 = limit to [-0.5, 0.5])")
    parser.add_argument("--reset-log-std", type=float, default=-1.0,
                        help="Reset actor log_std (default -1.0 -> std=0.31, 0=keep trained)")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    video_dir = None
    video_count = 0
    if args.record_video:
        video_dir = args.video_dir or os.path.join(args.logdir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        print(f"Recording first {args.record_first_n} scenarios to: {video_dir}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wm, actor, cfg = load_models(args.wm_ckpt, args.ac_ckpt, device)
    if args.scratch:
        # Reinitialize actor from scratch (Phase 3 policy is bad for real env)
        actor.log_std.data.fill_(args.init_log_std)
        for m in actor.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        print(f"  Actor reinitialized from scratch (log_std={args.init_log_std:.1f}, "
              f"std={float(torch.nn.functional.softplus(actor.log_std).mean()):.2f})")
    elif args.reset_log_std != 0:
        actor.log_std.data.fill_(args.reset_log_std)
        print(f"  Reset actor.log_std to {args.reset_log_std:.1f} "
              f"(std={float(torch.nn.functional.softplus(actor.log_std).mean()):.2f})")
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)

    all_traj = load_merge_cache(args.loc)
    print(f"\nFound {len(all_traj)} validation trajectories")

    from collections import defaultdict
    traj_by_loc = defaultdict(list)
    for t in all_traj:
        traj_by_loc[t["loc_id"]].append(t)

    train_results = []
    eval_results = []
    total_updates = 0
    t_start = time.time()

    for loc_id in args.loc:
        trajs = traj_by_loc.get(loc_id, [])
        if not trajs:
            continue

        print(f"\nLoading map for loc {loc_id} ({LOC_NAMES[loc_id]}) ...", flush=True)
        features, off_x, off_y = build_map_features(loc_id)
        n_scenarios = min(args.max_episodes, len(trajs))
        print(f"Location {loc_id}: {n_scenarios} scenarios | K={args.k} | "
              f"collision_w={args.collision_weight} success_b={args.success_bonus}",
              flush=True)

        for i in range(n_scenarios):
            traj = trajs[i]
            rec_id = traj["rid"]
            tid = traj["tid"]

            # Every eval_every scenarios, run eval (no gradient, K=1)
            is_eval = (i + 1) % args.eval_every == 0

            try:
                scenario_dict, t_len = build_scenario_dict(
                    rec_id, tid, loc_id, copy.deepcopy(features), off_x, off_y,
                    args.data_dir)
                scenario = ScenarioDescription(scenario_dict)

                if is_eval:
                    # Eval mode: K=1, deterministic, no update
                    actor.eval()
                    K_use = args.k_eval
                else:
                    K_use = args.k

                # Video recording for first N training scenarios
                _vid_dir = None
                _vid_prefix = ""
                if not is_eval and video_dir and video_count < args.record_first_n:
                    _vid_dir = video_dir
                    _vid_prefix = f"loc{loc_id}_rec{rec_id}_t{tid}"
                    video_count += 1

                log_probs_list, custom_rewards, ep_results = run_k_episodes(
                    _create_env, scenario, t_len, wm, actor,
                    device, cfg.bev_size, K_use,
                    video_dir=_vid_dir, video_name_prefix=_vid_prefix,
                    exploration_noise=args.exploration_noise if not is_eval else 0.0,
                    steer_scale=args.steer_scale if not is_eval else 1.0,
                    debug=(total_updates < 3))  # debug first 3 training scenarios

                if is_eval:
                    actor.train()
                    for r in ep_results:
                        r.update({"loc_id": loc_id, "rec_id": rec_id,
                                  "track_id": tid, "type": "eval"})
                    eval_results.extend(ep_results)
                else:
                    # DAPO update
                    actor_loss, adv_stats = dapo_update(
                        actor, optimizer, log_probs_list, custom_rewards)
                    total_updates += 1

                    for r in ep_results:
                        r.update({"loc_id": loc_id, "rec_id": rec_id,
                                  "track_id": tid, "type": "train"})
                    train_results.extend(ep_results)

                    if total_updates % 10 == 0:
                        surv_rate = sum(1 for r in ep_results if not r["collision"]) / len(ep_results)
                        avg_r = np.mean(custom_rewards)
                        act_std = float(F.softplus(actor.log_std).mean())
                        print(f"  [update {total_updates}] "
                              f"surv_K={surv_rate:.1%} avgR_K={avg_r:.1f} "
                              f"actor_loss={actor_loss:.2f} act_std={act_std:.2f} "
                              f"adv=({adv_stats['adv_min']:.1f},{adv_stats['adv_max']:.1f})",
                              flush=True)

            except Exception as e:
                print(f"  [scenario {i+1}] rec{rec_id} t{tid} ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

        # Per-location summary
        loc_train = [r for r in train_results if r["loc_id"] == loc_id]
        if loc_train:
            n_surv = sum(1 for r in loc_train if not r["collision"])
            avg_r = np.mean([r["reward"] for r in loc_train])
            print(f"  loc{loc_id} train: survival={n_surv}/{len(loc_train)} "
                  f"({n_surv/len(loc_train)*100:.1f}%) avgR={avg_r:.1f}")

        loc_eval = [r for r in eval_results if r["loc_id"] == loc_id]
        if loc_eval:
            n_surv = sum(1 for r in loc_eval if not r["collision"])
            print(f"  loc{loc_id} eval:  survival={n_surv}/{len(loc_eval)} "
                  f"({n_surv/len(loc_eval)*100:.1f}%)")

        # Save checkpoint
        ckpt_path = os.path.join(args.logdir, f"checkpoint_loc{loc_id}.pt")
        torch.save({
            "actor": actor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": total_updates,
            "config": cfg.to_dict(),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    # ── Final summary ───────────────────────────────────────────────────────
    all_combined = train_results + eval_results
    n = len(all_combined)

    if n > 0:
        survived = sum(1 for r in all_combined if not r["collision"])
        avg_reward = np.mean([r["reward"] for r in all_combined])

        # Training improvement: first half vs second half
        if len(train_results) >= 20:
            half = len(train_results) // 2
            first_half = train_results[:half]
            second_half = train_results[half:]
            first_surv = sum(1 for r in first_half if not r["collision"]) / len(first_half)
            second_surv = sum(1 for r in second_half if not r["collision"]) / len(second_half)
        else:
            first_surv = second_surv = 0.0

        print(f"\n{'='*60}")
        print(f"P4 Online DAPO Results ({n} episodes, {total_updates} updates)")
        print(f"{'='*60}")
        print(f"  Overall survival: {survived}/{n} ({survived/n*100:.1f}%)")
        print(f"  Avg reward: {avg_reward:.1f}")
        print(f"  Train first half:  {first_surv:.1%}")
        print(f"  Train second half: {second_surv:.1%}")
        print(f"  Improvement:       {second_surv - first_surv:+.1%}")
        print(f"{'='*60}")

        summary = {
            "n_episodes": n, "n_updates": total_updates,
            "overall_survival": survived / n,
            "avg_reward": float(avg_reward),
            "train_first_half_survival": first_surv,
            "train_second_half_survival": second_surv,
            "improvement": second_surv - first_surv,
            "train_results": train_results,
            "eval_results": eval_results,
        }
        out_path = os.path.join(args.logdir, "results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved: {out_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
