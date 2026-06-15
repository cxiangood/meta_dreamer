"""
P4 DAPO: Online post-training with group-based policy gradient.

Loads P2 WM + P3 actor, runs episodes on exiD val scenarios,
periodically refines the actor via DAPO (K-way WM imagination
with collision proxy from continue_head).

Variant 1: fast — uses WM imagination for K-way rollouts
  (no env snapshot needed, collision proxy instead)

Usage:
    python train_dapo_p4.py \
        --wm-ckpt logs/df_sigreg_p2/checkpoint_step2000.pt \
        --ac-ckpt logs/df_sig_p3/checkpoint_latest.pt \
        --data-dir /path/to/exiD-dataset-v2.1/data \
        --loc 1 3 --max-episodes 100 --k 8
"""

import argparse
import json
import math
import os
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
from metadrive.type import MetaDriveType as MT
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.scenario_env import ScenarioOnlineEnv

from config import Config
from models import WorldModel, Actor
from envs.world_model_policy import WorldModelPolicy, setup_wm_policy, get_global_last_state

# ── Constants ──────────────────────────────────────────────────────────────
LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}
VAL_LOCS = {1, 3}
BEV_W, BEV_H = 400, 300

# ── Map / scenario / cache (same as eval_exid_phase4.py) ────────────────────

def get_map_file(loc_id):
    map_dir = os.path.join(SCRIPT_DIR, "../mirro_data_map")
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
    SPACING = 2.0
    for v in features.values():
        if v.get("type") in (MT.LINE_BROKEN_SINGLE_WHITE, MT.LINE_BROKEN_SINGLE_YELLOW):
            pl = v.get("polyline", [])
            if len(pl) < 4:
                pl = np.array(pl, dtype=np.float32)
                d = pl[-1] - pl[0]
                length = float(np.hypot(d[0], d[1]))
                n = max(int(length / SPACING), 4)
                v["polyline"] = np.array([pl[0]] + [pl[0] + (i / n) * d for i in range(1, n)] + [pl[-1]], dtype=np.float32)
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


def build_scenario_dict(rec_id, track_id, loc_id, map_features, off_x, off_y, data_dir, max_vehicles=100):
    import pandas as pd
    tracks_csv = pd.read_csv(os.path.join(data_dir, f"{rec_id:02d}_tracks.csv"), low_memory=False)
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
        "id": f"exid-{rec_id:02d}-track{sdc_id}", "version": "MetaDrive v0.3.0.1",
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


def _create_env(scenario, t_len):
    return ScenarioOnlineEnv(config=dict(
        use_render=False, image_observation=True,
        agent_policy=WorldModelPolicy,
        horizon=t_len + 200, store_map=False, set_static=True,
        camera_smooth=False,
        vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False,
                            image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, BEV_W, BEV_H)},
        norm_pixel=False, height_scale=0.01,
    ))


# ── DAPO training step ──────────────────────────────────────────────────────

def dapo_step(wm, actor, actor_optimizer, start_states, cfg, K=8):
    """K-way WM imagination → group advantage → PG update.

    start_states: list of N RSSM state dicts (from N different observations).
    """
    B = len(start_states)
    if B == 0:
        return None
    H = cfg.imagination_horizon
    device = next(wm.parameters()).device

    # Stack states into batch (B, ...), then repeat each K times
    deter = torch.stack([s["deter"] for s in start_states]).to(device)  # (B, 1024)
    stoch = torch.stack([s["stoch"] for s in start_states]).to(device)  # (B, 32, 64)

    deter = deter.repeat_interleave(K, dim=0)   # (B*K, 1024)
    stoch = stoch.repeat_interleave(K, dim=0)    # (B*K, 32, 64)
    state = {"deter": deter, "stoch": stoch}

    rewards_list, log_probs_list, collision_risks_list = [], [], []

    for t in range(H):
        feature = wm.get_feature(state)
        action, log_prob = actor(feature.detach())

        with torch.no_grad():
            reward = wm.reward_head(feature).squeeze()
            cont = torch.sigmoid(wm.continue_head(feature).squeeze())
            collision_risk = F.relu(0.3 - cont)
            states_imag = wm.rssm.imagine(action.unsqueeze(0), state)
            state = states_imag[0]

        combined_reward = reward - cfg.dapo_collision_weight * collision_risk
        rewards_list.append(combined_reward)
        log_probs_list.append(log_prob)
        collision_risks_list.append(collision_risk)

    rewards = torch.stack(rewards_list)          # (H, B*K)
    log_probs = torch.stack(log_probs_list)      # (H, B*K)
    collision_risks = torch.stack(collision_risks_list)

    gamma_pow = torch.pow(cfg.gamma, torch.arange(H, device=device, dtype=torch.float32))
    total_rewards = (rewards * gamma_pow.unsqueeze(1)).sum(0)  # (B*K,)
    total_log_probs = log_probs.sum(0)

    total_rewards = total_rewards.reshape(K, B)
    total_log_probs = total_log_probs.reshape(K, B)

    mean_r = total_rewards.mean(0, keepdim=True)
    std_r = total_rewards.std(0, keepdim=True).clamp(min=1e-6)

    # ── DAPO: Dynamic Sampling ──
    # Skip groups where all K trajectories have near-identical rewards
    # (zero-advantage groups produce noisy gradients with no signal)
    valid_mask = std_r.squeeze(0) > 1e-4  # (B,)
    if valid_mask.sum() == 0:
        return {"actor_loss": 0.0, "imag_reward_mean": rewards.mean().item(),
                "collision_risk_mean": collision_risks.mean().item(),
                "return_group_max": total_rewards.max(0).values.mean().item(),
                "dynamic_skip": True}
    if valid_mask.sum() < B:
        total_rewards = total_rewards[:, valid_mask]
        total_log_probs = total_log_probs[:, valid_mask]
        mean_r = total_rewards.mean(0, keepdim=True)
        std_r = total_rewards.std(0, keepdim=True).clamp(min=1e-6)

    # ── DAPO: Clip-Higher (asymmetric) ──
    # Larger upper clip keeps positive advantage exploration active
    advantage = (total_rewards - mean_r) / std_r
    advantage_high = advantage.clamp(max=3.0)   # allow strong positive signal
    advantage_low = advantage.clamp(min=-1.0)     # bound negative penalty
    # Apply asymmetric clipping
    advantage = torch.where(advantage > 0, advantage_high, advantage_low)

    actor_loss = -(total_log_probs * advantage.detach()).mean()
    actor_loss -= cfg.entropy_weight * (-total_log_probs).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
    actor_optimizer.step()

    return {
        "actor_loss": actor_loss.item(),
        "imag_reward_mean": rewards.mean().item(),
        "collision_risk_mean": collision_risks.mean().item(),
        "return_group_max": total_rewards.max(0).values.mean().item(),
    }


# ── Episode runner ──────────────────────────────────────────────────────────

def run_dapo_episode(env, t_len, max_extra_steps=150, collect_states=None):
    """Run one closed-loop episode with WorldModelPolicy (handles WM+actor internally).

    Dummy actions passed to env.step() are overridden by WorldModelPolicy.act().
    RSSM states are collected via get_global_last_state() after each step.
    """
    obs, info = env.reset()
    ego = env.agent
    if ego is not None:
        p = ego.position
        ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)

    total_reward = 0.0
    collision = False
    crash_type = ""
    max_steps = t_len + max_extra_steps

    for i in range(max_steps):
        # Dummy action — WorldModelPolicy overrides via agent_manager
        obs, reward, done, truncated, info = env.step([0.0, 0.0])

        total_reward += float(reward) if reward else 0.0

        # Collect RSSM state for DAPO (from WorldModelPolicy's global state)
        if collect_states is not None:
            state = get_global_last_state()
            if state is not None:
                collect_states.append(state)

        if info.get("crash_vehicle", False):
            collision = True
            crash_type = "vehicle"
            break
        if info.get("crash", False):
            collision = True
            crash_type = info.get("crash_type", "unknown")
            break
        if info.get("arrive_destination", False):
            break
        if done or truncated:
            break

    return {
        "reward": total_reward, "steps": i + 1,
        "collision": collision, "crash_type": crash_type,
        "survived": not collision, "max_steps": max_steps,
    }


# ── Model loading ───────────────────────────────────────────────────────────

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


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P4 DAPO: online post-training")
    parser.add_argument("--wm-ckpt", required=True)
    parser.add_argument("--ac-ckpt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--loc", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--k", type=int, default=8, help="DAPO group size")
    parser.add_argument("--dapo-every", type=int, default=1,
                        help="DAPO update every N episodes")
    parser.add_argument("--dapo-steps", type=int, default=10,
                        help="DAPO updates per training cycle")
    parser.add_argument("--collision-weight", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logdir", default="./logs/dapo_p4")
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load models
    wm, actor, cfg = load_models(args.wm_ckpt, args.ac_ckpt, device)
    cfg.dapo_collision_weight = args.collision_weight

    # Setup WorldModelPolicy BEFORE any env is created
    # (MetaDrive calls WorldModelPolicy.get_input_space() at env init)
    setup_wm_policy(wm, actor, bev_size=cfg.bev_size, device=device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)

    # Load scenarios
    all_traj = load_merge_cache(args.loc)
    print(f"\nFound {len(all_traj)} trajectories for loc {args.loc}")

    # Group by location
    from collections import defaultdict
    traj_by_loc = defaultdict(list)
    for t in all_traj:
        traj_by_loc[t["loc_id"]].append(t)

    # State buffer for DAPO
    state_buffer = []
    max_buffer = 512
    dapo_update = 0
    all_results = []

    t_start = time.time()

    for loc_id in args.loc:
        trajs = traj_by_loc.get(loc_id, [])
        if not trajs:
            continue

        print(f"\nLoading map for loc {loc_id} ({LOC_NAMES[loc_id]}) ...")
        features, off_x, off_y = build_map_features(loc_id)

        n_episodes = min(args.max_episodes, len(trajs))
        print(f"Location {loc_id}: {n_episodes} episodes | K={args.k}")

        env = None

        for i in range(n_episodes):
            traj = trajs[i]
            rec_id = traj["rid"]
            tid = traj["tid"]
            merge_idx = traj.get("merge_idx", -1)

            try:
                scenario_dict, t_len = build_scenario_dict(
                    rec_id, tid, loc_id, features, off_x, off_y, args.data_dir)
                scenario = ScenarioDescription(scenario_dict)
                if env is not None:
                    env.close()
                env = _create_env(scenario, t_len)
                env.set_scenario(scenario)

                ep_states = []
                result = run_dapo_episode(
                    env, t_len,
                    collect_states=ep_states,
                )
                result.update({"loc_id": loc_id, "rec_id": rec_id, "track_id": tid})
                all_results.append(result)

                # Add states to buffer (keep recent ones)
                state_buffer.extend(ep_states)
                if len(state_buffer) > max_buffer:
                    state_buffer = state_buffer[-max_buffer:]

                status = "OK" if result["survived"] else f"CRASH({result['crash_type']})"
                print(f"  [{i+1}/{n_episodes}] rec{rec_id} t{tid} "
                      f"R={result['reward']:.1f} {status} "
                      f"buf={len(state_buffer)}")

                # DAPO training
                if (i + 1) % args.dapo_every == 0 and len(state_buffer) >= cfg.batch_size:
                    for _ in range(args.dapo_steps):
                        indices = np.random.choice(len(state_buffer), cfg.batch_size, replace=True)
                        batch_states = [state_buffer[j] for j in indices]
                        metrics = dapo_step(wm, actor, actor_optimizer,
                                            batch_states, cfg, K=args.k)
                        dapo_update += 1
                        if metrics and dapo_update % 10 == 0:
                            print(f"    [DAPO {dapo_update}] "
                                  f"actor={metrics['actor_loss']:.4f} "
                                  f"rew={metrics['imag_reward_mean']:.4f} "
                                  f"crash_risk={metrics['collision_risk_mean']:.4f}")

            except Exception as e:
                print(f"  [{i+1}/{n_episodes}] rec{rec_id} t{tid} ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

        if env is not None:
            env.close()

        # Save checkpoint per location
        save_path = os.path.join(args.logdir, f"checkpoint_loc{loc_id}.pt")
        torch.save({
            "actor": actor.state_dict(),
            "global_step": dapo_update,
            "config": cfg.to_dict(),
        }, save_path)
        print(f"  Saved: {save_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    n = len(all_results)
    if n > 0:
        survived = sum(1 for r in all_results if r["survived"])
        collisions = sum(1 for r in all_results if r["collision"])
        avg_reward = np.mean([r["reward"] for r in all_results])

        by_loc = defaultdict(list)
        for r in all_results:
            by_loc[r["loc_id"]].append(r)

        print(f"\n{'='*60}")
        print(f"P4 DAPO Results ({n} episodes, {dapo_update} DAPO updates)")
        print(f"{'='*60}")
        for loc_id in sorted(by_loc.keys()):
            loc_results = by_loc[loc_id]
            ln = len(loc_results)
            loc_survived = sum(1 for r in loc_results if r["survived"])
            loc_avg_r = np.mean([r["reward"] for r in loc_results])
            print(f"  loc{loc_id}: survival={loc_survived}/{ln} "
                  f"({loc_survived/ln*100:.1f}%) avgR={loc_avg_r:.1f}")
        print(f"  Overall: survival={survived}/{n} ({survived/n*100:.1f}%) "
              f"collision={collisions}/{n} ({collisions/n*100:.1f}%)")
        print(f"{'='*60}")

        summary = {
            "n_episodes": n, "dapo_updates": dapo_update,
            "survival_rate": survived / n,
            "collision_rate": collisions / n,
            "avg_reward": float(avg_reward),
            "per_location": {
                str(loc_id): {
                    "n": len(loc_results),
                    "survival": sum(1 for r in loc_results if r["survived"]),
                    "avg_reward": float(np.mean([r["reward"] for r in loc_results])),
                }
                for loc_id, loc_results in by_loc.items()
            },
        }
        out_path = os.path.join(args.logdir, "results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved: {out_path}")

        # Save final checkpoint
        final_path = os.path.join(args.logdir, "checkpoint_final.pt")
        torch.save({
            "actor": actor.state_dict(),
            "global_step": dapo_update,
            "config": cfg.to_dict(),
        }, final_path)
        print(f"Final checkpoint: {final_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
