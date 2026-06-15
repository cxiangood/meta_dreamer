"""
Phase 4: Closed-loop evaluation on exiD validation set (loc1, loc3).

Uses the SAME map + traffic setup as Phase 1 offline data collection:
  - exiD SUMO maps with internal edges (exid_loc{N}_orig.net.xml)
  - Real background traffic replayed from ScenarioDescription
  - Ego vehicle controlled by WorldModelPolicy (frozen WM + AC)

Reports: merge success rate, collision rate, avg episode reward, survival ratio.

Usage:
    # Evaluate on loc1 only
    python eval_exid_phase4.py \
        --wm-ckpt logs/df_sigreg_p2/checkpoint_step2000.pt \
        --data-dir /path/to/exiD-dataset-v2.1/data \
        --loc 1 --max-episodes 10

    # Evaluate on loc1 + loc3 with Phase 3 AC checkpoint
    python eval_exid_phase4.py \
        --wm-ckpt logs/df_sigreg_p2/checkpoint_step2000.pt \
        --ac-ckpt logs/p3_quick/checkpoint_final.pt \
        --data-dir /path/to/exiD-dataset-v2.1/data \
        --loc 1 3 --max-episodes 50

    # Quick smoke test (first 3 episodes, render debug images)
    python eval_exid_phase4.py \
        --wm-ckpt logs/df_sigreg_p2/checkpoint_step2000.pt \
        --data-dir /path/to/exiD-dataset-v2.1/data \
        --loc 1 --max-episodes 3 --debug
"""

import argparse
import json
import math
import os
import sys
import time
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# SUMO setup (local Mac path or HPC)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import Config
from models import WorldModel, Actor
from envs.world_model_policy import WorldModelPolicy, setup_wm_policy

# ── Constants ──────────────────────────────────────────────────────────────

LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}
TRAIN_LOCS = {0, 2, 4, 5, 6}
VAL_LOCS = {1, 3}
BEV_W, BEV_H = 400, 300

# ── Map loading (same as Phase 1 collect_merge_data.py) ────────────────────

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

    # Resample short broken lines (same as Phase 1)
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


# ── Cache loading ──────────────────────────────────────────────────────────

def load_merge_cache(loc_ids):
    """Load merge trajectory list for given locations from cache."""
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
        items = all_items.get(str(loc_id), [])
        trajectories.extend(items)
    return trajectories


# ── Scenario building (same as Phase 1 collect_merge_data.py) ──────────────

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


# ── Episode runner ─────────────────────────────────────────────────────────

def run_episode(env, t_len, max_extra_steps=150, save_video=False):
    """Run one closed-loop episode with WorldModelPolicy.

    WorldModelPolicy.act() is called internally by MetaDrive's agent manager
    on each env.step(). The dummy action [0.0, 0.0] passed to env.step() is
    ignored (same pattern as ReplayEgoCarPolicy in Phase 1 data collection).

    Args:
        env: ScenarioOnlineEnv with WorldModelPolicy configured
        t_len: length of pre-recorded trajectory
        max_extra_steps: extra steps beyond t_len before timeout
        save_video: if True, capture BEV frames for video export

    Returns:
        dict with keys: reward, steps, collision, crash_type, survived, frames, end_reason
    """
    obs, info = env.reset()

    # Fix ego z-height (same as Phase 1)
    ego = env.agent
    start_pos = None
    if ego is not None:
        p = ego.position
        ego.set_position(
            [float(p[0]), float(p[1])],
            height=float(ego.HEIGHT) / 2,
        )
        start_pos = (float(p[0]), float(p[1]))

    total_reward = 0.0
    collision = False
    crash_type = ""
    end_reason = "timeout"
    max_steps = t_len + max_extra_steps
    frames = []

    # Pre-fetch camera reference for frame capture
    cam = None
    engine_origin = None
    if save_video:
        cam = env.engine.sensors.get("rgb_camera")
        engine_origin = env.engine.origin

    for i in range(max_steps):
        # Capture BEV before step (state at beginning of step)
        if save_video and cam is not None:
            try:
                ego = env.agent
                if ego is not None:
                    ego_pos = ego.position
                    ego_hpr = ego.origin.getHpr()
                    bev = cam.perceive(
                        to_float=False,
                        new_parent_node=engine_origin,
                        position=(float(ego_pos[0]), float(ego_pos[1]), 50.0),
                        hpr=(ego_hpr.getX(), -89, 0),
                    )
                    if bev is not None and bev.ndim == 3:
                        frames.append(bev)
            except Exception:
                pass

        # Dummy action — WorldModelPolicy overrides via agent_manager
        obs, reward, done, truncated, info = env.step([0.0, 0.0])

        total_reward += float(reward) if reward else 0.0

        # Out-of-bounds check: vehicle too far from start position
        if ego is not None and start_pos is not None:
            dx = float(ego.position[0]) - start_pos[0]
            dy = float(ego.position[1]) - start_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > 500:  # 500m from start = lost
                end_reason = "out_of_bounds"
                collision = True
                crash_type = "out_of_bounds"
                break

        # Check termination conditions
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

    survived = not collision
    steps_taken = i + 1

    result = {
        "reward": total_reward,
        "steps": steps_taken,
        "collision": collision,
        "crash_type": crash_type,
        "survived": survived,
        "max_steps": max_steps,
        "end_reason": end_reason,
    }
    if save_video:
        result["frames"] = frames
    return result


def _create_env(scenario, t_len):
    """Create a ScenarioOnlineEnv matching Phase 1 data collection config."""
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


# ── Model loading ──────────────────────────────────────────────────────────

def load_models(wm_ckpt_path, ac_ckpt_path, device):
    """Load world model and actor from checkpoint(s).

    Prefers loading config from checkpoint, falls back to defaults.
    """
    print(f"Loading world model from: {wm_ckpt_path}")
    ckpt = torch.load(wm_ckpt_path, map_location=device)

    # Reconstruct config from checkpoint
    saved_cfg = ckpt.get("config", {})
    cfg = Config()
    # Apply saved config values
    for k, v in saved_cfg.items():
        if hasattr(cfg, k) and k not in ("train_phase", "logdir", "total_steps"):
            setattr(cfg, k, v)

    print(f"  Config: reg={cfg.regularization} decoder={cfg.use_decoder} "
          f"barlow={cfg.barlow_lambda if not cfg.use_decoder else 'N/A'} "
          f"bev={cfg.bev_size} downsample={cfg.bev_downsample}"
          f"{' factor=' + str(cfg.cnn_factor) if cfg.bev_downsample == 'cnn' else ''}")

    wm = WorldModel(cfg).to(device)
    wm.load_state_dict(ckpt["world_model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    feat_dim = wm.feature_dim()
    actor = Actor(feat_dim, action_dim=cfg.action_dim,
                  hidden_dim=cfg.actor_hidden, layers=cfg.actor_layers).to(device)
    actor.eval()

    # Load actor weights: try ac_ckpt first, then wm_ckpt, else random
    actor_loaded = False
    if ac_ckpt_path and os.path.exists(ac_ckpt_path):
        print(f"Loading actor from: {ac_ckpt_path}")
        ac_ckpt = torch.load(ac_ckpt_path, map_location=device)
        if "actor" in ac_ckpt:
            actor.load_state_dict(ac_ckpt["actor"])
            actor_loaded = True
            print(f"  AC step: {ac_ckpt.get('global_step', '?')}")
    elif "actor" in ckpt:
        print("Loading actor from WM checkpoint")
        actor.load_state_dict(ckpt["actor"])
        actor_loaded = True

    if not actor_loaded:
        print("WARNING: No actor checkpoint found, using random weights!")
        print("  Expect poor performance. Train Phase 3 first.")

    for p in actor.parameters():
        p.requires_grad = False

    param_count = sum(p.numel() for p in wm.parameters())
    print(f"  WM params: {param_count/1e6:.1f}M | Actor: {sum(p.numel() for p in actor.parameters())/1e3:.0f}K")
    return wm, actor, cfg


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Closed-loop WM+AC eval on exiD validation set")
    parser.add_argument("--wm-ckpt", required=True,
                        help="Path to Phase 2 world model checkpoint (.pt)")
    parser.add_argument("--ac-ckpt", default=None,
                        help="Optional Phase 3 AC checkpoint (default: use WM ckpt's actor)")
    parser.add_argument("--data-dir", required=True,
                        help="Path to exiD data/ directory (containing {NN}_tracks.csv etc.)")
    parser.add_argument("--loc", type=int, nargs="+", default=[1],
                        help="Location IDs to evaluate (default: 1)")
    parser.add_argument("--max-episodes", type=int, default=10,
                        help="Max episodes per location (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-video", action="store_true",
                        help="Save BEV video for each episode")
    parser.add_argument("--video-dir", default="./phase4_videos",
                        help="Directory for video output")
    parser.add_argument("--max-episodes-video", type=int, default=3,
                        help="Max episodes to save video for (default: 3)")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug BEV images for first few episodes")
    parser.add_argument("--out", default="./phase4_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    # Validate locations
    for loc_id in args.loc:
        if loc_id not in VAL_LOCS:
            print(f"WARNING: loc {loc_id} is not a validation location! "
                  f"Valid: {VAL_LOCS}")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    wm, actor, cfg = load_models(args.wm_ckpt, args.ac_ckpt, device)
    setup_wm_policy(wm, actor, bev_size=cfg.bev_size, device=device)

    # Load validation trajectories
    all_traj = load_merge_cache(args.loc)
    print(f"\nFound {len(all_traj)} validation trajectories for loc {args.loc}")

    # Group by location
    from collections import defaultdict
    traj_by_loc = defaultdict(list)
    for traj in all_traj:
        traj_by_loc[traj["loc_id"]].append(traj)

    # Run evaluation
    all_results = []
    env = None
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for loc_id in args.loc:
        trajs = traj_by_loc.get(loc_id, [])
        if not trajs:
            print(f"\n[loc {loc_id}] No trajectories found!")
            continue

        # Load map ONCE per location (same as Phase 1 collect_merge_data.py)
        print(f"\nLoading map for loc {loc_id} ({LOC_NAMES[loc_id]}) ...")
        features, off_x, off_y = build_map_features(loc_id)
        print(f"  {len(features)} map features, offset=({off_x:.1f}, {off_y:.1f})")

        n_episodes = min(args.max_episodes, len(trajs))

        print(f"\n{'='*60}")
        print(f"Location {loc_id} ({LOC_NAMES[loc_id]}): "
              f"{len(trajs)} trajectories, running {n_episodes}")
        print(f"{'='*60}")

        t_start = time.time()

        for i in range(n_episodes):
            traj = trajs[i]
            rec_id = traj["rid"]
            tid = traj["tid"]
            merge_idx = traj.get("merge_idx", -1)

            try:
                # Build scenario
                scenario_dict, t_len = build_scenario_dict(
                    rec_id, tid, loc_id, features, off_x, off_y, args.data_dir)
                scenario = ScenarioDescription(scenario_dict)

                # Create fresh env (same pattern as Phase 1 collect_merge_data.py)
                if env is not None:
                    env.close()
                env = _create_env(scenario, t_len)
                env.set_scenario(scenario)
                # NOTE: env.reset() is called inside run_episode()

                # Run episode
                do_video = args.save_video and i < args.max_episodes_video
                result = run_episode(env, t_len, save_video=do_video)
                frames = result.pop("frames", None)
                result.update({
                    "loc_id": loc_id,
                    "rec_id": rec_id,
                    "track_id": tid,
                    "t_len": t_len,
                    "merge_idx": merge_idx,
                })
                all_results.append(result)

                # Status
                if result["survived"]:
                    status = f"{result['end_reason']}"
                else:
                    status = f"CRASH({result['crash_type']})"
                print(f"  [{i+1}/{n_episodes}] rec{rec_id} t{tid} "
                      f"R={result['reward']:.1f} steps={result['steps']}/{t_len} "
                      f"{status}")

                # Video export
                if do_video and frames:
                    vid_path = os.path.join(args.video_dir,
                                            f"loc{loc_id}_rec{rec_id}_t{tid}.mp4")
                    save_episode_video(frames, vid_path)

                # Debug: save first-frame BEV
                if args.debug and i < 3:
                    _save_debug_bev(env, loc_id, rec_id, tid)

            except Exception as e:
                print(f"  [{i+1}/{n_episodes}] rec{rec_id} t{tid} ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        elapsed = time.time() - t_start
        print(f"  Location {loc_id} done in {elapsed:.1f}s "
              f"({elapsed/n_episodes:.1f}s/ep)")

    if env is not None:
        env.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    if all_results:
        n = len(all_results)
        survived = sum(1 for r in all_results if r["survived"])
        collisions = sum(1 for r in all_results if r["collision"])
        avg_reward = np.mean([r["reward"] for r in all_results])
        avg_steps = np.mean([r["steps"] for r in all_results])
        avg_t_len = np.mean([r["t_len"] for r in all_results])

        # Per-location breakdown
        by_loc = defaultdict(list)
        for r in all_results:
            by_loc[r["loc_id"]].append(r)

        print(f"\n{'='*60}")
        print(f"Phase 4 Results ({n} episodes)")
        print(f"{'='*60}")
        for loc_id in sorted(by_loc.keys()):
            loc_results = by_loc[loc_id]
            ln = len(loc_results)
            loc_survived = sum(1 for r in loc_results if r["survived"])
            loc_avg_r = np.mean([r["reward"] for r in loc_results])
            print(f"  loc{loc_id} ({LOC_NAMES[loc_id]}): "
                  f"survival={loc_survived}/{ln} ({loc_survived/ln*100:.1f}%) "
                  f"avgR={loc_avg_r:.1f}")
        print(f"  ---")
        print(f"  Overall survival: {survived}/{n} ({survived/n*100:.1f}%)")
        print(f"  Collision rate:   {collisions}/{n} ({collisions/n*100:.1f}%)")
        print(f"  Avg reward:       {avg_reward:.1f}")
        print(f"  Avg steps/t_len:  {avg_steps:.0f}/{avg_t_len:.0f} "
              f"({avg_steps/avg_t_len*100:.1f}%)")
        print(f"{'='*60}")

        # Save results
        summary = {
            "n_episodes": n,
            "survival_rate": survived / n,
            "collision_rate": collisions / n,
            "avg_reward": float(avg_reward),
            "avg_steps": float(avg_steps),
            "avg_t_len": float(avg_t_len),
            "per_location": {
                str(loc_id): {
                    "n": len(loc_results),
                    "survival": sum(1 for r in loc_results if r["survived"]),
                    "collisions": sum(1 for r in loc_results if r["collision"]),
                    "avg_reward": float(np.mean([r["reward"] for r in loc_results])),
                }
                for loc_id, loc_results in by_loc.items()
            },
            "episodes": all_results,
        }
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.out}")

    else:
        print("No results collected.")


def _save_debug_bev(env, loc_id, rec_id, tid):
    """Save a debug BEV snapshot from the current env state."""
    import cv2
    out_dir = "./debug_bev"
    os.makedirs(out_dir, exist_ok=True)

    try:
        engine = env.engine
        rgb_cam = engine.sensors.get("rgb_camera")
        if rgb_cam is None:
            return

        ego = env.agent
        if ego is None:
            return

        ego_pos = ego.position
        ego_hpr = ego.origin.getHpr()
        bev = rgb_cam.perceive(
            to_float=False,
            new_parent_node=engine.origin,
            position=(float(ego_pos[0]), float(ego_pos[1]), 50.0),
            hpr=(ego_hpr.getX(), -89, 0),
        )
        if bev is not None and bev.ndim == 3:
            path = os.path.join(out_dir, f"loc{loc_id}_rec{rec_id}_t{tid}.png")
            cv2.imwrite(path, cv2.cvtColor(bev, cv2.COLOR_RGB2BGR))
            print(f"    Debug BEV saved: {path}")
    except Exception as e:
        print(f"    Debug BEV failed: {e}")


def save_episode_video(frames, out_path, fps=25):
    """Export list of BEV frames as MP4 video."""
    import cv2
    if not frames:
        print("  [WARN] No frames to save")
        return
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for codec in ["avc1", "mp4v", "XVID"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer is not None and writer.isOpened():
            break
    else:
        print(f"  [WARN] No video codec available, skipping {out_path}")
        return
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Video saved: {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
