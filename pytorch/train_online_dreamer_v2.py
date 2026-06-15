"""
Online DreamerV3: true online RL on exiD highway merge scenarios.

Follows the original DreamerV3 interleaved training pattern:
  while step < total: interact(10 steps) → train(ratio-based) → repeat

Experimental design (offline WM + online driving):
  Offline Phase 2: image understanding + physics (df_ph WM, frozen online)
  Online: driving only -- no offline BC actor checkpoint
  Phase A (GT warmup): PID → buffer stores executed actions
  Phase B (optional): short actor init from GT buffer (online imitation, not offline BC)
  Phase C (curriculum / RL): actor explores in MetaDrive

Success metric: cleared merge zone (frame >= merge_idx + MERGE_ZONE_FRAMES, no crash).

Usage:
    python train_online_dreamer_v2.py \
        --wm-ckpt logs/df_ph/checkpoint_step14000.pt \
        --actor-init gt_buffer \
        --data-dir /path/to/exiD-dataset-v2.1/data \
        --selection-file ./logs/exid_online_selection.json \
        --video-dir ./online_videos
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import numpy as np
import torch
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

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
from metadrive.engine.engine_utils import close_engine
from metadrive.engine.base_engine import BaseEngine
from metadrive.engine.core.engine_core import EngineCore


def force_reset_engine():
    """Aggressively reset MetaDrive engine singleton so a new env can be created."""
    try:
        close_engine()
    except Exception:
        pass
    # Belt-and-suspenders: directly clear singleton in case close_engine() failed
    try:
        BaseEngine.singleton = None
    except Exception:
        pass
    # Also clear global_config to prevent stale config issues
    try:
        EngineCore.global_config = None
    except Exception:
        pass

from config import Config
from models import WorldModel, Actor, Critic
from models.crash_detector import CrashDetector, CrashFeatureBuffer
from envs.world_model_policy import (
    WorldModelPolicy, setup_wm_policy, get_last_control_is_actor,
    get_global_feature_buffer,
)
from training.replay_buffer import ReplayBuffer
from training.offline_buffer import OfflineDataset

LOC_NAMES = {
    0: "cologne_butzweiler", 1: "cologne_fortiib", 2: "aachen_brand",
    3: "bergheim_roemer", 4: "cologne_klettenberg", 5: "aachen_laurensberg",
    6: "merzenich_rather",
}
BEV_W, BEV_H = 400, 300
MERGE_ZONE_FRAMES = 20  # 20 env steps × 2 frames / 25fps ≈ 1.6s merge window


def merge_zone_cleared(final_step, merge_idx, collision):
    """Success if the vehicle reached the end of the merge window without crashing."""
    if merge_idx < 0:
        return False
    return final_step >= merge_idx + MERGE_ZONE_FRAMES and not collision


def curriculum_gt_blend(track_ep, gt_warmup, curriculum_per_track):
    """GT/PID probability: 1.0 during warmup, linear 1→0 in curriculum, 0 in RL."""
    if track_ep < gt_warmup:
        return 1.0
    curr_i = track_ep - gt_warmup
    if curriculum_per_track > 0 and curr_i < curriculum_per_track:
        return 1.0 - (curr_i + 1) / float(curriculum_per_track)
    return 0.0


def ignore_merge_navigation_done(done, info, ego, step_idx, merge_idx):
    """Keep rolling when MetaDrive sets done=out_of_road but ego is still on a lane.

    ScenarioEnv uses the ramp reference (max_lateral_dist=4 m). During merge the
    ego is often on the main-road lane (on_lane=True) while far from that ref.
    """
    if not done or merge_idx < 0:
        return False
    if not info.get("out_of_road", False):
        return False
    on_lane = getattr(ego, "on_lane", None)
    if on_lane is None or not on_lane:
        return False
    return step_idx < merge_idx + MERGE_ZONE_FRAMES + 30


# ═══════════════════════════════════════════════════════════════════════════
#  Failure Reflection Buffer (Self-Evolving Agent)
# ═══════════════════════════════════════════════════════════════════════════

class FailureBuffer:
    """Stores pre-crash state windows for prioritized imagination rehearsal.

    When an episode ends in crash/out_of_road, the last N steps are extracted
    and stored here. During imagination training, start states are drawn from
    this buffer with 50% probability, forcing the actor to learn crash avoidance.
    """
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.window = 25
        self.obs = []   # list of (C, H, W) uint8 arrays
        self.act = []   # list of (2,) float32 arrays
        self.rew = []   # list of float
        self.con = []   # list of float
        self.gt_act = []  # list of (2,) float32 arrays

    def push_window(self, obs_list, act_list, rew_list, con_list, gt_act_list=None):
        """Store the last `window` steps of a failed episode."""
        n = min(len(obs_list), self.window)
        start = len(obs_list) - n
        for i in range(start, len(obs_list)):
            self.obs.append(obs_list[i])
            self.act.append(act_list[i])
            self.rew.append(rew_list[i])
            self.con.append(con_list[i])
            if gt_act_list and i < len(gt_act_list):
                self.gt_act.append(gt_act_list[i])
        # Trim to capacity
        while len(self.obs) > self.capacity:
            self.obs.pop(0)
            self.act.pop(0)
            self.rew.pop(0)
            self.con.pop(0)
            if self.gt_act:
                self.gt_act.pop(0)

    def sample(self, batch_size):
        """Sample batch_size start states as (obs, act, rew, con) tuples.

        Returns single-step sequences suitable for imagination start states.
        Format matches buffer.sample(batch_size, 1) output.
        """
        if len(self.obs) < batch_size:
            return None
        idx = np.random.choice(len(self.obs), batch_size, replace=False)
        obs_batch = np.stack([self.obs[i] for i in idx])  # (B, C, H, W)
        act_batch = np.stack([self.act[i] for i in idx])  # (B, 2)
        rew_batch = np.array([self.rew[i] for i in idx])  # (B,)
        con_batch = np.array([self.con[i] for i in idx])  # (B,)
        gt_batch = None
        if self.gt_act:
            gt_batch = np.stack([self.gt_act[i] for i in idx])
        return obs_batch, act_batch, rew_batch, con_batch, gt_batch

    def __len__(self):
        return len(self.obs)


# ═══════════════════════════════════════════════════════════════════════════
#  Phase-Aware Reward
# ═══════════════════════════════════════════════════════════════════════════

def _compute_vehicle_distances(ego):
    """Compute ego-centric (longitudinal dx, lateral dy) and TTC for same-direction vehicles.

    Returns:
        nearest_front: (dx, dy, TTC) of nearest vehicle ahead (dx>0, same dir), or None
        nearest_rear:  (dx, dy, TTC) of nearest vehicle behind (dx<0, same dir), or None
        min_gap:       minimum longitudinal gap to any same-direction vehicle
        lateral_overlap: bool, whether any same-direction vehicle has |dy|<2m AND |dx|<15m
    """
    try:
        pos = np.array(ego.position[:2])
        speed_ego = float(ego.speed)
        heading_rad = math.radians(float(ego.heading))
        cos_h, sin_h = math.cos(heading_rad), math.sin(heading_rad)

        nearest_front = None  # (dx, dy, TTC)
        nearest_rear = None   # (dx, dy, TTC)
        min_front_dx = float('inf')
        min_rear_dx = float('inf')
        min_gap = float('inf')
        lateral_overlap = False

        for vid, v in ego.engine.traffic_manager.traffic_vehicles.items():
            if vid == ego.name:
                continue
            vpos = np.array(v.position[:2])
            # World-frame delta
            dwx, dwy = vpos[0] - pos[0], vpos[1] - pos[1]
            # Rotate to ego body frame
            dx = dwx * cos_h + dwy * sin_h      # longitudinal: + = ahead
            dy = -dwx * sin_h + dwy * cos_h     # lateral: + = left

            # Check same direction (heading similarity)
            v_heading = float(getattr(v, 'heading', 0) or 0)
            v_heading_rad = math.radians(v_heading)
            if math.cos(heading_rad - v_heading_rad) < 0.5:
                continue  # not same-direction

            # Speed
            v_speed = float(getattr(v, 'speed', 0) or 0)
            rel_vx = speed_ego * math.cos(heading_rad) - v_speed * math.cos(v_heading_rad)

            # TTC: how many seconds until collision at current relative speed
            # Only meaningful when approaching (rel_vx > 0 means ego faster and closing)
            if rel_vx > 0.5 and dx > 0:
                ttc = dx / rel_vx
            elif rel_vx < -0.5 and dx < 0:
                ttc = abs(dx) / abs(rel_vx)
            else:
                ttc = float('inf')

            # Track nearest front/rear
            if dx > 0 and dx < min_front_dx:
                min_front_dx = dx
                nearest_front = (dx, dy, ttc)
            if dx < 0 and abs(dx) < min_rear_dx:
                min_rear_dx = abs(dx)
                nearest_rear = (dx, dy, ttc)

            # Min gap (absolute)
            min_gap = min(min_gap, abs(dx))

            # Lateral overlap: same lane-level, close longitudinally
            if abs(dy) < 2.0 and abs(dx) < 15.0:
                lateral_overlap = True

        return nearest_front, nearest_rear, min_gap, lateral_overlap
    except Exception:
        return None, None, float('inf'), False


def compute_gt_deviation(ego, gt_positions, gt_headings):
    """Compute cross-track error and heading deviation from GT human trajectory.

    Finds the spatially-closest point on the GT path and computes:
    - Signed lateral deviation (positive = right of GT direction)
    - Heading error (degrees, 0-180)

    Args:
        ego: MetaDrive ego vehicle
        gt_positions: (T, 2) float array of GT x,y (MetaDrive world coordinates)
        gt_headings: (T,) float array of GT headings in degrees
    Returns:
        (abs_lat_dev_m, hdg_err_deg)
    """
    try:
        ego_pos = np.array(ego.position[:2], dtype=np.float64)
        ego_heading = float(ego.heading)
    except Exception:
        return 0.0, 0.0

    # Spatially-closest GT point
    dists = np.linalg.norm(gt_positions - ego_pos, axis=1)
    idx = int(np.argmin(dists))
    gt_pos = gt_positions[idx]
    gt_hdg = float(gt_headings[idx])

    # Cross-track error via cross product with GT heading direction
    gt_hdg_rad = math.radians(gt_hdg)
    along = np.array([math.cos(gt_hdg_rad), math.sin(gt_hdg_rad)], dtype=np.float64)
    lateral = np.array([-along[1], along[0]], dtype=np.float64)  # right-pointing
    delta = ego_pos - gt_pos
    lat_dev = float(np.dot(delta, lateral))  # signed

    # Heading error
    hdg_err = abs(gt_hdg - ego_heading)
    hdg_err = min(hdg_err, 360.0 - hdg_err)

    return abs(lat_dev), float(hdg_err)


def compute_merge_reward(ego, frame, merge_idx, prev_speed=None,
                         prev_lat=None, info=None, start_pos=None,
                         prev_action=None, gt_traj=None):
    """Per-phase reward with same-direction vehicle distances and TTC.

    Phase 0 (ramp):    accelerate + find gap + lateral progress + align heading
    Phase 1 (merge):   survive lane-change + TTC safety + merge progress
    Phase 2 (main):    lane-keep + speed-hold + safe following + smooth

    Terminal: crash=-10, out_of_road=-8, arrive_dest=+10

    gt_traj: dict with keys 'pos' ((T,2) array) and 'heading' ((T,) array in degrees).
             If provided, adds cross-track error penalty to keep ego on the GT path.
    """

    if frame < merge_idx:
        phase = 0
    elif frame < merge_idx + MERGE_ZONE_FRAMES:
        phase = 1
    else:
        phase = 2

    # ── Terminal events ──
    if info is not None:
        if info.get("crash_vehicle", False) or info.get("crash", False):
            return -10.0
        if info.get("arrive_destination", False):
            return 10.0
        if info.get("out_of_road", False):
            return -8.0

    reward = 0.5  # alive bonus (was +0.3)

    # ── Lateral deviation (navigation includes lane change; halved)
    try:
        nav = getattr(ego, 'navigation', None)
        if nav is not None and hasattr(nav, 'current_lateral'):
            lat_dev = abs(float(nav.current_lateral))
            reward -= 0.5 * lat_dev  # (was -1.0)
    except Exception:
        pass

    # ── Ego state ──
    try:
        speed = float(ego.speed)
        pos = np.array(ego.position[:2])
        heading = float(ego.heading)
    except Exception:
        speed, pos, heading = 0.0, np.zeros(2), 0.0

    # ── Same-direction vehicle distances ──
    front, rear, min_gap, lateral_overlap = _compute_vehicle_distances(ego)

    # ── Action smoothness ──
    smooth_penalty = 0.0
    if prev_action is not None:
        try:
            steer = float(getattr(ego, 'steering', 0) or 0)
            throttle = float(getattr(ego, 'throttle', 0) or 0)
        except Exception:
            steer, throttle = 0.0, 0.0
        smooth_penalty = -0.1 * (abs(steer - prev_action[0]) + abs(throttle - prev_action[1]))

    TARGET_SPEED = 30.0  # m/s ≈ 108 km/h

    # ── Global speed incentive (all phases) ──
    # Reward band: accelerate toward target, penalize overspeed + low-speed
    if speed < TARGET_SPEED:
        reward += 0.3 * (speed / TARGET_SPEED)    # 0 at v=0, +0.3 at v=30
    else:
        overspeed = min((speed - TARGET_SPEED) / 5.0, 1.0)
        reward -= 0.2 * overspeed                  # up to -0.2 for 5+ overspeed
    # Low speed = dangerous on highway, regardless of phase
    if speed < 5.0:
        reward -= 0.8                               # nearly stopped
    elif speed < 10.0:
        reward -= 0.4 * (1.0 - speed / 10.0)        # fading penalty

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 0: Pre-merge (ramp → approach merge point)
    # ═══════════════════════════════════════════════════════════════
    if phase == 0:
        # 1. Speed toward target (global already covers most)
        speed_frac = min(speed / TARGET_SPEED, 1.0)
        reward += 0.2 * speed_frac  # (was +0.5)

        # 2. Acceleration bonus
        if prev_speed is not None and speed > prev_speed + 0.3:
            reward += 0.2

        # 3. Lateral progress toward merge point
        if start_pos is not None:
            lateral_dist = np.linalg.norm(pos - start_pos)
            reward += min(0.01 * lateral_dist, 0.2)

        # 5. Gap matching: adjust speed to target gap between front/rear vehicles
        if front is not None and rear is not None:
            gap_dx = front[0] - (-rear[0])  # approximate gap size (longitudinal)
            if gap_dx > 5:
                # Target: put ego in the middle 1/3 of the gap
                gap_center = (front[0] + rear[0]) / 2
                ego_pos_in_gap = 0  # ego is at origin in body frame
                gap_deviation = abs(ego_pos_in_gap - gap_center) / max(gap_dx * 0.5, 1)
                reward += 0.1 * (1.0 - min(gap_deviation, 1.0))

        # 6. Rear vehicle distance: don't cut off traffic behind
        if rear is not None and rear[0] > -20:
            reward -= 0.1 * (1.0 - abs(rear[0]) / 20.0)

        # Lighter proximity/smoothness on ramp
        if front is not None and front[0] < 10:
            reward -= 0.3 * (1.0 - front[0] / 10.0)  # too close to front
        reward += smooth_penalty * 0.5

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 1: Merge-zone (lane change into main road traffic)
    # ═══════════════════════════════════════════════════════════════
    elif phase == 1:
        # Merge window: penalties halved -- lane-change is inherently risky.
        # 1. TTC-based safety
        if front is not None and front[2] < 5.0:
            ttc_f = front[2]
            if ttc_f < 1.5:
                reward -= 1.5                      # (was -3.0)
            elif ttc_f < 3.0:
                reward -= 0.5                      # (was -1.0)
            else:
                reward -= 0.15 * (1.0 - ttc_f / 5.0)  # (was -0.3)

        if rear is not None and rear[2] < 3.0:
            ttc_r = rear[2]
            if ttc_r < 1.0:
                reward -= 1.0                      # (was -2.0)
            else:
                reward -= 0.25 * (1.0 - ttc_r / 3.0)  # (was -0.5)

        # 2. Lateral overlap: normal during lane change, lighter touch
        if lateral_overlap:
            reward -= 1.0                          # (was -2.0/-3.0)

        # 3. Speed match (stronger)
        speed_err = abs(speed - TARGET_SPEED) / TARGET_SPEED
        reward += 0.5 * (1.0 - min(speed_err, 1.0))  # (was +0.3)

        # 4. Merge progress
        merge_progress = (frame - merge_idx) / max(MERGE_ZONE_FRAMES, 1)
        reward += 0.3 * merge_progress             # (was +0.2)

        # 5. Heading
        heading_dev = min(abs(heading) % 360, 360 - abs(heading) % 360)
        if heading_dev > 15:
            reward -= 0.01 * min(heading_dev - 15, 45)

        # 6. Front gap: urgent only when extremely close
        if front is not None and front[0] < 5:
            reward -= 1.0 * (1.0 - front[0] / 5.0)  # (was -2.0)

        reward += smooth_penalty * 0.5             # lighter during merge

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 2: Post-merge (main road cruising)
    # ═══════════════════════════════════════════════════════════════
    else:
        # 1. Speed hold
        speed_err = abs(speed - TARGET_SPEED) / TARGET_SPEED
        reward += 0.5 * (1.0 - min(speed_err, 1.0))

        # 2. Speed stability (penalize oscillation)
        if prev_speed is not None:
            speed_delta = abs(speed - prev_speed)
            if speed_delta > 2.0:
                reward -= 0.05 * min(speed_delta, 10.0)

        # 3. Safe following: maintain distance to front vehicle
        if front is not None and front[0] < 50:
            safe_dist = max(2.0 * speed, 20.0)  # 2-second rule, min 20m
            if front[0] < safe_dist:
                reward -= 0.3 * (1.0 - front[0] / safe_dist)

        # 4. Lane keeping: penalize heading deviation
        heading_dev = min(abs(heading) % 360, 360 - abs(heading) % 360)
        reward -= 0.005 * min(heading_dev, 30)

        # 5. Lateral position: prefer not to drift
        if lateral_overlap:
            reward -= 0.3

        # 6. Smooth driving (stronger weight in cruise)
        reward += smooth_penalty * 2.0

    # ── GT trajectory deviation penalty (all phases) ──
    if gt_traj is not None:
        lat_dev, hdg_err = compute_gt_deviation(ego, gt_traj["pos"], gt_traj["heading"])
        reward -= 2.0 * lat_dev      # cross-track error (m)
        reward -= 0.3 * hdg_err      # heading misalignment (deg)

    return reward


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario Setup (same as Phase 1 / eval_exid_phase4.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_map_file(loc_id):
    map_dir = os.path.join(os.path.dirname(__file__), "../mirro_data_map")
    for name in [f"exid_loc{loc_id}_orig.net.xml", f"exid_loc{loc_id}.net.xml"]:
        path = os.path.join(map_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No SUMO map for {loc_id}")


def build_map_features(loc_id):
    net_xml = get_map_file(loc_id)
    raw_net = sumolib.net.readNet(net_xml, withInternal=True)
    xmin, ymin, xmax, ymax = raw_net.getBoundary()
    off_x = -(xmax + xmin) / 2
    off_y = -(ymax + ymin) / 2
    graph = RoadLaneJunctionGraph(net_xml)
    features = extract_map_features(graph)
    SPACING = 2.0
    for k, v in features.items():
        # Sanitize empty/scalar polygon to pass _check_map_features validation.
        # Some junction lanes have empty polygon lists that crash np.mean(., axis=0).
        # Remove only the invalid polygon key -- polyline is preserved for rendering.
        if "polygon" in v:
            poly = v["polygon"]
            if not isinstance(poly, (np.ndarray, list, tuple)) or len(poly) == 0:
                del v["polygon"]
            else:
                poly_arr = np.array(poly, dtype=np.float64)
                if poly_arr.ndim < 2 or poly_arr.shape[0] < 2:
                    del v["polygon"]
        # Resample short broken-line polylines
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
            default_len, default_wid = float(mrow["length"]), float(mrow["width"])
            default_height = _CLASS_HEIGHT.get(str(mrow.get("class", "car")), 1.5)
        except (KeyError, TypeError):
            default_len, default_wid, default_height = 4.5, 2.0, 1.5
        sub = window[window["trackId"] == tid].sort_values("frame")
        if len(sub) == 0:
            continue
        pos, heading_arr, vel = np.zeros((t_len, 3), dtype=np.float32), \
                                 np.zeros(t_len, dtype=np.float32), \
                                 np.zeros((t_len, 2), dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)
        for _, r in sub.iterrows():
            fi = int(r["frame"]) - f0
            if 0 <= fi < t_len:
                pos[fi, 0] = float(r["xCenter"]) + off_x
                pos[fi, 1] = float(r["yCenter"]) + off_y
                heading_arr[fi] = math.radians(float(r["heading"]))
                vel[fi, 0] = float(r["xVelocity"])
                vel[fi, 1] = float(r["yVelocity"])
                valid[fi] = True
        if not valid.any():
            continue
        tracks[str(tid)] = {
            "type": MT.VEHICLE,
            "state": {"position": pos, "velocity": vel, "heading": heading_arr,
                      "valid": valid,
                      "length": np.full(t_len, default_len, np.float32),
                      "width": np.full(t_len, default_wid, np.float32),
                      "height": np.full(t_len, default_height, np.float32)},
            "metadata": {"type": MT.VEHICLE, "object_id": str(tid), "dataset": "exiD"},
        }
    return {
        "id": f"exid-{rec_id:02d}-track{sdc_id}", "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {"metadrive_processed": True, "coordinate": MT.COORDINATE_METADRIVE,
                     "ts": ts_arr, "sdc_id": str(sdc_id),
                     "scenario_id": f"exid_{rec_id:02d}_{sdc_id}",
                     "dataset": "exiD", "source_file": f"recording_{rec_id:02d}",
                     "ego_vehicle_class": "car",
                     "frame_range": (f0, f1), "location_id": loc_id},
        "tracks": tracks, "dynamic_map_states": {}, "map_features": map_features,
    }, t_len, (f0, f1)


# ═══════════════════════════════════════════════════════════════════════════
#  BEV Capture
# ═══════════════════════════════════════════════════════════════════════════

def capture_bev(env, bev_size=300):
    try:
        engine = env.engine
        rgb_cam = engine.sensors.get("rgb_camera")
        if rgb_cam is None or env.agent is None:
            return None
        ego = env.agent
        ego_pos = ego.position
        ego_hpr = ego.origin.getHpr()
        heading = ego_hpr.getX()
        bev_img = rgb_cam.perceive(
            to_float=False, new_parent_node=engine.origin,
            position=(float(ego_pos[0]), float(ego_pos[1]), 50.0),
            hpr=(heading, -89, 0),
        )
        if bev_img is None or bev_img.ndim != 3:
            return None
        H, W = bev_img.shape[:2]
        size = min(H, W)
        dh, dw = (H - size) // 2, (W - size) // 2
        bev_img = bev_img[dh:dh + size, dw:dw + size]
        if size != bev_size:
            from PIL import Image
            bev_img = np.array(Image.fromarray(bev_img).resize(
                (bev_size, bev_size), Image.BILINEAR))
        return bev_img
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Phase-Staged MoE Actor (Wan-style soft blending)
# ═══════════════════════════════════════════════════════════════════════════

class PhaseBlendedActor(torch.nn.Module):
    """Wrapper that blends 3 phase-specialized actors via phase_head soft routing.

    During env interaction: a = sum(phase_prob_i * actor_i(feature))
    During training: each actor_i is trained on its own imagination rollout,
    creating natural specialization for pre-merge / merge / post-merge phases.
    """
    def __init__(self, actors, wm):
        super().__init__()
        self.actors = torch.nn.ModuleList(actors)
        self.wm = wm
        self._individual = False  # set True during training to get per-actor outputs

    def forward(self, feature, deterministic=False):
        if self._individual:
            # Return per-actor outputs for separate training
            outs = []
            for act in self.actors:
                o = act(feature, deterministic=deterministic)
                outs.append(o)
            return outs  # list of (action, log_prob) tuples

        with torch.no_grad():
            phase_logits = self.wm.phase_head(feature)
            phase_probs = torch.softmax(phase_logits, dim=-1)

        actions, lps = [], []
        for act in self.actors:
            out = act(feature, deterministic=deterministic)
            a, lp = out if isinstance(out, tuple) else (out, None)
            actions.append(a)
            if lp is not None:
                lps.append(lp)
        actions = torch.stack(actions)  # (3, B, A)
        blended = (phase_probs.unsqueeze(-1) * actions).sum(0)  # (B, A)
        if lps:
            lps = torch.stack(lps)  # (3, B)
            blended_lp = (phase_probs * lps).sum(0)  # (B,)
            return blended, blended_lp
        return blended

    def individual_mode(self, mode=True):
        self._individual = mode

    def train(self, mode=True):
        for a in self.actors:
            a.train(mode)
        return self


# ═══════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_models_from_scratch(device, explore_std=5.0, residual_action=True):
    """Create WM + actor + critic with random init (no pretrained checkpoint).

    Standard DreamerV3 from scratch: SIGReg regularization, decoder-free.
    No phase_head (no pretrained classifier), no BC regularization.
    """
    cfg = Config()
    cfg.regularization = "sigreg"
    cfg.sigreg_lambda = 0.1
    cfg.use_decoder = False  # Decoder-free: decoder for 300×300 BEV uses ~75GB OOM
    cfg.barlow_lambda = 0.005  # Barlow Twins temporal alignment (self-supervised)
    cfg.use_phase_head = False
    cfg.use_jepa = False
    cfg.use_traj_head = False
    cfg.batch_length = 50

    print(f"Initializing WM from scratch (random init)")
    print(f"  reg={cfg.regularization} decoder={cfg.use_decoder} "
          f"bev={cfg.bev_size} batch={cfg.batch_size}x{cfg.batch_length} "
          f"explore_std={explore_std}")

    wm = WorldModel(cfg).to(device)
    wm.train()
    for p in wm.parameters():
        p.requires_grad = True

    feat_dim = wm.feature_dim()
    actor = Actor(feat_dim, cfg.action_dim, cfg.actor_hidden, cfg.actor_layers,
                  init_std=explore_std, residual_action=residual_action).to(device)
    critic = Critic(feat_dim, cfg.critic_hidden, cfg.critic_layers).to(device)
    slow_critic = Critic(feat_dim, cfg.critic_hidden, cfg.critic_layers).to(device)

    actor.train()
    critic.train()
    slow_critic.load_state_dict(critic.state_dict())
    for p in slow_critic.parameters():
        p.requires_grad = False

    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.world_lr)
    a_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    print(f"  Actor: residual_action={residual_action}")
    return wm, actor, critic, slow_critic, wm_opt, a_opt, c_opt, cfg


def load_models_for_online(wm_ckpt_path, ac_ckpt_path, device,
                         explore_std=1.0, phase_actors=False,
                         residual_action=True):
    print(f"Loading WM from: {wm_ckpt_path}")
    ckpt = torch.load(wm_ckpt_path, map_location=device)
    saved_cfg = ckpt.get("config", {})
    cfg = Config()
    for k, v in saved_cfg.items():
        if hasattr(cfg, k) and k not in ("train_phase", "logdir", "total_steps"):
            setattr(cfg, k, v)

    # Online overrides
    # Enable phase_head to use DEMO-style phase progress reward in imagination
    cfg.batch_length = 50
    cfg.use_phase_head = True  # DEMO: phase_head provides dense progress signal
    cfg.rssm_phase_conditional = False
    cfg.rssm_moe = False

    print(f"  reg={cfg.regularization} decoder={cfg.use_decoder} "
          f"bev={cfg.bev_size} batch={cfg.batch_size}x{cfg.batch_length}"
          + (" phase_actors=3" if phase_actors else "")
          + f" residual_action={residual_action}")

    # If checkpoint has no phase_head, fall back
    if cfg.use_phase_head and not any('phase_head' in k for k in ckpt["world_model"].keys()):
        cfg.use_phase_head = False
        print("  (checkpoint has no phase_head, disabling phase reward)")
        if phase_actors:
            print("  ERROR: --phase-actors requires a checkpoint with phase_head (use df_ph checkpoint)")
            sys.exit(1)

    wm = WorldModel(cfg).to(device)
    wm.load_state_dict(ckpt["world_model"])
    wm.train()
    for p in wm.parameters():
        p.requires_grad = True

    feat_dim = wm.feature_dim()

    # ── Phase-Staged MoE: 3 actor-critic pairs ──
    if phase_actors:
        actors_raw = [Actor(feat_dim, cfg.action_dim, cfg.actor_hidden, cfg.actor_layers,
                            init_std=explore_std, residual_action=residual_action).to(device)
                      for _ in range(3)]
        actor = PhaseBlendedActor(actors_raw, wm).to(device)
        critics = [Critic(feat_dim, cfg.critic_hidden, cfg.critic_layers).to(device)
                   for _ in range(3)]
        slow_critics = [Critic(feat_dim, cfg.critic_hidden, cfg.critic_layers).to(device)
                        for _ in range(3)]
        # Each actor starts with a slightly different random permutation for specialization
        for i, a in enumerate(actors_raw):
            a._phase_id = i
        # Try loading actor weights from AC checkpoint (copy to all 3)
        if ac_ckpt_path and os.path.exists(ac_ckpt_path):
            ac_ckpt = torch.load(ac_ckpt_path, map_location=device)
            if "actor" in ac_ckpt:
                for a in actors_raw:
                    a.load_state_dict(ac_ckpt["actor"], strict=False)
                print(f"  Phase actors: loaded BC weights (all 3)")
        actor.train()
        for c in critics:
            c.train()
        for sc in slow_critics:
            sc.load_state_dict(critics[0].state_dict())
            for p in sc.parameters():
                p.requires_grad = False
        a_opts = [torch.optim.Adam(a.parameters(), lr=cfg.actor_lr) for a in actors_raw]
        c_opts = [torch.optim.Adam(c.parameters(), lr=cfg.critic_lr) for c in critics]
        wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.world_lr)
        return wm, actor, critics, slow_critics, wm_opt, a_opts, c_opts, cfg

    # ── Single actor (standard) ──
    actor = Actor(feat_dim, cfg.action_dim, cfg.actor_hidden, cfg.actor_layers,
                  init_std=explore_std, residual_action=residual_action).to(device)
    critic = Critic(feat_dim, cfg.critic_hidden, cfg.critic_layers).to(device)
    slow_critic = Critic(feat_dim, cfg.critic_hidden, cfg.critic_layers).to(device)

    actor_loaded = False
    if ac_ckpt_path and os.path.exists(ac_ckpt_path):
        ac_ckpt = torch.load(ac_ckpt_path, map_location=device)
        if "actor" in ac_ckpt:
            ckpt_state = ac_ckpt["actor"]
            missing, unexpected = actor.load_state_dict(ckpt_state, strict=False)
            if missing:
                print(f"  Actor: missing keys (new layers, ok): {missing}")
            if unexpected:
                print(f"  Actor: unexpected keys: {unexpected}")
            actor_loaded = True
    if ac_ckpt_path:
        print("  Actor: random init (offline BC --ac-ckpt is deprecated; use --actor-init gt_buffer)")
    else:
        print(f"  Actor: random init (explore_std={explore_std}), WM-only from checkpoint")

    if "critic" in ckpt and ac_ckpt_path and os.path.exists(ac_ckpt_path):
        critic.load_state_dict(ckpt["critic"])
    actor.train()
    critic.train()
    slow_critic.load_state_dict(critic.state_dict())
    for p in slow_critic.parameters():
        p.requires_grad = False

    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.world_lr)
    a_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    return wm, actor, critic, slow_critic, wm_opt, a_opt, c_opt, cfg


def init_actor_from_online_buffer(wm, actor, buffer, cfg, device, n_steps=200,
                                  prefer_protected=0.85):
    """Online actor warm-start: imitate actions from protected (GT / merge-OK) buffer."""
    if len(buffer) < cfg.batch_length + 1:
        print("  [GT-actor-init] buffer too small, skip")
        return None

    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=1e-5)
    losses = []
    batch_size = min(cfg.batch_size, 8)

    for step in range(n_steps):
        obs, actions, _, _ = buffer.sample(
            batch_size, cfg.batch_length, prefer_protected=prefer_protected)
        B, L = obs.shape[:2]
        with torch.no_grad():
            obs_flat = torch.as_tensor(obs, device=device).reshape(-1, *obs.shape[2:]).float() / 255.0
            embeds = wm.encode(obs_flat).reshape(B, L, -1).permute(1, 0, 2)
            actions_seq = torch.as_tensor(actions, device=device).permute(1, 0, 2)
            prev_state = wm.get_initial_state(B, device)
            post_states, _ = wm.rssm.observe(embeds, actions_seq, prev_state)
            features = torch.stack([wm.get_feature(s) for s in post_states]).reshape(-1, wm.feature_dim())

        target = torch.as_tensor(actions, device=device).reshape(-1, cfg.action_dim)
        pred = actor(features)
        if isinstance(pred, tuple):
            pred = pred[0]
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    for p in wm.parameters():
        p.requires_grad = True
    wm.train()
    mean_loss = float(np.mean(losses)) if losses else 0.0
    print(f"  [GT-actor-init] {n_steps} steps, MSE={mean_loss:.4f} "
          f"(prefer_protected={prefer_protected:.0%})")
    return mean_loss


# ═══════════════════════════════════════════════════════════════════════════
#  Episode Collection
# ═══════════════════════════════════════════════════════════════════════════

def run_collect_chunk(env, t_len, merge_idx, buffer, bev_size=300,
                     max_extra=150, save_frames=False, start_step=0,
                     prev_speed=None, max_steps=None,
                     pid_warmup=False,
                     track_failure=False,
                     debug_per_step=False,
                     gt_traj=None):
    """Collect up to max_steps transitions in an episode (DreamerV3 interleaved).

    WorldModelPolicy drives the ego (PID during pid_warmup, actor otherwise).
    Buffer stores the executed control actually applied in the sim.

    Args:
        pid_warmup: if True, setup_wm_policy was given gt_speeds (PID drives ego).
        track_failure: if True, return last N transitions for failure reflection buffer.
        gt_traj: dict with 'pos' ((T,2) array) and 'heading' ((T,) array), for GT path
                 deviation penalty in compute_merge_reward.
    """
    ego = env.agent
    total_reward = 0.0
    start_pos = None
    if ego is not None:
        try:
            p = ego.position
            start_pos = np.array([float(p[0]), float(p[1])])
        except Exception:
            pass
    max_possible = t_len + max_extra - start_step
    if max_steps is not None:
        max_possible = min(max_possible, max_steps)
    if max_possible <= 0:
        return {"reward": 0.0, "steps": 0, "done": True, "end_reason": "timeout",
                "frames": [], "prev_speed": prev_speed, "failure_data": None}

    frames = [] if save_frames else None
    end_reason = "timeout"
    speed = prev_speed
    collected = 0

    # Failure + action tracking
    fail_obs, fail_act, fail_rew, fail_con = [], [], [], []
    all_actions = []  # track all stored actions for statistics
    merge_actions = []  # (step, steer, throttle, is_actor) around merge point

    for local_i in range(max_possible):
        i = start_step + local_i

        # Capture BEV BEFORE step
        bev = capture_bev(env, bev_size)
        if bev is None:
            bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        if save_frames:
            frames.append(bev.copy())

        # env.step with dummy action -- policy (WorldModelPolicy) controls ego
        obs_dict, _, done, truncated, info = env.step([0.0, 0.0])

        # Get actor action (policy always runs, RSSM state maintained)
        from envs.world_model_policy import _GLOBAL_PREV_ACTION
        try:
            actor_action = _GLOBAL_PREV_ACTION.copy()
        except Exception:
            actor_action = np.zeros(2, dtype=np.float32)

        # Always store what WorldModelPolicy actually executed (PID or actor)
        stored_action = actor_action

        all_actions.append(stored_action.copy())

        # Capture actions around merge point (±20 frames)
        if merge_idx >= 0 and abs(i - merge_idx) <= 20:
            from_actor = get_last_control_is_actor()
            merge_actions.append((i, float(stored_action[0]), float(stored_action[1]), from_actor))
        reward = compute_merge_reward(ego, i, merge_idx, speed, info=info,
                                      start_pos=start_pos, prev_action=stored_action,
                                      gt_traj=gt_traj)
        total_reward += reward

        # ── Per-step diagnostic logging ──
        if debug_per_step:
            try:
                _s = float(ego.speed)
                _h = math.degrees(float(ego.heading_theta))
                _nav = getattr(ego, 'navigation', None)
                _lat = float(_nav.current_lateral) if (_nav is not None and hasattr(_nav, 'current_lateral')) else 0.0
                _on_lane = getattr(ego, 'on_lane', None)
            except Exception:
                _s, _h, _lat, _on_lane = 0.0, 0.0, 0.0, None
            _phase = 0 if i < merge_idx else (1 if i < merge_idx + MERGE_ZONE_FRAMES else 2)
            _pid = "(PID)" if pid_warmup else "(ACT)"
            _done_info = ""
            if info.get("crash_vehicle") or info.get("crash"):
                _done_info = " CRASH"
            elif info.get("out_of_road"):
                _done_info = " OUT_OF_ROAD"
            elif info.get("arrive_destination"):
                _done_info = " ARRIVE_DEST"
            elif done:
                _done_info = f" DONE(on_lane={_on_lane})"
            print(f"  [{_pid}] step={i:3d} phase={_phase} steer={stored_action[0]:+.3f} "
                  f"thr={stored_action[1]:+.3f} spd={_s:.1f} hdg={_h:.1f}° "
                  f"lat={_lat:+.2f}m r={reward:+.2f} cumR={total_reward:+.1f}"
                  f"{_done_info}", flush=True)

        try:
            speed = float(ego.speed)
        except Exception:
            pass

        if bev.ndim == 3 and bev.shape[-1] == 3:
            bev_chw = bev.transpose(2, 0, 1)
        else:
            bev_chw = bev
        is_first = (start_step == 0 and local_i == 0)
        store_done = done and not ignore_merge_navigation_done(done, info, ego, i, merge_idx)
        buffer.add(bev_chw.astype(np.uint8), stored_action, reward, store_done,
                   is_first=is_first)

        # Track for failure reflection
        if track_failure:
            fail_obs.append(bev_chw.astype(np.uint8))
            fail_act.append(stored_action)
            fail_rew.append(reward)
            fail_con.append(float(not store_done))

        collected = local_i + 1

        if info.get("crash_vehicle", False) or info.get("crash", False):
            end_reason = "crash"
            break
        if info.get("arrive_destination", False):
            end_reason = "arrive_destination"
            break
        if info.get("out_of_road", False):
            # ScenarioEnv max_lateral_dist=4 from reference trajectory (ramp lane).
            # During merge, ego can be on main-road lane surface (on_lane=True) but
            # far from ramp ref → out_of_road flag.  Only terminate when truly off
            # any drivable surface.
            on_any_lane = getattr(ego, 'on_lane', None)
            if on_any_lane is not None and not on_any_lane:
                end_reason = "out_of_road"
                break
            # else: on a valid lane → allow lane-change merge, don't terminate
        if done and not ignore_merge_navigation_done(done, info, ego, i, merge_idx):
            # When env terminates with done=True but no crash/out_of_road flag,
            # check whether vehicle is on a drivable surface.
            on_any_lane = getattr(ego, 'on_lane', None)
            if (not info.get("out_of_road", False)
                    and not info.get("crash", False)
                    and not info.get("arrive_destination", False)
                    and on_any_lane is not None and not on_any_lane):
                # Driving off the map/navmesh → effective out_of_road
                end_reason = "out_of_road"
                # total_reward already includes the phase-based reward for this step;
                # swap it for the out_of_road penalty
                total_reward = total_reward - reward - 8.0
            else:
                end_reason = "env_done"
                # Dump info dict to understand WHY done=True
                if debug_per_step:
                    done_keys = {k: v for k, v in info.items()
                                 if any(x in str(k).lower() for x in
                                        ['done', 'out', 'arrive', 'crash', 'max', 'step', 'route', 'completion'])}
                    try:
                        rc = ego.navigation.route_completion if hasattr(ego, 'navigation') else 'N/A'
                    except Exception:
                        rc = 'N/A'
                    print(f"  [DONE] env_done at step={i}, info_keys={done_keys}, "
                          f"route_completion={rc}", flush=True)
                if local_i < 5 or local_i % 50 == 0:
                    print(f"[env] done=True step={i}/{t_len + max_extra - 1} "
                          f"on_lane={on_any_lane}", flush=True)
            break
        if truncated:
            end_reason = "truncated"
            break

    done_episode = end_reason != "timeout"

    # Debug: check act() call count
    from envs.world_model_policy import _GLOBAL_ACT_COUNT
    _act_start = getattr(run_collect_chunk, '_last_act_count', 0)
    act_calls_this_ep = _GLOBAL_ACT_COUNT - _act_start
    run_collect_chunk._last_act_count = _GLOBAL_ACT_COUNT
    if act_calls_this_ep < collected and not pid_warmup:
        print(f"[ACT COUNT MISMATCH] episode collected={collected} steps but "
              f"act() called only {act_calls_this_ep} times! "
              f"start_count={_act_start} end_count={_GLOBAL_ACT_COUNT}", flush=True)

    # Failure window: crash, out_of_road, env_done (driving out of bounds) are all failures
    failure_data = None
    if track_failure and end_reason in ("crash", "crash_vehicle", "out_of_road", "env_done"):
        failure_data = (fail_obs, fail_act, fail_rew, fail_con)

    return {
        "reward": total_reward, "steps": collected, "done": done_episode,
        "end_reason": end_reason, "frames": frames, "prev_speed": speed,
        "failure_data": failure_data,
        "act_mean": float(np.mean([a[0] for a in all_actions])),
        "act_std": float(np.std([a[0] for a in all_actions])),
        "thr_mean": float(np.mean([a[1] for a in all_actions])),
        "thr_std": float(np.std([a[1] for a in all_actions])),
        "merge_actions": merge_actions,
        "merged": merge_zone_cleared(
            start_step + collected - 1, merge_idx,
            end_reason in ("crash", "crash_vehicle")),
        "collision": end_reason in ("crash", "crash_vehicle"),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Eval Episode (WorldModelPolicy)
# ═══════════════════════════════════════════════════════════════════════════

def run_eval_episode(env, t_len, merge_idx, bev_size=300, max_extra=150,
                     save_frames=False, gt_traj=None):
    obs, info = env.reset()
    ego = env.agent
    start_pos = None
    if ego is not None:
        p = ego.position
        ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT) / 2)
        start_pos = (float(p[0]), float(p[1]))

    total_reward = 0.0
    max_steps = t_len + max_extra
    frames = [] if save_frames else None
    end_reason = "timeout"
    collision = False
    crash_type = ""
    prev_speed = None

    for i in range(max_steps):
        if save_frames:
            bev = capture_bev(env, bev_size)
            if bev is not None:
                frames.append(bev.copy())

        obs, _, done, truncated, info = env.step([0.0, 0.0])

        # Phase-aware reward for eval metrics
        reward = compute_merge_reward(ego, i, merge_idx, prev_speed, info=info,
                                      start_pos=start_pos, gt_traj=gt_traj)
        total_reward += reward
        try:
            prev_speed = float(ego.speed)
        except Exception:
            pass

        if ego is not None and start_pos is not None:
            dx = float(ego.position[0]) - start_pos[0]
            dy = float(ego.position[1]) - start_pos[1]
            if (dx*dx + dy*dy)**0.5 > 500:
                end_reason = "out_of_bounds"
                collision = True
                crash_type = "out_of_bounds"
                break

        if info.get("crash_vehicle", False):
            collision, crash_type, end_reason = True, "vehicle", "crash_vehicle"
            break
        if info.get("crash", False):
            collision, crash_type, end_reason = True, info.get("crash_type", "?"), "crash"
            break
        if info.get("out_of_road", False):
            # ScenarioEnv max_lateral_dist=4 from reference trajectory (ramp lane).
            # During merge, ego can be on main-road lane surface (on_lane=True) but
            # far from ramp ref → out_of_road flag.  Only terminate when truly off
            # any drivable surface.
            on_any_lane = getattr(ego, 'on_lane', None)
            if on_any_lane is not None and not on_any_lane:
                end_reason = "out_of_road"
                break
            # else: on a valid lane → allow lane-change merge, don't terminate
        if info.get("arrive_destination", False):
            end_reason = "arrive_destination"
            break
        if done and not ignore_merge_navigation_done(done, info, ego, i, merge_idx):
            end_reason = "env_done"
            break
        if truncated:
            end_reason = "truncated"
            break

    # Merge success: cleared merge window (out_of_road/env_done after merge still counts)
    merged = merge_zone_cleared(i, merge_idx, collision)
    survived = not collision and end_reason not in ("env_done", "out_of_road")

    return {
        "reward": total_reward, "steps": i + 1, "collision": collision,
        "crash_type": crash_type,
        "survived": survived,
        "merged": merged,
        "end_reason": end_reason, "frames": frames,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Video Saving
# ═══════════════════════════════════════════════════════════════════════════

def save_video(frames, path, fps=10):
    if not frames:
        return
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps)
        print(f"  Video saved: {path} ({len(frames)} frames)")
    except ImportError:
        print("  imageio not available, saving frames as npz")
        np.savez_compressed(path.replace(".mp4", ".npz"), frames=np.array(frames))


# ═══════════════════════════════════════════════════════════════════════════
#  Training Helpers
# ═══════════════════════════════════════════════════════════════════════════

class Normalize:
    """Running statistics normalization (EMA), matches embodied.jax.Normalize.

    Original DreamerV3 uses three: retnorm, valnorm, advnorm.
    Critical for stable policy gradient - without this, advantage scales vary
    wildly and actor gradients become unreliable.
    """

    def __init__(self, decay=0.99, max_val=1e8):
        self.decay = decay
        self.max_val = max_val
        self.mean = 0.0
        self.std = 1.0
        self.n = 0

    def __call__(self, x, update=True):
        """Normalize x, optionally updating running stats.

        Args:
            x: tensor of any shape
            update: if True, update EMA stats (set False during eval)
        Returns:
            normalized x (same shape)
        """
        if update:
            m = x.detach().mean().item()
            s = x.detach().std(unbiased=False).item()
            if self.n == 0:
                self.mean = m
                self.std = max(s, 1e-8)
            else:
                self.mean = self.decay * self.mean + (1 - self.decay) * m
                self.std = max(self.decay * self.std + (1 - self.decay) * s, 1e-8)
            self.n += 1
        return (x - self.mean) / max(self.std, 1e-8)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "n": self.n}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.std = d["std"]
        self.n = d["n"]


# Global normalization instances (one per training run)
_G_RETNORM = Normalize(decay=0.99)
_G_VALNORM = Normalize(decay=0.99)
_G_ADVNORM = Normalize(decay=0.99)


def train_wm_online(wm, buffer, wm_opt, cfg, device, success_sample_ratio=0.4):
    """Train WM on online replay buffer only (no offline data mixing).

    Offline data has a different reward distribution (speed-based scalars)
    vs online data (phase-aware + crash/off-road penalties). Mixing them
    confuses the reward head and makes imagination returns unreliable.
    """
    on_result = buffer.sample(
        cfg.batch_size, cfg.batch_length, prefer_protected=success_sample_ratio)
    on_obs_np, on_act_np, on_rew_np, on_con_np = on_result[:4]
    obs = torch.FloatTensor(on_obs_np).to(device).permute(1, 0, 2, 3, 4) / 255.0
    actions = torch.FloatTensor(on_act_np).to(device).permute(1, 0, 2)
    rewards = torch.FloatTensor(on_rew_np).to(device).permute(1, 0)
    # contdisc: scale continues by (1-1/horizon) to encourage WM to learn termination
    continues = torch.FloatTensor(on_con_np).to(device).permute(1, 0)
    continues = continues * (1.0 - 1.0 / max(cfg.horizon, 333))

    prev_state = wm.get_initial_state(obs.shape[1], device)
    wm_opt.zero_grad()
    loss, metrics = wm.compute_world_loss(obs, actions, rewards, continues, prev_state)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
    wm_opt.step()
    return metrics


def train_ac_imagination(wm, actor, critic, slow_critic, a_opt, c_opt,
                         buffer, cfg, device,
                         failure_buffer=None, phase_actors=False,
                         success_sample_ratio=0.4,
                         bc_anchor_weight=0.05,
                         gt_bc_weight=0.3,
                         crash_detector=None, crash_ready=False):
    """DreamerV3 AC imagination training -- matches original DreamerV3 (agent.py).

    Key design (from original):
      - contdisc: c *= (1 - 1/horizon), so continue head learns episode termination
      - lambda returns with slow-critic bootstrap (no explicit γ, disc=1)
      - Return/Advantage normalization (EMA running stats) for stable PG
      - Continue-weighted policy loss (cumprod(c) downweights post-termination steps)
      - Real-first-step: prepend real feature to imagination for grounding

    Failure Reflection (--failure-reflection):
      With 50% probability, sample start states from the failure buffer instead of
      the main buffer. This forces the actor to rehearse crash-avoidance on
      states that historically led to crashes, "reflecting" on past failures.

    Phase-Staged MoE (--phase-actors):
      Randomly selects one of 3 phase-specialized actor-critic pairs to train
      each step. Natural specialization emerges as each actor sees different
      training examples over time.
    """
    retnorm = _G_RETNORM

    # ── Select which actor/critic to train (Phase-Staged MoE) ──
    if phase_actors:
        train_phase_id = np.random.randint(0, 3)
        sel_actor = actor.actors[train_phase_id]
        sel_critic = critic[train_phase_id]
        sel_slow = slow_critic[train_phase_id]
        sel_a_opt = a_opt[train_phase_id]
        sel_c_opt = c_opt[train_phase_id]
    else:
        sel_actor, sel_critic, sel_slow = actor, critic, slow_critic
        sel_a_opt, sel_c_opt = a_opt, c_opt

    # ── Sample start states (failure reflection: 50% from failure buffer) ──
    use_fb = (failure_buffer is not None and len(failure_buffer) >= cfg.batch_size
              and np.random.random() < 0.5)
    fb_actions_np = None  # crash-causing actions for negative margin loss
    if use_fb:
        fb_result = failure_buffer.sample(cfg.batch_size)
        obs_np = fb_result[0]  # (B, C, H, W)
        obs = torch.FloatTensor(obs_np).to(device) / 255.0
        fb_actions_np = fb_result[1]  # crash-causing actions
    else:
        sample_result = buffer.sample(
            cfg.batch_size, 1, prefer_protected=success_sample_ratio)
        obs_np = sample_result[0]
        obs = torch.FloatTensor(obs_np).to(device).squeeze(1) / 255.0

    # ── Encode real start state (grounding) ──
    H = cfg.imagination_horizon
    B = cfg.batch_size
    contdisc_factor = 1.0 - 1.0 / max(cfg.horizon, 333)  # match original DreamerV3 ~0.997
    lam = 0.95
    actent = 1e-2

    with torch.no_grad():
        embed = wm.encode(obs)
        prev_state = wm.get_initial_state(B, device)
        dummy = torch.zeros(B, cfg.action_dim, device=device)
        states, _ = wm.rssm.observe(embed.unsqueeze(0), dummy.unsqueeze(0), prev_state)
        start_state = states[-1]

    # ── Imagination rollout (H steps) ──
    imag_features, imag_rewards, imag_continues, imag_log_probs = [], [], [], []
    imag_p0_bonus, imag_p1_bonus, imag_p2_bonus = [], [], []
    imag_steer, imag_throttle = [], []
    state = start_state
    prev_a = torch.zeros(B, cfg.action_dim, device=device)
    for t in range(H):
        feature = wm.get_feature(state)
        action, lp = sel_actor(feature.detach(), prev_action=prev_a)
        steer, throttle = action[:, 0], action[:, 1]
        imag_steer.append(steer)
        imag_throttle.append(throttle)
        with torch.no_grad():
            c_raw = wm.continue_head(feature).squeeze()
            c = torch.sigmoid(c_raw) * contdisc_factor
            if wm.phase_head is not None:
                # ── Phase progress (longitudinal) ──
                phase_logits = wm.phase_head(feature)
                phase_probs = torch.softmax(phase_logits, dim=-1)
                phase_values = torch.tensor([0.0, 0.5, 1.0], device=device)
                base_reward = (phase_probs * phase_values).sum(-1)

                # ── Per-phase action guidance ──
                # Phase 0 (匝道): 加速到高速 + 车道保持
                p0_thr = torch.exp(-((throttle - 0.6) / 0.5) ** 2)   # encourage accel, peak 0.6
                p0_steer = -0.5 * steer ** 2                          # lane keeping
                # Phase 1 (换道汇入, ~1.6s): 换道需要转向 + 不能刹车
                p1_thr = torch.exp(-((throttle - 0.4) / 0.5) ** 2)   # moderate, don't brake
                p1_steer = -((steer.abs() - 0.2) / 0.4) ** 2         # encourage |steer|≈0.2
                # Phase 2 (主路): 车道保持 + 平稳巡航
                p2_thr = torch.exp(-((throttle - 0.35) / 0.4) ** 2)  # steady cruise, peak 0.35
                p2_steer = -0.5 * steer ** 2                          # lane keeping

                # Soft-routed by phase_probs, weight 0.5 (increased to give strong action gradient)
                phase_action = (
                    phase_probs[:, 0] * (p0_thr + p0_steer) +
                    phase_probs[:, 1] * (p1_thr + p1_steer) +
                    phase_probs[:, 2] * (p2_thr + p2_steer)
                )
                action_bonus = 0.5 * phase_action

                # Crash penalty: online detector when ready, else continue_head heuristic
                if crash_ready and crash_detector is not None:
                    crash_penalty = crash_detector(feature)
                else:
                    crash_penalty = torch.nn.functional.relu(0.3 - c)
                r = base_reward + action_bonus - 10.0 * crash_penalty
                p0_bonus = base_reward
                p1_bonus = action_bonus
                p2_bonus = torch.zeros_like(base_reward)
                imag_p0_bonus.append(p0_bonus)
                imag_p1_bonus.append(p1_bonus)
                imag_p2_bonus.append(p2_bonus)
            else:
                r = wm.reward_head(feature).squeeze()
        with torch.no_grad():
            state = wm.rssm.imagine(action.unsqueeze(0), state)[0]
            prev_a = action
        imag_features.append(feature)
        imag_rewards.append(r)
        imag_continues.append(c)
        imag_log_probs.append(lp)

    imag_features = torch.stack(imag_features)  # (H, B, feat_dim)
    imag_rewards = torch.stack(imag_rewards)    # (H, B)
    imag_continues = torch.stack(imag_continues)  # (H, B)
    imag_log_probs = torch.stack(imag_log_probs)  # (H, B, act_dim)

    # ── Critic values (second pass) ──
    with torch.no_grad():
        sv_all, _ = sel_slow(imag_features.reshape(-1, imag_features.shape[-1]))
        sv_all = sv_all.reshape(H, B)
    sv = sv_all

    # ── Lambda returns (matches original DreamerV3, disc=1 in contdisc mode) ──
    rets = torch.zeros(H, B, device=device)
    rets[-1] = imag_rewards[-1] + imag_continues[-1] * sv[-1]
    for t in range(H - 2, -1, -1):
        n_step = imag_rewards[t] + imag_continues[t] * sv[t + 1]
        rets[t] = n_step + imag_continues[t] * lam * (rets[t + 1] - sv[t + 1])

    # ── Return normalization (EMA running stats) ──
    rets_normed = retnorm(rets, update=True)  # (H, B) normalized

    # ── Continue-weighted policy loss (original DreamerV3) ──
    # weight[t] = cumprod(c[0..t]) -- steps beyond termination get zero weight
    with torch.no_grad():
        weight = torch.cumprod(imag_continues, dim=0)  # (H, B)
        # Normalize so mean weight ≈ 1 (prevents loss scale drift)
        weight = weight / weight.mean().clamp(min=1e-8)

    # ── Critic loss (2-hot cross-entropy, symlog targets) ──
    v_all, vl_all = sel_critic(imag_features.detach().reshape(-1, imag_features.shape[-1]))
    v_all = v_all.reshape(H, B)
    vl_all = vl_all.reshape(-1, 255)

    # Value target: retnorm-normalized returns (H steps → H*B targets)
    vt = sel_critic.compute_target(rets_normed.detach().reshape(-1))  # (H*B,)

    # Value loss: cross-entropy with continue weighting
    c_loss_raw = torch.nn.functional.cross_entropy(
        vl_all, vt, reduction='none')  # (H*B,)
    c_loss = (weight.detach().reshape(-1) * c_loss_raw).mean()
    # Slow-reg: regularize critic output toward slow critic
    with torch.no_grad():
        sv_normed = retnorm(sv, update=False)
    mse_raw = torch.nn.functional.mse_loss(
        v_all, sv_normed.detach(), reduction='none')  # (H, B)
    slowreg = 1.0 * (weight.detach() * mse_raw).mean()

    sel_c_opt.zero_grad()
    (c_loss + slowreg).backward()
    torch.nn.utils.clip_grad_norm_(sel_critic.parameters(), 100.0)
    sel_c_opt.step()

    # ── Advantage (no advnorm -- match DreamerV3 advnorm: none) ──
    adv = rets - sv.detach()  # (H, B)

    # ── Actor loss (REINFORCE + entropy bonus, continue-weighted) ──
    a_loss = -(weight.detach() * imag_log_probs * adv.detach()).mean()
    # Entropy bonus (encourages exploration, original actent=3e-4)
    a_loss = a_loss + actent * imag_log_probs.mean()

    # ── Failure reflection: negative margin loss ──
    neg_margin_loss = 0.0
    if fb_actions_np is not None:
        try:
            fb_act_tensor = torch.FloatTensor(fb_actions_np).to(device)
            with torch.no_grad():
                embed_fb = wm.encode(obs)
                prev_fb = wm.get_initial_state(cfg.batch_size, device)
                _, states_fb = wm.rssm.observe(
                    embed_fb.unsqueeze(0), fb_act_tensor.unsqueeze(0), prev_fb)
                feat_fb = wm.get_feature(states_fb[-1])
            pred_fb, _ = sel_actor(feat_fb, deterministic=True)
            margin = 0.5
            dist = torch.norm(pred_fb - fb_act_tensor, dim=-1)
            neg_margin_loss = torch.clamp(margin - dist, min=0).mean()
        except Exception:
            pass

    a_loss_total = a_loss + 0.05 * neg_margin_loss

    # ── Action L2 regularization: penalize extreme actions to prevent saturation ──
    # steer → 0 (center), throttle → 0.5 (moderate speed)
    if len(imag_steer) > 0:
        all_steer = torch.stack(imag_steer)   # (H, B)
        all_throttle = torch.stack(imag_throttle)  # (H, B)
        action_l2 = 0.02 * (all_steer.pow(2).mean() + (all_throttle - 0.5).pow(2).mean())
        a_loss_total = a_loss_total + action_l2
    else:
        action_l2 = 0.0

    # ── BC anchor: single-frame MSE on protected actions (light touch) ──
    bc_anchor_loss = 0.0
    if bc_anchor_weight > 0:
        try:
            bc_result = buffer.sample(cfg.batch_size, 1, prefer_protected=1.0)
            bc_obs_np, bc_act_np = bc_result[0], bc_result[1]
            bc_obs = torch.FloatTensor(bc_obs_np).to(device).squeeze(1) / 255.0
            bc_target = torch.FloatTensor(bc_act_np).to(device).squeeze(1)
            with torch.no_grad():
                bc_embed = wm.encode(bc_obs)
                bc_prev = wm.get_initial_state(cfg.batch_size, device)
                bc_dummy = torch.zeros(cfg.batch_size, cfg.action_dim, device=device)
                bc_states, _ = wm.rssm.observe(bc_embed.unsqueeze(0), bc_dummy.unsqueeze(0), bc_prev)
                bc_feat = wm.get_feature(bc_states[-1])
            bc_pred, _ = sel_actor(bc_feat, deterministic=True)
            bc_anchor_loss = torch.nn.functional.mse_loss(bc_pred, bc_target)
            a_loss_total = a_loss_total + bc_anchor_weight * bc_anchor_loss
        except Exception:
            pass

    # ── GT Trajectory BC: anchor actor to PID trajectory segments ──
    # Uses protected buffer (GT warmup PID actions). Phase-aware weighting:
    #   Phase 0 (ramp):     0.5x -- moderate guidance for acceleration + lane-keep
    #   Phase 1 (merge):    0.3x -- light touch, let actor explore lane-change
    #   Phase 2 (post-merge): 1.0x -- STRONG lane-keeping anchor to prevent drift
    gt_bc_loss = 0.0
    if gt_bc_weight > 0:
        try:
            L_bc = min(cfg.batch_length, 20)
            bc_result = buffer.sample(cfg.batch_size, L_bc, prefer_protected=1.0)
            bc_obs_np, bc_act_np = bc_result[0], bc_result[1]
            bc_obs = torch.FloatTensor(bc_obs_np).to(device).permute(1, 0, 2, 3, 4) / 255.0
            bc_target = torch.FloatTensor(bc_act_np).to(device).permute(1, 0, 2)

            with torch.no_grad():
                bc_embed = wm.encode(bc_obs.reshape(-1, *bc_obs.shape[2:]))
                bc_embed = bc_embed.reshape(L_bc, cfg.batch_size, -1)
                bc_prev = wm.get_initial_state(cfg.batch_size, device)
                # Use GT (PID) actions for RSSM -- produces correct state sequence
                bc_states, _ = wm.rssm.observe(bc_embed, bc_target, bc_prev)
                bc_features = torch.stack([wm.get_feature(s) for s in bc_states])

            # Teacher forcing: GT as prev_action, compare predictions to GT
            bc_preds = []
            prev_a = torch.zeros(cfg.batch_size, cfg.action_dim, device=device)
            for t in range(L_bc):
                pred, _ = sel_actor(bc_features[t].detach(), prev_action=prev_a, deterministic=True)
                bc_preds.append(pred)
                prev_a = bc_target[t]
            bc_preds = torch.stack(bc_preds)

            # Phase-aware per-frame weighting
            if wm.phase_head is not None:
                with torch.no_grad():
                    ph_logits = wm.phase_head(bc_features.reshape(-1, bc_features.shape[-1]))
                    ph_probs = torch.softmax(ph_logits.reshape(L_bc, cfg.batch_size, -1), dim=-1)
                ph_w = (0.5 * ph_probs[:, :, 0] +
                        0.3 * ph_probs[:, :, 1] +
                        1.0 * ph_probs[:, :, 2])
                per_step_mse = ((bc_preds - bc_target) ** 2).mean(dim=-1)
                gt_bc_loss = (ph_w * per_step_mse).mean()
            else:
                gt_bc_loss = torch.nn.functional.mse_loss(bc_preds, bc_target)

            a_loss_total = a_loss_total + gt_bc_weight * gt_bc_loss
        except Exception:
            pass

    sel_a_opt.zero_grad()
    a_loss_total.backward()
    torch.nn.utils.clip_grad_norm_(sel_actor.parameters(), 100.0)
    sel_a_opt.step()

    # EMA update for slow critic
    for ps, pf in zip(sel_slow.parameters(), sel_critic.parameters()):
        ps.data.lerp_(pf.data, 0.005)

    metrics = {"actor_loss": a_loss.item(),
               "critic_loss": c_loss.item(),
               "bc_anchor_loss": bc_anchor_loss.item() if isinstance(bc_anchor_loss, torch.Tensor) else 0.0,
               "gt_bc_loss": gt_bc_loss.item() if isinstance(gt_bc_loss, torch.Tensor) else 0.0,
               "imag_reward_mean": imag_rewards.mean().item(),
               "imag_reward_std": imag_rewards.std().item(),
               "imag_continue_mean": imag_continues.mean().item(),
               "adv_mag": adv.abs().mean().item(),
               "ret_range": f"[{rets.min().item():.1f}, {rets.max().item():.1f}]",
               "weight_mean": weight.mean().item(),
               "has_phase_reward": wm.phase_head is not None,
               "imag_action_bonus": wm.phase_head is not None,
               "use_failure_buffer": use_fb}
    if imag_steer:
        steer_t = torch.stack(imag_steer)
        throttle_t = torch.stack(imag_throttle)
        metrics["act_steer_std"] = steer_t.std().item()
        metrics["act_throttle_std"] = throttle_t.std().item()
    if imag_p0_bonus:
        metrics["p0_bonus_mean"] = torch.stack(imag_p0_bonus).mean().item()
        metrics["p1_bonus_mean"] = torch.stack(imag_p1_bonus).mean().item()
        metrics["p2_bonus_mean"] = torch.stack(imag_p2_bonus).mean().item()
    return metrics


def train_ac_real(wm, actor, critic, slow_critic, a_opt, c_opt,
                  buffer, cfg, device, phase_actors=False):
    """Train AC on REAL trajectory data from buffer using stored env rewards.

    经验回放: actor sees actual env rewards (crash=-10, out_of_road=-8, etc.),
    not synthetic imagination rewards. Combined with imagination training,
    this provides grounding in real consequences.

    Half batch from protected data (positive anchor), half from any (negative).
    """
    B = cfg.batch_size
    L = min(cfg.batch_length, 20)
    contdisc_factor = 1.0 - 1.0 / max(cfg.horizon, 333)
    lam = 0.95

    # ── Selectors (same logic as train_ac_imagination) ──
    if phase_actors:
        train_phase_id = np.random.randint(0, 3)
        sel_actor = actor.actors[train_phase_id]
        sel_critic = critic[train_phase_id]
        sel_slow = slow_critic[train_phase_id]
        sel_a_opt = a_opt[train_phase_id]
        sel_c_opt = c_opt[train_phase_id]
    else:
        sel_actor, sel_critic, sel_slow = actor, critic, slow_critic
        sel_a_opt, sel_c_opt = a_opt, c_opt

    half = max(B // 2, 1)
    result_prot = buffer.sample(half, L, prefer_protected=1.0)
    result_any = buffer.sample(B - half, L, prefer_protected=0.0)

    def process(result):
        onp, anp, rnp, cnp = result[:4]
        obs = torch.FloatTensor(onp).to(device).permute(1, 0, 2, 3, 4) / 255.0
        act = torch.FloatTensor(anp).to(device).permute(1, 0, 2)
        rew = torch.FloatTensor(rnp).to(device).permute(1, 0)
        con = torch.FloatTensor(cnp).to(device).permute(1, 0) * contdisc_factor
        return obs, act, rew, con

    obs_p, act_p, rew_p, con_p = process(result_prot)
    obs_a, act_a, rew_a, con_a = process(result_any)
    obs_all = torch.cat([obs_p, obs_a], dim=1)
    act_all = torch.cat([act_p, act_a], dim=1)
    rew_all = torch.cat([rew_p, rew_a], dim=1)
    con_all = torch.cat([con_p, con_a], dim=1)

    with torch.no_grad():
        embed = wm.encode(obs_all.reshape(-1, *obs_all.shape[2:]))
        embed = embed.reshape(L, B, -1)
        prev_state = wm.get_initial_state(B, device)
        states, _ = wm.rssm.observe(embed, act_all, prev_state)
        features = torch.stack([wm.get_feature(s) for s in states])

    prev_a = torch.zeros(B, cfg.action_dim, device=device)
    real_log_probs = []
    for t in range(L):
        _, lp = sel_actor(features[t].detach(), prev_action=prev_a)
        real_log_probs.append(lp)
        prev_a = act_all[t]
    real_log_probs = torch.stack(real_log_probs)

    v_all, vl_all = sel_critic(features.reshape(-1, features.shape[-1]))
    v_all = v_all.reshape(L, B)
    vl_all = vl_all.reshape(-1, 255)

    with torch.no_grad():
        sv_all, _ = sel_slow(features.detach().reshape(-1, features.shape[-1]))
        sv_all = sv_all.reshape(L, B)

    rets = torch.zeros(L, B, device=device)
    rets[-1] = rew_all[-1] + con_all[-1] * sv_all[-1]
    for t in range(L - 2, -1, -1):
        n_step = rew_all[t] + con_all[t] * sv_all[t + 1]
        rets[t] = n_step + con_all[t] * lam * (rets[t + 1] - sv_all[t + 1])

    rets_normed = _G_RETNORM(rets, update=True)

    with torch.no_grad():
        weight = torch.cumprod(con_all, dim=0)
        weight = weight / weight.mean().clamp(min=1e-8)

    vt = sel_critic.compute_target(rets_normed.detach().reshape(-1))
    c_loss = (weight.detach().reshape(-1) *
              torch.nn.functional.cross_entropy(vl_all, vt, reduction='none')).mean()
    mse_raw = torch.nn.functional.mse_loss(
        v_all, _G_RETNORM(sv_all, update=False).detach(), reduction='none')
    slowreg = 1.0 * (weight.detach() * mse_raw).mean()

    sel_c_opt.zero_grad()
    (c_loss + slowreg).backward()
    torch.nn.utils.clip_grad_norm_(sel_critic.parameters(), 100.0)
    sel_c_opt.step()

    adv = rets - sv_all.detach()
    a_loss = -(weight.detach() * real_log_probs * adv.detach()).mean()
    a_loss = a_loss + 1e-2 * real_log_probs.mean()

    sel_a_opt.zero_grad()
    a_loss.backward()
    torch.nn.utils.clip_grad_norm_(sel_actor.parameters(), 100.0)
    sel_a_opt.step()

    for ps, pf in zip(sel_slow.parameters(), sel_critic.parameters()):
        ps.data.lerp_(pf.data, 0.005)

    return {"actor_loss": a_loss.item(), "critic_loss": c_loss.item(),
            "real_actor_loss": a_loss.item(), "real_critic_loss": c_loss.item(),
            "real_reward_mean": rew_all.mean().item(),
            "real_reward_max": rew_all.max().item()}


# ═══════════════════════════════════════════════════════════════════════════
#  Trajectory Selection
# ═══════════════════════════════════════════════════════════════════════════

def compute_traffic_density(traj, data_dir, window=50):
    import pandas as pd
    rec_id, tid = traj["rid"], traj["tid"]
    merge_idx = traj.get("merge_idx", -1)
    if merge_idx < 0:
        return 0.0
    try:
        tracks_csv = pd.read_csv(
            os.path.join(data_dir, f"{rec_id:02d}_tracks.csv"), low_memory=False)
    except Exception:
        return 0.0
    ego_sub = tracks_csv[tracks_csv["trackId"] == int(tid)].sort_values("frame")
    if len(ego_sub) == 0:
        return 0.0
    f0, f1 = int(ego_sub["frame"].iloc[0]), int(ego_sub["frame"].iloc[-1])
    mf = f0 + merge_idx
    ws, we = max(f0, mf - window), min(f1, mf + window)
    if we <= ws:
        return 0.0
    ego_win = ego_sub[(ego_sub["frame"] >= ws) & (ego_sub["frame"] <= we)]
    ego_h = ego_win["heading"].mean()
    win_data = tracks_csv[(tracks_csv["frame"] >= ws) & (tracks_csv["frame"] <= we)]
    other = win_data[win_data["trackId"] != int(tid)]
    ego_rad = math.radians(ego_h)
    ego_dir = np.array([math.cos(ego_rad), math.sin(ego_rad)])
    counts = []
    for frame in range(ws, we + 1):
        ft = other[other["frame"] == frame]
        c = 0
        for _, row in ft.iterrows():
            h = math.radians(row["heading"])
            if np.dot(ego_dir, np.array([math.cos(h), math.sin(h)])) > 0.5:
                c += 1
        counts.append(c)
    return float(np.mean(counts)) if counts else 0.0


def select_trajectories(merge_cache_path, data_dir, train_locs, max_per_loc=20):
    with open(merge_cache_path) as f:
        all_items = json.load(f)
    selected = []
    for loc_id in train_locs:
        items = all_items.get(str(loc_id), [])
        if not items:
            continue
        print(f"  [loc {loc_id}] Scoring {len(items)} trajectories...")
        scored = [(compute_traffic_density(it, data_dir), it) for it in items]
        scored.sort(key=lambda x: x[0])
        n = len(scored)
        per_bin = max(1, (max_per_loc or n) // 3)
        bin_size = n // 3
        bins = [scored[:bin_size], scored[bin_size:2*bin_size], scored[2*bin_size:]]
        rng = np.random.RandomState(42)
        for bn, bd in zip(["low", "mid", "high"], bins):
            if not bd:
                continue
            n_sel = min(per_bin, len(bd))
            for idx in rng.choice(len(bd), n_sel, replace=False):
                selected.append(bd[idx][1])
        print(f"    density: {scored[0][0]:.1f}-{scored[-1][0]:.1f} | "
              f"selected {min(max_per_loc, len(scored))}")
    print(f"  Total: {len(selected)} trajectories")
    return selected


def select_eval_trajectories(merge_cache_path, eval_locs, max_per_loc=5):
    """Fast eval trajectory selection (random, no CSV density scan)."""
    with open(merge_cache_path) as f:
        all_items = json.load(f)
    selected = []
    rng = np.random.RandomState(123)
    for loc_id in eval_locs:
        items = all_items.get(str(loc_id), [])
        if not items:
            continue
        n_sel = min(max_per_loc, len(items))
        for idx in rng.choice(len(items), n_sel, replace=False):
            selected.append(items[idx])
        print(f"  [loc {loc_id}] Random selected {n_sel}/{len(items)} trajectories")
    return selected


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Online DreamerV3 v2")
    parser.add_argument("--wm-ckpt", default="")
    parser.add_argument("--ac-ckpt", default=None,
                        help="Deprecated: offline BC actor. Use --actor-init gt_buffer instead.")
    parser.add_argument("--actor-init", default="gt_buffer",
                        choices=["scratch", "gt_buffer"],
                        help="scratch=random actor; gt_buffer=imitate PID actions after GT warmup")
    parser.add_argument("--gt-actor-init-steps", type=int, default=200,
                        help="Gradient steps for online GT-buffer actor init (per track)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--npz-dir", default=None)
    parser.add_argument("--train-locs", type=int, nargs="+", default=[0, 2, 4, 5, 6])
    parser.add_argument("--eval-locs", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--max-traj-per-loc", type=int, default=20)
    parser.add_argument("--selection-file", default=None,
                        help="Pre-computed selection JSON (skip CSV density scan)")
    parser.add_argument("--total-episodes", type=int, default=200)
    parser.add_argument("--train-ratio", type=int, default=32,
                        help="Training steps per batch_steps env steps (DreamerV3 default: 32)")
    parser.add_argument("--policy-steps", type=int, default=10,
                        help="Env steps per interaction chunk before training")
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-same-track", action="store_true",
                        help="Eval on the same tracks as training (no separate eval selection)")
    parser.add_argument("--video-dir", default="./online_videos")
    parser.add_argument("--video-every", type=int, default=10,
                        help="Save training mp4 every N completed episodes (0=disable)")
    parser.add_argument("--save-gt-video", action="store_true",
                        help="Save one GT-warmup (PID) reference mp4 per track as gt_{name}.mp4")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training (reduce for L40: 8)")
    parser.add_argument("--batch-length", type=int, default=50,
                        help="Sequence length for training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logdir", default="./logs/online_dreamer_v2")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--wandb-run-name", default="online_dreamer_v2")
    parser.add_argument("--wm-warmup-episodes", type=int, default=0,
                        help="First N episodes: WM-only training (no AC) to calibrate reward head")
    parser.add_argument("--gt-warmup-episodes", type=int, default=50,
                        help="First N episodes: collect with GT actions to seed buffer with successes")
    parser.add_argument("--curriculum-episodes", type=int, default=100,
                        help="After GT warmup, N episodes of GT/actor mixing, GT ratio anneals 1→0")
    parser.add_argument("--explore-std", type=float, default=1.0,
                        help="Actor log_std init for online exploration (GT-buffer init, no offline BC)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train WM+AC from random init (no pretrained checkpoint)")
    parser.add_argument("--failure-reflection", action="store_true",
                        help="Self-evolving agent: replay pre-crash states in imagination")
    parser.add_argument("--success-sample-ratio", type=float, default=0.9,
                        help="Fraction of AC/WM batches sampled from protected (success/GT) buffer")
    parser.add_argument("--phase-actors", action="store_true",
                        help="Phase-staged MoE: 3 actor-critic pairs with soft blending via phase_head")
    parser.add_argument("--episodes-per-track", type=int, default=30,
                        help="Repeat each training track this many times (standard RL: same env × N)")
    parser.add_argument("--gt-warmup-per-track", type=int, default=5,
                        help="GT warmup episodes per track (first N reps of each track)")
    parser.add_argument("--curriculum-per-track", type=int, default=5,
                        help="Curriculum episodes per track (GT→actor mixing, after warmup)")
    parser.add_argument("--gt-video-only", action="store_true",
                        help="Only run GT-warmup reps and save ep001..epN videos (like videos2); then exit")
    parser.add_argument("--no-residual-action", action="store_true",
                        help="Disable ResWM residual actions (action = tanh(Δ), not tanh(prev_a + Δ))")
    parser.add_argument("--no-action-bonus", action="store_true",
                        help="Use pure phase-progress reward in imagination (no action guidance)")
    parser.add_argument("--freeze-wm", action="store_true",
                        help="Freeze WM during RL (only train AC, keep continue/phase heads calibrated)")
    parser.add_argument("--bc-anchor-weight", type=float, default=0.05,
                        help="Weight of BC anchor loss (0=disabled)")
    parser.add_argument("--gt-bc-weight", type=float, default=0.3,
                        help="Trajectory-level GT BC loss weight -- anchors actor to PID actions "
                             "from protected buffer with phase-aware weighting (0=disabled)")
    args = parser.parse_args()

    if args.gt_video_only:
        args.save_gt_video = False
        args.actor_init = "scratch"
        args.video_every = 1
        args.eval_interval = 10 ** 9
        args.curriculum_per_track = 0
        if args.total_episodes > 100:
            args.total_episodes = args.gt_warmup_per_track
        print(f"[gt-video-only] total_episodes={args.total_episodes} "
              f"gt_warmup={args.gt_warmup_per_track} video_every=1 save_gt_video=OFF")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    from_scratch = args.from_scratch or not args.wm_ckpt
    residual_action = not args.no_residual_action
    print(f"ResWM residual actions: {residual_action}")
    if from_scratch:
        if not args.from_scratch:
            print("No --wm-ckpt provided, initializing from scratch")
        wm, actor, critic, slow_critic, wm_opt, a_opt, c_opt, cfg = \
            load_models_from_scratch(device, explore_std=args.explore_std,
                                     residual_action=residual_action)
        # Disable phase_head in imagination (no pretrained classifier)
        cfg.use_phase_head = False
    else:
        if args.ac_ckpt:
            print("WARNING: --ac-ckpt ignored (offline BC removed). Use --actor-init gt_buffer.")
        wm, actor, critic, slow_critic, wm_opt, a_opt, c_opt, cfg = \
            load_models_for_online(args.wm_ckpt, None, device,
                                   explore_std=args.explore_std,
                                   phase_actors=args.phase_actors,
                                   residual_action=residual_action)

    # ── Freeze WM (--freeze-wm): only train AC, keep continue/phase heads calibrated ──
    if args.freeze_wm:
        for p in wm.parameters():
            p.requires_grad = False
        wm.eval()
        print("WM frozen: only AC trained (continue/phase/reward heads unchanged)")

    # Replay buffer
    obs_shape = (cfg.input_channels, cfg.bev_size, cfg.bev_size)
    buffer = ReplayBuffer(cfg.online_buffer_capacity, obs_shape, cfg.action_dim)

    # Failure reflection buffer (self-evolving agent)
    failure_buffer = FailureBuffer(capacity=500) if args.failure_reflection else None
    if failure_buffer:
        print("Failure Reflection: enabled (50% imagination from pre-crash states)")

    # Online crash detector: binary classifier on RSSM features
    feat_dim = wm.feature_dim()
    crash_detector = CrashDetector(feat_dim).to(device)
    crash_buffer = CrashFeatureBuffer(max_samples=10000)
    crash_opt = torch.optim.Adam(crash_detector.parameters(), lr=1e-4, eps=1e-5)
    crash_ready = False
    bce_loss = torch.nn.BCELoss()

    print(f"  Crash detector: {feat_dim}d features, trained online when buffer ready")
    print(f"  Success replay: {args.success_sample_ratio:.0%} batches from protected buffer")

    # Override batch config from CLI
    cfg.batch_size = args.batch_size
    cfg.batch_length = args.batch_length
    merge_cache = os.path.join(os.path.dirname(__file__), "../mirro_data_map/exid_merge_cache.json")

    # Select trajectories (from pre-computed JSON or on-the-fly density scan)
    if args.selection_file and os.path.exists(args.selection_file):
        print(f"Loading pre-computed selection: {args.selection_file}")
        with open(args.selection_file) as f:
            sel_data = json.load(f)
        selected = []
        for loc_id in args.train_locs:
            items = sel_data.get(str(loc_id), [])
            selected.extend(items)
            print(f"  Loc {loc_id}: {len(items)} trajectories")
        print(f"  Total: {len(selected)} trajectories (pre-computed)")
    else:
        selected = select_trajectories(merge_cache, args.data_dir, args.train_locs,
                                       args.max_traj_per_loc)
    if not selected:
        print("ERROR: No trajectories!")
        sys.exit(1)

    # Group by location
    traj_by_loc = defaultdict(list)
    for t in selected:
        traj_by_loc[t["loc_id"]].append(t)

    # Build maps
    print("Building maps...")
    map_cache = {}
    for loc_id in sorted(traj_by_loc.keys()):
        map_cache[loc_id] = build_map_features(loc_id)
        print(f"  loc {loc_id} ({LOC_NAMES[loc_id]})")

    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    # WandB
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(project="online-dreamer-v2", name=args.wandb_run_name,
                               config=cfg.to_dict(), dir=args.logdir)
    except Exception:
        pass

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── DreamerV3 training loop ──
    # Original DreamerV3 pattern: while step < total: driver(policy, steps=10) → train
    # We adapt to single-env, scenario-based collection with fractional training accumulation
    policy_steps = args.policy_steps          # env steps per interaction chunk
    batch_steps = cfg.batch_size * cfg.batch_length   # steps per batch
    train_ratio = args.train_ratio             # DreamerV3 default: 32
    train_per_env_step = train_ratio / batch_steps    # fractional training credit per env step
    train_accum = 0.0                    # fractional training accumulator

    print(f"\n{'='*60}")
    print(f"Online DreamerV3 -- per-track repeated RL (standard online RL paradigm)")
    print(f"  Per track: GT warmup {args.gt_warmup_per_track} → "
          f"curriculum {args.curriculum_per_track} (GT blend 1→0) → RL")
    print(f"  Each track repeated {args.episodes_per_track}× ({len(selected)} tracks)")
    print(f"  Interact: {policy_steps} env steps → train (ratio={train_ratio})")
    if args.freeze_wm:
        print(f"  WM: FROZEN (no training, only AC learns)")
    else:
        print(f"  WM: online-only (reward/continue/phase heads calibrate on real data)")
    print(f"  AC: imagination RL + 经验回放 (real env rewards from buffer)")
    print(f"  BC anchor weight: {args.bc_anchor_weight} (0=disabled)")
    print(f"  Success sample ratio: {args.success_sample_ratio:.0%}")
    print(f"  Success: merge zone cleared (step >= merge_idx+{MERGE_ZONE_FRAMES}, no crash)")
    print(f"  Reward: phase-aware, scaled [-10, 5]")
    if args.video_every > 0:
        print(f"  Video: every {args.video_every} episodes → {args.video_dir}")
    else:
        print(f"  Video: disabled (eval still saves if enabled)")
    if args.save_gt_video:
        print(f"  GT reference: one PID-warmup mp4 per track → gt_{{name}}.mp4")
    print(f"  Train: {args.train_locs} ({len(selected)} traj)")
    if args.eval_same_track:
        print(f"  Eval: same tracks as training, every {args.eval_interval} episodes")
    else:
        print(f"  Eval: {args.eval_locs} every {args.eval_interval} episodes")
    print(f"{'='*60}")

    env = None
    total_episodes = 0
    total_env_steps = 0
    total_train_steps = 0
    best_eval_survival = 0
    t_start = time.time()

    # ── Build repeat schedule: each track repeated many times ──
    # Standard online RL: reset same environment hundreds of times.
    # We select low/mid/high traffic tracks per location and repeat each one
    # in blocks, with per-track GT warmup → curriculum → pure RL.
    eps_per_track = args.episodes_per_track
    track_schedule = []  # [(track_dict, track_id), ...]
    for t in selected:
        for _ in range(eps_per_track):
            track_schedule.append(t)
    # Block-structure: all episodes of same track are contiguous
    # (selected is already ordered by loc, then density bin)
    # Shuffle blocks for some diversity
    np.random.shuffle(track_schedule)  # interleave tracks a bit
    print(f"\n  Training tracks: {len(selected)} × {eps_per_track} reps = {len(track_schedule)} episodes")
    print(f"  Per-track: GT_warmup={args.gt_warmup_per_track} curriculum={args.curriculum_per_track}")

    # Track-level episode counter
    track_ep_count = defaultdict(int)  # key: track_id → num episodes completed
    current_track_id = None

    ep_state = None
    saved_gt_videos = set()  # track names that already have gt_{name}.mp4
    gt_actor_inited_tracks = set()  # per-track online GT-buffer actor init
    while total_episodes < args.total_episodes:
        if args.gt_video_only and total_episodes >= args.gt_warmup_per_track:
            print(f"[gt-video-only] finished {total_episodes} GT episodes, exit.")
            break
        # ── Pick next track (repeated) ─────────────────────────────────
        # Track changes only when the current track has had enough episodes
        # or at the start of a new training block
        if ep_state is None:
            # Determine next track
            schedule_idx = total_episodes % len(track_schedule)
            traj = track_schedule[schedule_idx]
            traj_id = f"loc{traj['loc_id']}_rec{traj['rid']}_t{traj['tid']}"

            # Check if this is a new track (track boundary)
            new_track = (traj_id != current_track_id)
            if new_track:
                current_track_id = traj_id
                track_ep_count[traj_id] = 0
                print(f"\n--- New track: {traj_id} (ep {total_episodes}) ---")

            track_ep = track_ep_count[traj_id]  # how many times this track has been run

            # ── Load scenario (reset env every episode) ─────────────────
            loc_id = traj["loc_id"]
            rec_id = traj["rid"]
            tid = traj["tid"]
            merge_idx = traj.get("merge_idx", -1)
            ep_name = f"loc{loc_id}_rec{rec_id}_t{tid}"

            try:
                features_raw, off_x, off_y = map_cache[loc_id]
                features = copy.deepcopy(features_raw)  # prevent centralize_to_ego mutation
                scenario_dict, t_len, (f0, f1) = build_scenario_dict(
                    rec_id, tid, loc_id, features, off_x, off_y, args.data_dir)

                import pandas as pd
                tracks_csv = pd.read_csv(
                    os.path.join(args.data_dir, f"{rec_id:02d}_tracks.csv"), low_memory=False)
                ego_sub = tracks_csv[tracks_csv["trackId"] == int(tid)].sort_values("frame")
                ego_seg = ego_sub[(ego_sub["frame"] >= f0) & (ego_sub["frame"] <= f1)]
                gt_speeds = np.sqrt(ego_seg["xVelocity"].values**2 + ego_seg["yVelocity"].values**2)
                if len(gt_speeds) < t_len:
                    gt_speeds = np.pad(gt_speeds, (0, t_len - len(gt_speeds)), 'edge')
                elif len(gt_speeds) > t_len:
                    gt_speeds = gt_speeds[:t_len]

                # GT trajectory for navigation guidance
                gt_positions = ego_seg[["xCenter", "yCenter"]].values.astype(np.float64)
                gt_headings = ego_seg["heading"].values.astype(np.float64)
                if len(gt_positions) < t_len:
                    gt_positions = np.pad(gt_positions, ((0, t_len - len(gt_positions)), (0, 0)), 'edge')
                    gt_headings = np.pad(gt_headings, (0, t_len - len(gt_headings)), 'edge')
                elif len(gt_positions) > t_len:
                    gt_positions = gt_positions[:t_len]
                    gt_headings = gt_headings[:t_len]
                gt_traj = {"pos": gt_positions, "heading": gt_headings}

                ego_valid = int(scenario_dict["tracks"][str(tid)]["state"]["valid"].sum())
                print(f"[scenario] {ep_name} t_len={t_len} frames=({f0},{f1}) "
                      f"ego_valid={ego_valid} merge_idx={merge_idx}", flush=True)

                scenario = ScenarioDescription(scenario_dict)

                # Fresh env per scenario
                try:
                    if env is not None:
                        env.close()
                finally:
                    env = None
                    force_reset_engine()
                gt_blend = curriculum_gt_blend(
                    track_ep, args.gt_warmup_per_track, args.curriculum_per_track)
                setup_wm_policy(
                    wm, actor, bev_size=cfg.bev_size, device=device,
                    gt_speeds=gt_speeds if gt_blend > 0.0 else None,
                    gt_blend=gt_blend)
                if track_ep < args.gt_warmup_per_track + args.curriculum_per_track:
                    phase = "GT" if track_ep < args.gt_warmup_per_track else "CURR"
                    print(f"  [{phase}] track_ep={track_ep} gt_blend={gt_blend:.2f}", flush=True)
                env = ScenarioOnlineEnv(config=dict(
                    use_render=False, image_observation=True,
                    agent_policy=WorldModelPolicy,
                    horizon=t_len + 200, store_map=False, set_static=True,
                    camera_smooth=False,
                    decision_repeat=2,  # 2*0.02s=0.04s=25fps, matches exiD data rate
                    vehicle_config=dict(no_wheel_friction=False, show_navi_mark=False,
                                        image_source="rgb_camera"),
                    sensors={"rgb_camera": (RGBCamera, BEV_W, BEV_H)},
                    norm_pixel=False, height_scale=0.01,
                ))
                # Highway: DefaultVehicle max_engine_force=800N, max_speed_km_h=80
                # (22.2 m/s) tops at ~23m/s. Increase to match exiD highway speeds
                # (up to 34 m/s = 122 km/h).
                env.config["vehicle_config"].update(
                    {"max_engine_force": 3000.0, "max_brake_force": 600.0,
                     "max_speed_km_h": 130.0},
                    allow_add_new_key=True)
                scenario_dict["metadata"]["location_id"] = loc_id
                env.set_scenario(scenario)
                env.reset()
                ego = env.agent
                if ego is not None:
                    p = ego.position
                    ego.set_position([float(p[0]), float(p[1])], height=float(ego.HEIGHT)/2)

                ep_state = {
                    "traj": traj, "name": ep_name, "t_len": t_len,
                    "merge_idx": merge_idx,
                    "step": 0, "prev_speed": None,
                    "ep_reward": 0.0, "frames": [],
                    "done": False, "end_reason": "timeout",
                    "track_ep": track_ep,  # how many times this track has been run
                    "buf_start": buffer._ptr,  # for success buffer protection
                    "gt_traj": gt_traj,  # GT human trajectory for navigation guidance
                }
            except Exception as e:
                print(f"[setup] {ep_name} ERROR: {e}")
                import traceback
                traceback.print_exc()
                ep_state = None  # retry with next scenario
                total_episodes += 1  # skip broken track (avoids infinite retry loop)
                continue

        # ── Interact: collect policy_steps env steps ─────────────────────
        # Per-track: PID warmup → curriculum (actor, AC on) → pure RL
        track_ep = ep_state["track_ep"]
        pid_warmup = track_ep < args.gt_warmup_per_track
        in_curriculum = (args.gt_warmup_per_track <= track_ep
                         < args.gt_warmup_per_track + args.curriculum_per_track)

        # GT reference: record entire first GT-warmup ep on this track (PID drives sim)
        save_gt_ep = (args.save_gt_video and track_ep == 0
                      and ep_state["name"] not in saved_gt_videos)
        save_frames = save_gt_ep or (
            args.video_every > 0
            and (total_episodes + 1) % args.video_every == 0)
        # Debug: per-step logging for first warmup episode of each track
        debug_steps = (pid_warmup and track_ep == 0)
        chunk = run_collect_chunk(
            env, ep_state["t_len"], ep_state["merge_idx"], buffer,
            bev_size=cfg.bev_size,
            save_frames=save_frames, start_step=ep_state["step"],
            prev_speed=ep_state["prev_speed"], max_steps=policy_steps,
            pid_warmup=pid_warmup,
            track_failure=(failure_buffer is not None),
            debug_per_step=debug_steps,
            gt_traj=ep_state.get("gt_traj"),
        )

        ep_state["step"] += chunk["steps"]
        ep_state["prev_speed"] = chunk["prev_speed"]
        ep_state["ep_reward"] += chunk["reward"]
        ep_state["frames"].extend(chunk.get("frames") or [])
        total_env_steps += chunk["steps"]

        # ── Episode boundary ────────────────────────────────────────────
        if chunk["done"]:
            ep_state["done"] = True
            ep_state["end_reason"] = chunk["end_reason"]
            total_episodes += 1

            # GT reference video (PID warmup -- not RL actor)
            if save_gt_ep and ep_state["frames"]:
                gt_path = os.path.join(args.video_dir, f"gt_{ep_state['name']}.mp4")
                save_video(ep_state["frames"], gt_path)
                saved_gt_videos.add(ep_state["name"])
            # Periodic RL/curriculum video (ep 10, 20, ...)
            elif ep_state["frames"] and not save_gt_ep:
                vid_path = os.path.join(args.video_dir,
                                        f"ep{total_episodes:03d}_{ep_state['name']}.mp4")
                save_video(ep_state["frames"], vid_path)

            # Increment track episode counter
            track_ep_count[ep_state["name"]] = track_ep + 1

            # Log episode (per-track phase tag)
            if pid_warmup:
                etag = "GT"
            elif in_curriculum:
                etag = "CURR"
            else:
                etag = "RL"
            act_m = chunk.get("act_mean", 0)
            act_s = chunk.get("act_std", 0)
            thr_m = chunk.get("thr_mean", 0)
            thr_s = chunk.get("thr_std", 0)
            # Format merge-window actions
            ma = chunk.get("merge_actions", [])
            ma_str = ""
            if ma:
                ma_compact = []
                for step, steer, throttle, is_act in ma:
                    tag = "A" if is_act else "G"
                    ma_compact.append(f"{step}:{tag}{steer:+.3f}/{throttle:+.2f}")
                ma_str = " | merge: " + " ".join(ma_compact[-8:])  # last 8 (closest)
            merged_ok = chunk.get("merged", False)
            print(f"[ep{total_episodes}][{etag}] {ep_state['name']}#{track_ep+1} "
                  f"R={ep_state['ep_reward']:.1f} steps={ep_state['step']} "
                  f"end={ep_state['end_reason']} merge={'OK' if merged_ok else 'FAIL'} "
                  f"act=[{act_m:+.3f}±{act_s:.3f}] thr=[{thr_m:+.3f}±{thr_s:.3f}]"
                  f"{ma_str} | "
                  f"buf={len(buffer)} env_steps={total_env_steps}")

            # Protect GT-warmup trajectories and merge-OK actor episodes
            buf_end = buffer._ptr
            protect_ep = False
            if pid_warmup and ep_state["step"] >= 200:
                protect_ep = True
            elif merged_ok and not pid_warmup:
                protect_ep = True
            if protect_ep:
                buffer.protect_last_episode(ep_state["buf_start"], buf_end)
                if total_episodes % 25 == 0 or pid_warmup:
                    print(f"  [buffer] protected ep{total_episodes} "
                          f"({ep_state['step']} steps, prot={buffer.get_protected_ratio():.1%})",
                          flush=True)

            # Online GT-buffer actor init once per track (after last GT warmup ep)
            if (not args.gt_video_only
                    and args.actor_init == "gt_buffer"
                    and pid_warmup
                    and track_ep + 1 == args.gt_warmup_per_track
                    and ep_state["name"] not in gt_actor_inited_tracks):
                init_actor_from_online_buffer(
                    wm, actor, buffer, cfg, device, n_steps=args.gt_actor_init_steps,
                    prefer_protected=min(0.95, args.success_sample_ratio + 0.5))
                gt_actor_inited_tracks.add(ep_state["name"])

            if wandb_run:
                wandb_run.log({
                    "episode": total_episodes,
                    "ep_reward": ep_state["ep_reward"],
                    "ep_steps": ep_state["step"],
                    "end_reason": ep_state["end_reason"],
                    "buffer_size": len(buffer),
                    "env_steps": total_env_steps,
                }, step=total_episodes)

            # Failure reflection: store pre-crash window for prioritized rehearsal
            if failure_buffer and chunk.get("failure_data"):
                failure_buffer.push_window(*chunk["failure_data"])

            # Crash detector: collect RSSM features from this episode
            ep_features = get_global_feature_buffer()
            if ep_features:
                end_reason = ep_state["end_reason"]
                if end_reason in ("crash", "crash_vehicle", "out_of_road"):
                    k = min(20, len(ep_features))
                    crash_buffer.add_crash(ep_features[-k:])
                if len(ep_features) > 10:
                    n_normal = min(50, len(ep_features) // 2)
                    idx = torch.randperm(len(ep_features))[:n_normal]
                    crash_buffer.add_normal([ep_features[i] for i in idx])
                if crash_buffer.ready() and total_episodes % 5 == 0:
                    feats, labels = crash_buffer.sample_batch(64, device)
                    preds = crash_detector(feats)
                    loss = bce_loss(preds, labels)
                    crash_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(crash_detector.parameters(), 10.0)
                    crash_opt.step()
                    crash_ready = True
                    if total_episodes % 25 == 0:
                        acc = ((preds > 0.5).float() == labels).float().mean()
                        print(f"  [crash_det] trained loss={loss.item():.4f} acc={acc.item():.3f} "
                              f"buf={crash_buffer.size()}", flush=True)

            ep_state = None  # triggers next scenario load

        # ── Train: DreamerV3 interleaved with fractional accumulation ───
        if len(buffer) >= cfg.batch_length:
            train_accum += chunk["steps"] * train_per_env_step
            train_count = 0
            wm_losses, ac_losses = [], []

            while train_accum >= 1.0:
                # WM: online-only (skip if --freeze-wm to keep continue/phase heads calibrated)
                if not args.freeze_wm:
                    wm_losses.append(train_wm_online(
                        wm, buffer, wm_opt, cfg, device,
                        success_sample_ratio=args.success_sample_ratio))

                # AC: skip during GT warmup AND WM-only warmup (let WM calibrate first)
                in_gt_warmup = (track_ep < args.gt_warmup_per_track)
                in_wm_warmup = (total_episodes < args.wm_warmup_episodes)
                if not in_gt_warmup and not in_wm_warmup:
                    ac_losses.append(train_ac_imagination(
                        wm, actor, critic, slow_critic,
                        a_opt, c_opt, buffer, cfg, device,
                        failure_buffer=failure_buffer,
                        phase_actors=args.phase_actors,
                        success_sample_ratio=args.success_sample_ratio,
                        bc_anchor_weight=args.bc_anchor_weight,
                        gt_bc_weight=args.gt_bc_weight,
                        crash_detector=crash_detector, crash_ready=crash_ready))
                    # 经验回放: train AC on real buffer data with stored env rewards
                    if len(buffer) >= cfg.batch_length * 2:
                        ac_losses.append(train_ac_real(
                            wm, actor, critic, slow_critic,
                            a_opt, c_opt, buffer, cfg, device,
                            phase_actors=args.phase_actors))
                train_count += 1
                train_accum -= 1.0

            if train_count > 0:
                total_train_steps += train_count
                reg_key = f"loss_{cfg.regularization}"
                reg_val = np.mean([m.get(reg_key, m.get("loss_dyn", 0)) for m in wm_losses])
                rec_val = np.mean([m.get("loss_barlow", m.get("loss_rec", 0)) for m in wm_losses])
                a_loss = np.mean([m["actor_loss"] for m in ac_losses]) if ac_losses else 0
                avg_steer_std = np.mean([m.get("act_steer_std", 0) for m in ac_losses])
                avg_rew = np.mean([m.get("imag_reward_mean", 0) for m in ac_losses])
                avg_p0 = np.mean([m.get("p0_bonus_mean", 0) for m in ac_losses])
                avg_p1 = np.mean([m.get("p1_bonus_mean", 0) for m in ac_losses])

                # Print training progress (also to stdout for HPC logs)
                if total_train_steps % 10 == 1 or total_train_steps <= 5:
                    if pid_warmup:
                        phase_tag = " [GT-WARMUP]"
                    elif in_curriculum:
                        phase_tag = " [CURRICULUM]"
                    else:
                        phase_tag = " [RL]"
                    print(f"  [train {total_train_steps}]{phase_tag} barlow={rec_val:.5f} {reg_key}={reg_val:.5f} "
                          f"a={a_loss:.2f} s_std={avg_steer_std:.3f} iR={avg_rew:.3f} "
                          f"p0={avg_p0:.3f} p1={avg_p1:.3f} buf={len(buffer)}", flush=True)

                if total_episodes > 0 and wandb_run:
                    wandb_run.log({
                        "train_steps": total_train_steps,
                        "wm/barlow": rec_val, f"wm/{reg_key}": reg_val,
                        "ac/actor_loss": a_loss,
                    }, step=total_episodes)

        # ── Eval ────────────────────────────────────────────────────────
        if total_episodes > 0 and total_episodes % args.eval_interval == 0:
            # Only eval once per interval (guard against calling eval on
            # consecutive chunks when an interval boundary is crossed mid-episode)
            last_eval_ep = int(getattr(main, "_last_eval_ep", -1))
            if total_episodes != last_eval_ep:
                main._last_eval_ep = total_episodes
                print(f"\n--- Eval @ ep {total_episodes} (env_steps={total_env_steps}) ---")
                wm.eval()
                actor.eval()

                eval_results = []
                if args.eval_same_track:
                    e_trajs_all = selected  # eval on same tracks as training
                else:
                    e_trajs_all = select_eval_trajectories(
                        merge_cache, args.eval_locs, max_per_loc=args.eval_episodes)
                eval_locs = sorted(set(t["loc_id"] for t in e_trajs_all))
                for eval_loc in eval_locs:
                    e_features_raw, e_ox, e_oy = build_map_features(eval_loc)
                    e_trajs = [t for t in e_trajs_all if int(t.get("loc_id", -1)) == eval_loc]
                    for et in e_trajs[:args.eval_episodes]:
                        try:
                            e_features = copy.deepcopy(e_features_raw)
                            es_dict, et_len, (ef0, ef1) = build_scenario_dict(
                                et["rid"], et["tid"], eval_loc,
                                e_features, e_ox, e_oy, args.data_dir)
                            es_dict["metadata"]["location_id"] = eval_loc
                            e_scenario = ScenarioDescription(es_dict)
                            try:
                                if env is not None:
                                    env.close()
                            finally:
                                env = None
                                force_reset_engine()
                            setup_wm_policy(wm, actor, bev_size=cfg.bev_size, device=device)
                            env = ScenarioOnlineEnv(config=dict(
                                use_render=False, image_observation=True,
                                agent_policy=WorldModelPolicy,
                                horizon=et_len + 200,
                                store_map=False, set_static=True,
                                camera_smooth=False,
                                decision_repeat=2,  # 25fps
                                vehicle_config=dict(no_wheel_friction=False,
                                                    show_navi_mark=False,
                                                    image_source="rgb_camera"),
                                sensors={"rgb_camera": (RGBCamera, BEV_W, BEV_H)},
                                norm_pixel=False, height_scale=0.01,
                            ))
                            env.config["vehicle_config"].update(
                                {"max_engine_force": 3000.0, "max_brake_force": 600.0,
                                 "max_speed_km_h": 130.0},
                                allow_add_new_key=True)
                            env.set_scenario(e_scenario)
                            er = run_eval_episode(
                                env, et_len, et.get("merge_idx", -1),
                                bev_size=cfg.bev_size, save_frames=True)
                            er["loc_id"] = eval_loc
                            er["name"] = f"loc{eval_loc}_rec{et['rid']}_t{et['tid']}"
                            eval_results.append(er)

                            if er.get("frames"):
                                evid_path = os.path.join(
                                    args.video_dir, f"eval_{er['name']}.mp4")
                                save_video(er["frames"], evid_path)
                        except Exception:
                            import traceback
                            traceback.print_exc()
                            if env is not None:
                                try:
                                    env.close()
                                except Exception:
                                    pass
                                env = None
                                force_reset_engine()
                            continue

                # Close last eval env
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = None
                    force_reset_engine()

                if eval_results:
                    n = len(eval_results)
                    surv = sum(1 for r in eval_results if r["survived"])
                    avg_r = np.mean([r["reward"] for r in eval_results])
                    print(f"  Eval: survival={surv}/{n} ({surv/n*100:.1f}%) avgR={avg_r:.1f}")
                    for er in eval_results:
                        print(f"    {er['name']}: {er['end_reason']} R={er['reward']:.1f}")

                    # Track merge success (reached post-merge) in addition to survival
                    merge_ok = sum(1 for r in eval_results if r.get("merged", False))
                    merge_rate = merge_ok / n
                    print(f"  Merge zone cleared: {merge_ok}/{n} ({merge_rate*100:.1f}%)")

                    if merge_rate > getattr(main, "_best_eval_merge", 0.0):
                        main._best_eval_merge = merge_rate
                        ckpt_data = {
                            "world_model": wm.state_dict(),
                            "actor": actor.state_dict(),
                            "critic": critic[0].state_dict() if args.phase_actors else critic.state_dict(),
                            "total_episodes": total_episodes,
                            "eval_merge_rate": merge_rate,
                            "config": cfg.to_dict(),
                        }
                        if args.phase_actors:
                            for i, a in enumerate(actor.actors):
                                ckpt_data[f"actor_{i}"] = a.state_dict()
                            for i, c in enumerate(critic):
                                ckpt_data[f"critic_{i}"] = c.state_dict()
                        torch.save(ckpt_data, os.path.join(args.logdir, "checkpoint_merge_best.pt"))
                        print(f"  Saved merge_best (merge_rate={merge_rate:.3f})")

                    if surv / n > best_eval_survival:
                        best_eval_survival = surv / n
                        ckpt_data = {
                            "world_model": wm.state_dict(),
                            "actor": actor.state_dict(),
                            "critic": critic[0].state_dict() if args.phase_actors else critic.state_dict(),
                            "total_episodes": total_episodes,
                            "eval_survival": best_eval_survival,
                            "config": cfg.to_dict(),
                        }
                        if args.phase_actors:
                            for i, a in enumerate(actor.actors):
                                ckpt_data[f"actor_{i}"] = a.state_dict()
                            for i, c in enumerate(critic):
                                ckpt_data[f"critic_{i}"] = c.state_dict()
                        torch.save(ckpt_data, os.path.join(args.logdir, "checkpoint_best.pt"))
                        print(f"  Saved best (survival={best_eval_survival:.3f})")

                    if wandb_run:
                        wandb_run.log({
                            "eval/survival": surv / n,
                            "eval/merge_rate": merge_rate,
                            "eval/avg_reward": avg_r,
                        }, step=total_episodes)

                # Recreate env for next training episode (eval closed it)
                env = None
                ep_state = None  # force fresh scenario load
                wm.train()
                actor.train()

        # ── Checkpoint ──────────────────────────────────────────────────
        if total_episodes > 0 and total_episodes % args.save_every == 0:
            last_save_ep = int(getattr(main, "_last_save_ep", -1))
            if total_episodes != last_save_ep:
                main._last_save_ep = total_episodes
                ckpt_data = {
                    "world_model": wm.state_dict(), "actor": actor.state_dict(),
                    "critic": critic[0].state_dict() if args.phase_actors else critic.state_dict(),
                    "total_episodes": total_episodes,
                    "total_env_steps": total_env_steps,
                    "config": cfg.to_dict(),
                }
                if args.phase_actors:
                    for i, a in enumerate(actor.actors):
                        ckpt_data[f"actor_{i}"] = a.state_dict()
                    for i, c in enumerate(critic):
                        ckpt_data[f"critic_{i}"] = c.state_dict()
                torch.save(ckpt_data, os.path.join(args.logdir, f"checkpoint_ep{total_episodes}.pt"))

    if env is not None:
        env.close()

    ckpt_data_final = {
        "world_model": wm.state_dict(), "actor": actor.state_dict(),
        "critic": critic[0].state_dict() if args.phase_actors else critic.state_dict(),
        "total_episodes": total_episodes,
        "total_env_steps": total_env_steps,
        "config": cfg.to_dict(),
    }
    if args.phase_actors:
        for i, a in enumerate(actor.actors):
            ckpt_data_final[f"actor_{i}"] = a.state_dict()
        for i, c in enumerate(critic):
            ckpt_data_final[f"critic_{i}"] = c.state_dict()
    torch.save(ckpt_data_final, os.path.join(args.logdir, "checkpoint_final.pt"))

    elapsed = time.time() - t_start
    print(f"\nDone! {total_episodes} ep, {total_env_steps} env_steps, "
          f"{total_train_steps} train_steps in {elapsed:.0f}s "
          f"best survival={best_eval_survival:.3f}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
