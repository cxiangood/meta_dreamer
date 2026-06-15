"""
WorldModelPolicy: closed-loop control policy backed by DreamerV3 world model + actor-critic.

Usage:
    from envs.world_model_policy import setup_wm_policy

    # Before creating the env, set up models
    setup_wm_policy(world_model, actor, bev_size, device)

    # Then use the policy class
    env = ScenarioOnlineEnv({"agent_policy": WorldModelPolicy, ...})
"""

import math
import sys

import numpy as np
import torch
from metadrive.policy.base_policy import BasePolicy


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi

# Module-level globals set by setup_wm_policy()
_GLOBAL_WM = None
_GLOBAL_ACTOR = None
_GLOBAL_DEVICE = None
_GLOBAL_BEV_SIZE = 64
_GLOBAL_LAST_STATE = None  # Exposed for DAPO state collection
_GLOBAL_PREV_ACTION = np.zeros(2, dtype=np.float32)  # Exposed for action logging
_GLOBAL_ACT_COUNT = 0  # Debug: count total act() calls
_GLOBAL_GT_SPEEDS = None   # Per-frame GT speeds (m/s) for PID warmup; None = actor control
_GLOBAL_GT_BLEND = 0.0     # 1.0 = always PID; 0.0 = always actor; (0,1) = stochastic mix
_GLOBAL_LAST_IS_PID = False
_GLOBAL_FEATURE_BUFFER = []  # Crash detector: RSSM features collected during episode


def get_global_feature_buffer():
    """Return and clear the feature buffer for crash detector training."""
    global _GLOBAL_FEATURE_BUFFER
    buf = list(_GLOBAL_FEATURE_BUFFER)
    _GLOBAL_FEATURE_BUFFER = []
    return buf


def get_global_last_state():
    """Return the last RSSM state dict (deter, stoch) from WorldModelPolicy.
    Used by DAPO P4 for collecting start states."""
    return _GLOBAL_LAST_STATE


def setup_wm_policy(world_model, actor, bev_size=64, device=None,
                    gt_speeds=None, gt_blend=1.0):
    """Set global models for WorldModelPolicy to use.

    Args:
        gt_speeds: (T,) exiD speed profile (m/s) for PID; None = actor only.
        gt_blend: 1.0 = PID; 0.0 = actor; (0,1) = per-step stochastic curriculum mix.
    """
    global _GLOBAL_WM, _GLOBAL_ACTOR, _GLOBAL_DEVICE, _GLOBAL_BEV_SIZE
    global _GLOBAL_GT_SPEEDS, _GLOBAL_GT_BLEND
    _GLOBAL_WM = world_model
    _GLOBAL_ACTOR = actor
    _GLOBAL_BEV_SIZE = bev_size
    _GLOBAL_DEVICE = device or next(world_model.parameters()).device
    _GLOBAL_GT_SPEEDS = gt_speeds
    _GLOBAL_GT_BLEND = float(gt_blend) if gt_speeds is not None else 0.0


def get_last_control_is_actor():
    """True if the last act() step used the RL actor (not PID)."""
    return not _GLOBAL_LAST_IS_PID


class WorldModelPolicy(BasePolicy):
    """
    Non-replay policy that uses World Model + Actor for closed-loop control.

    Captures BEV from RGBCamera → WM encoder → RSSM observe → Actor → [steering, throttle].

    IMPORTANT: This is NOT a ReplayTrafficParticipantPolicy subclass, so the agent
    manager treats it as a standard policy (calls act() before physics step, expects
    a 2D action return).
    """

    # PID gains for GT speed tracking (calibrated on MetaDrive DefaultVehicle)
    _PID_KP = 0.4
    _PID_KI = 0.08
    _PID_KD = 0.05
    _PID_DT = 1.0 / 25.0

    def __init__(self, control_object, random_seed=None):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self._wm = _GLOBAL_WM
        self._actor = _GLOBAL_ACTOR
        self._device = _GLOBAL_DEVICE
        self._bev_size = _GLOBAL_BEV_SIZE
        self._prev_state = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._step = 0
        self._speed_err_int = 0.0  # PID integral for GT speed tracking
        self._prev_speed_err = 0.0
        self._debug_lat = 0.0
        self._debug_rh = 0.0
        self._debug_he = 0.0

    def _pid_action(self):
        """Navigation-based PID: track reference trajectory (merge path).

        Uses TrajectoryNavigation (reference_trajectory = SDC route = merge path).
        Lateral error + heading error → steering. Speed error → throttle.
        """
        try:
            nav = self.control_object.navigation
            # Lateral offset from reference trajectory (neg = right of path)
            lat = float(nav.current_lateral) if hasattr(nav, 'current_lateral') else 0.0
            # Desired heading: reference trajectory tangent at ego's longitudinal position
            road_heading = float(nav.current_heading_theta_at_long) if hasattr(nav, 'current_heading_theta_at_long') else 0.0
            ego_heading = float(self.control_object.heading_theta)

            # Heading error: road_heading relative to ego
            # Positive error = road is to the right → steer right (positive)
            heading_err = wrap_angle(road_heading - ego_heading)

            # Lateral: K=0.08 (reduced from 0.25 to prevent oscillation at highway speed)
            steer_lat = float(np.clip(lat * 0.08, -0.3, 0.3))
            # Heading: K=0.15 (reduced from 0.5)
            steer_hdg = float(np.clip(heading_err * 0.15, -0.2, 0.2))
            steer = float(np.clip(steer_lat + steer_hdg, -1.0, 1.0))

            self._debug_lat = lat
            self._debug_rh = road_heading
            self._debug_he = heading_err
        except Exception:
            steer = 0.0

        desired = float(_GLOBAL_GT_SPEEDS[self._step])
        current = float(self.control_object.speed)
        err = desired - current
        self._speed_err_int += err * self._PID_DT
        self._speed_err_int = float(np.clip(self._speed_err_int, -3.0, 3.0))
        deriv = (err - self._prev_speed_err) / self._PID_DT
        self._prev_speed_err = err
        throttle = (self._PID_KP * err + self._PID_KI * self._speed_err_int +
                    self._PID_KD * deriv)
        throttle = float(np.clip(throttle, -1.0, 1.0))
        return np.array([steer, throttle], dtype=np.float32), err

    def act(self, agent_id=None):
        global _GLOBAL_LAST_STATE, _GLOBAL_LAST_IS_PID
        bev = self._capture_bev()
        bev_t = torch.FloatTensor(bev).unsqueeze(0).to(self._device)

        # Debug: check if BEV is all zeros (camera failure)
        if self._step <= 0:
            bev_max = float(bev_t.max().cpu())
            print(f"[WM DEBUG] act() called, step={self._step}, bev_max={bev_max:.4f}", flush=True)

        with torch.no_grad():
            embed = self._wm.encode(bev_t)  # (1, embed_dim)

            if self._prev_state is None:
                self._prev_state = self._wm.get_initial_state(1, self._device)

            # Observe current frame with previous action
            act_t = torch.FloatTensor(self._prev_action).unsqueeze(0).unsqueeze(0).to(self._device)
            embed_t = embed.unsqueeze(0)  # (1, 1, embed_dim)
            states, _ = self._wm.rssm.observe(embed_t, act_t, self._prev_state)
            self._prev_state = states[-1]

            # Expose RSSM state for DAPO (squeeze batch=1 dim)
            _GLOBAL_LAST_STATE = {
                "deter": self._prev_state["deter"].squeeze(0).detach().cpu(),
                "stoch": self._prev_state["stoch"].squeeze(0).detach().cpu(),
            }

            # Collect feature for online crash detector training
            _GLOBAL_FEATURE_BUFFER.append(
                self._wm.get_feature(self._prev_state).squeeze(0).detach().cpu()
            )

            use_pid = False
            if (_GLOBAL_GT_SPEEDS is not None
                    and self._step < len(_GLOBAL_GT_SPEEDS)
                    and _GLOBAL_GT_BLEND > 0.0):
                if _GLOBAL_GT_BLEND >= 1.0 - 1e-6:
                    use_pid = True
                else:
                    use_pid = np.random.random() < _GLOBAL_GT_BLEND

            _GLOBAL_LAST_IS_PID = use_pid

            if use_pid:
                action_np, err = self._pid_action()

                desired = float(_GLOBAL_GT_SPEEDS[self._step])
                if self._step == 0:
                    ego = self.control_object
                    p = ego.position
                    h = float(ego.heading_theta)
                    s = float(ego.speed)
                    rh = math.degrees(getattr(self, '_debug_rh', 0))
                    he = math.degrees(getattr(self, '_debug_he', 0))
                    blend = _GLOBAL_GT_BLEND
                    print(f"[WM Policy] PID mode ON step=0 blend={blend:.2f}, "
                          f"pos=({p[0]:.1f},{p[1]:.1f}), "
                          f"ego_h={math.degrees(h):.1f}° road_h={rh:.1f}° h_err={he:.1f}°, "
                          f"speed={s:.1f}, desired={desired:.1f}")
                    sys.stdout.flush()
                if self._step % 20 == 0:
                    ego = self.control_object
                    p = ego.position
                    h = float(ego.heading_theta)
                    s = float(ego.speed)
                    sl = float(np.clip(getattr(self, '_debug_lat', 0) * 0.08, -0.3, 0.3))
                    sh = float(np.clip(getattr(self, '_debug_he', 0) * 0.15, -0.2, 0.2))
                    rh = math.degrees(getattr(self, '_debug_rh', 0))
                    he = math.degrees(getattr(self, '_debug_he', 0))
                    la = getattr(self, '_debug_lat', 0)
                    print(f"[WM Policy] step={self._step} steer={action_np[0]:+.3f} "
                          f"(lat{sl:+.3f}+hdg{sh:+.3f}), "
                          f"PID: err={err:+.2f} int={self._speed_err_int:+.3f} "
                          f"thr={action_np[1]:+.3f} "
                          f"speed={s:.1f}/{desired:.1f} pos=({p[0]:.1f},{p[1]:.1f}) "
                          f"ego_h={math.degrees(h):.1f}° road_h={rh:.1f}° h_err={he:.1f}° lat={la:+.2f}m")
                    sys.stdout.flush()
            else:
                _GLOBAL_LAST_IS_PID = False
                # Actor selects residual action (ResWM-style)
                feature = self._wm.get_feature(self._prev_state)
                prev_a = torch.FloatTensor(self._prev_action).unsqueeze(0).to(self._device)
                actor_out = self._actor(feature, prev_action=prev_a)
                if isinstance(actor_out, tuple):
                    action, _ = actor_out
                else:
                    action = actor_out
                action_np = action[0].cpu().numpy().astype(np.float32)
                # Debug: detect zero-action (intermittent bug investigation)
                if self._step <= 2 or (abs(float(action_np[0])) < 1e-6 and abs(float(action_np[1])) < 1e-6):
                    f_mean = float(feature.mean().cpu())
                    f_std = float(feature.std().cpu())
                    p0, p1 = float(action_np[0]), float(action_np[1])
                    print(f"[WM DEBUG] step={self._step} act=[{p0:+.6f},{p1:+.6f}] "
                          f"feat_mean={f_mean:+.6f} feat_std={f_std:+.6f} "
                          f"prev_act=[{self._prev_action[0]:+.4f},{self._prev_action[1]:+.4f}]",
                          flush=True)

        self._step += 1
        self._prev_action = action_np.copy()
        global _GLOBAL_PREV_ACTION, _GLOBAL_ACT_COUNT
        _GLOBAL_PREV_ACTION = action_np.copy()
        _GLOBAL_ACT_COUNT += 1
        # Debug: detect zero action (first occurrence per episode)
        if abs(float(action_np[0])) < 1e-6 and abs(float(action_np[1])) < 1e-6:
            if not getattr(self, '_zero_warned', False):
                self._zero_warned = True
                print(f"[WM ZERO!!] step={self._step} act=[0,0] use_pid={use_pid} "
                      f"call_count={_GLOBAL_ACT_COUNT}", flush=True)
        return action_np

    def current_step(self):
        return self._step

    def _capture_bev(self):
        """Capture top-down BEV, same logic as MetaDriveBEVEnv._capture_bev."""
        try:
            engine = self.control_object.engine
            rgb_cam = engine.sensors.get("rgb_camera")
            if rgb_cam is None:
                return np.zeros((3, self._bev_size, self._bev_size), dtype=np.uint8)

            ego = self.control_object
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
                return np.zeros((3, self._bev_size, self._bev_size), dtype=np.float32)

            # Center-crop to square then resize
            H, W = bev_img.shape[:2]
            size = min(H, W)
            dh = (H - size) // 2
            dw = (W - size) // 2
            bev_img = bev_img[dh:dh + size, dw:dw + size]

            if size != self._bev_size:
                from PIL import Image
                bev_img = np.array(
                    Image.fromarray(bev_img).resize(
                        (self._bev_size, self._bev_size), Image.BILINEAR
                    )
                )

            # Normalize to [0, 1] float (encoder was trained on this range)
            return bev_img.transpose(2, 0, 1).astype(np.float32) / 255.0

        except Exception:
            return np.zeros((3, self._bev_size, self._bev_size), dtype=np.float32)

    def reset(self):
        self._prev_state = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._step = 0
        self._speed_err_int = 0.0
        self._prev_speed_err = 0.0

    @classmethod
    def get_input_space(cls):
        import gymnasium as gym
        return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
