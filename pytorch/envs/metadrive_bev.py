"""
MetaDrive BEV Environment Wrapper.

Uses RGBCamera for top-down BEV rendering — same as offline data collection,
ensuring observation format alignment across Phase 2-4.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MetaDriveBEVEnv(gym.Env):
    """
    MetaDrive on-ramp env with RGBCamera BEV observation.

    Observation: 3-channel RGB BEV, (C, H, W) uint8 [0,255]
    Action: [steering, throttle] continuous, box [-1, 1]^2
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}

        self.bev_size = cfg.get("bev_size", 64)
        self.map_config = cfg.get("map_config", "SSrSS")
        self.traffic_density = cfg.get("traffic_density", 0.3)
        self.horizon = cfg.get("horizon", 1000)

        # Render BEV at higher res then downsample for cleaner output
        self._render_w = cfg.get("render_w", self.bev_size * 4)
        self._render_h = cfg.get("render_h", self.bev_size * 3)

        # Observation space: 3-channel RGB BEV
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, self.bev_size, self.bev_size),
            dtype=np.uint8,
        )

        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        self._env = None
        self._rgb_cam = None
        self._engine_origin = None
        self._create_metadrive_env()

        self.step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

    def _create_metadrive_env(self):
        """Create MetaDrive env with RGBCamera sensor."""
        try:
            from metadrive import MetaDriveEnv
            from metadrive.component.sensors.rgb_camera import RGBCamera

            self._env = MetaDriveEnv(dict(
                map_config=self.map_config,
                traffic_density=self.traffic_density,
                horizon=self.horizon,
                use_render=False,
                image_observation=True,
                sensors={"rgb_camera": (RGBCamera, self._render_w, self._render_h)},
                norm_pixel=False,
                vehicle_config=dict(
                    show_navi_mark=False,
                    use_special_color=True,
                ),
                interface_panel=[],
            ))
            self._has_metadrive = True
        except Exception as e:
            print(f"[WARN] MetaDrive not available ({e})")
            self._has_metadrive = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if self._has_metadrive:
            obs, info = self._env.reset()
            self._rgb_cam = self._env.engine.sensors.get("rgb_camera")
            self._engine_origin = self._env.engine.origin
        else:
            info = {}

        self.step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

        bev_obs = self._capture_bev()
        return bev_obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        if self._has_metadrive:
            obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            reward = self._compute_reward(obs, reward, info)
        else:
            reward = 0.0
            done = self.step_count >= self.horizon
            info = {}

        bev_obs = self._capture_bev()
        self.step_count += 1
        self._prev_action = action.copy()

        return bev_obs, reward, done, False, info

    def _capture_bev(self):
        """Capture top-down BEV from RGBCamera at ego position, same as offline data collection."""
        if not self._has_metadrive or self._rgb_cam is None:
            return np.zeros((3, self.bev_size, self.bev_size), dtype=np.uint8)

        try:
            ego = getattr(self._env, "agent", None) or self._env.vehicle
            ego_pos = ego.position
            ego_hpr = ego.origin.getHpr()
            heading = ego_hpr.getX()

            # Camera above ego looking straight down (same as collect_merge_data.py)
            bev_img = self._rgb_cam.perceive(
                to_float=False,
                new_parent_node=self._engine_origin,
                position=(float(ego_pos[0]), float(ego_pos[1]), 50.0),
                hpr=(heading, -90, 0),
            )

            if bev_img is None or bev_img.ndim != 3:
                return np.zeros((3, self.bev_size, self.bev_size), dtype=np.uint8)

            # Center-crop to square then resize to target
            H, W = bev_img.shape[:2]
            if H != W:
                size = min(H, W)
                dh = (H - size) // 2
                dw = (W - size) // 2
                bev_img = bev_img[dh:dh + size, dw:dw + size]

            if size != self.bev_size:
                from PIL import Image
                bev_img = np.array(
                    Image.fromarray(bev_img).resize(
                        (self.bev_size, self.bev_size), Image.BILINEAR
                    )
                )

            # HWC -> CHW
            return bev_img.transpose(2, 0, 1).astype(np.uint8)

        except Exception:
            return np.zeros((3, self.bev_size, self.bev_size), dtype=np.uint8)

    def _compute_reward(self, obs, env_reward, info):
        """Merge-specific reward shaping."""
        reward = 0.0

        if not isinstance(obs, dict):
            return env_reward if env_reward is not None else 0.0

        # Speed maintenance
        speed = obs.get("speed", [0, 0])
        if isinstance(speed, (list, np.ndarray)):
            speed_val = np.linalg.norm(speed)
        else:
            speed_val = float(speed)
        reward += min(speed_val / 20.0, 1.0) * 1.0

        # On lane bonus
        if obs.get("on_lane", True):
            reward += 0.5

        # Collision penalty
        if info.get("crash", False) or info.get("crash_vehicle", False):
            reward -= 50.0

        # Off road penalty
        if not obs.get("on_lane", True):
            reward -= 5.0

        # Success bonus
        if info.get("arrive_destination", False):
            reward += 100.0

        # Action smoothness
        reward -= 0.5 * np.abs(self._prev_action).sum()

        return float(reward)

    def close(self):
        if self._env is not None:
            self._env.close()
