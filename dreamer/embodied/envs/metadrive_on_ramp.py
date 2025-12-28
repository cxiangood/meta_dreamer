import functools
import os
import numpy as np
import elements
import embodied

try:
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.constants import DEFAULT_AGENT
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False
    DEFAULT_AGENT = None


class MetaDriveOnRamp(embodied.Env):
    """
    MetaDrive On-Ramp Merge Environment for DreamerV3
    Adapted from on-ramp_1.py and aligned with MetaDriveLaneKeeping interface.
    """

    def __init__(self, task, size=(64, 64), repeat=1, length=1000, **kwargs):
        if not METADRIVE_AVAILABLE:
            raise ImportError(
                "MetaDrive is not installed. Please install it with: pip install metadrive-simulator"
            )

        self._size = size
        self._repeat = repeat
        self._length = length
        self._random = np.random.RandomState()

        enable_render = os.environ.get('METADRIVE_RENDER', '0') == '1'
        if not hasattr(MetaDriveOnRamp, '_render_instance_created'):
            MetaDriveOnRamp._render_instance_created = False
        if enable_render and not MetaDriveOnRamp._render_instance_created:
            MetaDriveOnRamp._render_instance_created = True
            print(f"[MetaDrive] Enabling rendering for this environment instance")
        else:
            if enable_render:
                print(f"[MetaDrive] Rendering disabled for this instance (only first instance renders)")

        # Base configuration for an on-ramp merge scenario (match on-ramp_1.py).
        config = dict(
            # Keep onscreen rendering to avoid MetaDrive 'Render mode error',
            # matching metadrive_lane_keeping behavior in this repo.
            use_render=True,
            start_seed=42, # 固定种子
            horizon=length,
            map_config=dict(
                type="block_sequence",
                config="SSrSS",  # main road - ramp - main road
                lane_num=3,
                lane_width=3.5,
            ),
            traffic_density=0.1,
            random_traffic=True,
            random_spawn_lane_index=False,
            vehicle_config=dict(
                # Match spawn fields in on-ramp_1.py
                spawn_lane_index=None,
                spawn_longitude=10.0,
                spawn_lateral=0.0,
            ),
            sensors=dict(
                rgb_camera=(RGBCamera, size[0], size[1]),
            ),
            image_observation=True,
            norm_pixel=False,
            stack_size=1,
            # Reward knobs from MetaDrive; we will add light shaping in step().
            success_reward=20.0,
            out_of_road_penalty=5.0,
            crash_vehicle_penalty=10.0,
            crash_object_penalty=5.0,
            use_lateral_reward=True,
        )

        config.update(kwargs)

        if 'manual_control' not in config:
            config['manual_control'] = False
        if os.environ.get('METADRIVE_MANUAL_CONTROL') == '1':
            config['manual_control'] = True

        self._base_config = config.copy()

        self._env = MetaDriveEnv(config)
        self._done = True
        self._step_count = 0
        self._last_throttle_brake = 0.0
        self._last_steering = 0.0
        self._total_steps = 0
        self._initial_ramp_lane = None  # 记录初始匝道lane，用于检测汇入成功

    def _create_env_headless(self):
        print("[MetaDrive] Creating headless environment...")
        config = self._base_config.copy()
        # 渲染模式搞不懂了，反正就是True
        config['use_render'] = True
        config['manual_control'] = False
        self._env = MetaDriveEnv(config)
        print("[MetaDrive] Headless environment created successfully")

    @functools.cached_property
    def obs_space(self):
        spaces = {}
        spaces['image'] = elements.Space(np.uint8, (*self._size, 3), 0, 255)
        spaces['speed'] = elements.Space(np.float32, (), -np.inf, np.inf)
        spaces['acceleration'] = elements.Space(np.float32, (), -np.inf, np.inf)
        spaces['angular_velocity'] = elements.Space(np.float32, (3,), -np.inf, np.inf)
        spaces['current_steering'] = elements.Space(np.float32, (), -1.0, 1.0)
        spaces['current_throttle_brake'] = elements.Space(np.float32, (), -1.0, 1.0)
        spaces['distance_to_route'] = elements.Space(np.float32, (), -np.inf, np.inf)
        spaces['route_completion'] = elements.Space(np.float32, (), 0.0, 1.0)
        spaces['reward'] = elements.Space(np.float32)
        spaces['is_first'] = elements.Space(bool)
        spaces['is_last'] = elements.Space(bool)
        spaces['is_terminal'] = elements.Space(bool)
        return spaces

    @functools.cached_property
    def act_space(self):
        return {
            'steering': elements.Space(np.float32, (), -1.0, 1.0),
            'throttle_brake': elements.Space(np.float32, (), -1.0, 1.0),
            'reset': elements.Space(bool),
        }

    def _place_agent_on_ramp(self):
        """Place agent onto the ramp lane similarly to on-ramp_1.py approach."""
        try:
            # Try to locate a ramp-like lane by speed limit heuristic.
            from metadrive.utils.pg.utils import get_all_lanes
            roadnet = self._env.engine.current_map.road_network
            all_lanes = get_all_lanes(roadnet)
            ramp_lane = None
            for lane in all_lanes:
                try:
                    if hasattr(lane, 'speed_limit') and lane.speed_limit < 25:
                        ramp_lane = lane
                        break
                except Exception:
                    continue
            if ramp_lane is None and all_lanes:
                ramp_lane = all_lanes[0]
            if ramp_lane is None:
                return
            agent = self._env.agent
            # Use config values if provided, otherwise defaults from on-ramp_1.py
            vcfg = getattr(agent, 'config', {}) if hasattr(agent, 'config') else {}
            spawn_longitude = vcfg.get('spawn_longitude', 10.0)
            spawn_lateral = vcfg.get('spawn_lateral', 0.0)
            new_position = ramp_lane.position(spawn_longitude, spawn_lateral)
            new_heading = ramp_lane.heading_theta_at(spawn_longitude)
            if hasattr(agent, 'set_position'):
                agent.set_position(new_position, height=agent.HEIGHT / 2)
            if hasattr(agent, 'set_heading_theta'):
                agent.set_heading_theta(new_heading)
            agent.spawn_place = new_position
            if hasattr(agent, 'set_static'):
                agent.set_static(False)
            # 记录初始匝道lane，用于后续检测汇入成功
            self._initial_ramp_lane = ramp_lane
            print("[MetaDrive] Agent placed on ramp lane.")
        except Exception as e:
            print(f"[MetaDrive] Failed to place agent on ramp: {e}")

    def step(self, action):
        if action['reset'] or self._done:
            return self._reset()

        steering = np.float32(action['steering'])
        throttle_brake = np.float32(action['throttle_brake'])

        steering = np.clip(steering, -1.0, 1.0)
        throttle_brake = np.clip(throttle_brake, -1.0, 1.0)

        md_action = [float(steering), float(throttle_brake)]

        # ⭐ 关键修复：MetaDrive 的 EnvInputPolicy 在 before_step 时会访问 external_actions[agent_id]
        # 必须在调用 step() 之前设置 external_actions
        # 注意：虽然 DreamerV3 会创建多个并行环境实例（envs: 4），但每个环境实例只有一个 agent
        # 我们需要获取实际的 agent_id，因为 MetaDrive 可能使用 'default_agent' 或其他 ID
        agent_id = DEFAULT_AGENT
        if hasattr(self._env, 'agents') and len(self._env.agents) > 0:
            agent_id = list(self._env.agents.keys())[0]
        
        # 确保 external_actions 存在并设置动作
        # 同时设置实际 agent_id 和 DEFAULT_AGENT，因为 MetaDrive 内部可能使用不同的 ID
        if not hasattr(self._env.engine, 'external_actions') or self._env.engine.external_actions is None:
            self._env.engine.external_actions = {}
        
        total_reward = 0.0
        info = {}
        for _ in range(self._repeat):
            # 在每次循环迭代中重新设置 external_actions（可能被清空）
            # 设置实际 agent_id 和 DEFAULT_AGENT 两个键，确保 MetaDrive 能找到动作
            self._env.engine.external_actions[agent_id] = md_action
            if agent_id != DEFAULT_AGENT:
                self._env.engine.external_actions[DEFAULT_AGENT] = md_action
            
            obs, reward, terminated, truncated, info = self._env.step(md_action)

            # Light reward shaping tailored for merging:
            time_penalty = -0.01
            reward += time_penalty
            vehicle = self._env.agent
            velocity = getattr(vehicle, 'velocity', [0, 0, 0])
            speed = np.linalg.norm(velocity[:2])
            reward += 0.5 * speed
            if info.get('on_lane', False):
                reward += 0.2
            if info.get('crash_vehicle', False) or info.get('crash', False):
                reward -= 20.0
            if info.get('out_of_road', False):
                reward -= 10.0
            
            # 检测是否成功汇入主路
            # 成功条件：已经离开初始匝道lane，并且当前lane的速度限制>=25（主路特征）
            merged_successfully = False
            if hasattr(vehicle, 'lane') and vehicle.lane is not None and self._initial_ramp_lane is not None:
                try:
                    current_lane = vehicle.lane
                    # 检查是否已经离开初始匝道lane
                    if current_lane is not self._initial_ramp_lane:
                        # 检查当前lane的速度限制（主路通常>=25 m/s）
                        if hasattr(current_lane, 'speed_limit'):
                            if current_lane.speed_limit >= 25.0:
                                # 已经离开匝道进入主路
                                merged_successfully = True
                        else:
                            # 如果没有speed_limit属性，检查是否在on_lane上（作为备选判断）
                            if info.get('on_lane', False) and self._step_count > 10:
                                # 已经离开初始匝道且保持在车道上，认为已汇入
                                merged_successfully = True
                except Exception as e:
                    # 如果检测失败，不标记为成功
                    pass
            
            if merged_successfully:
                reward += 50.0  # 成功汇入主路的大奖励
                terminated = True  # 标记为成功结束
                print("[MetaDrive] Successfully merged into main road! Episode completed.")
            
            # 保留原有的到达终点奖励（虽然现在可能不会触发）
            if info.get('arrive_dest', False):
                reward += 20.0

            # Smooth action change to discourage oscillations.
            throttle_change = abs(throttle_brake - self._last_throttle_brake)
            if throttle_change < 0.3:
                reward += 0.1
            else:
                reward -= 0.15
            steering_change = abs(steering - self._last_steering)
            if steering_change > 0.6:
                reward -= 0.5 * steering_change

            self._last_throttle_brake = throttle_brake
            self._last_steering = steering

            total_reward += reward
            self._step_count += 1
            self._total_steps += 1

            if terminated or truncated:
                self._done = True
                break

        final_obs = self._get_obs(obs, total_reward, info, terminated or truncated)
        return final_obs

    def _reset(self):
        self._done = False
        self._step_count = 0
        self._last_throttle_brake = 0.0
        self._last_steering = 0.0
        self._initial_ramp_lane = None  # 重置初始匝道lane记录
        if hasattr(self, '_last_step_speed'):
            delattr(self, '_last_step_speed')
        if hasattr(self, '_last_speed'):
            delattr(self, '_last_speed')

        seed = 42
        try:
            obs, info = self._env.reset(seed=seed)
            # After reset, put agent on ramp.
            self._place_agent_on_ramp()
        except Exception as e:
            print(f"[RESET] Reset failed: {type(e).__name__}, recreating environment...")
            try:
                self._env.close()
            except Exception:
                pass
            try:
                self._create_env_headless()
                obs, info = self._env.reset(seed=seed)
                self._place_agent_on_ramp()
            except Exception as recreate_error:
                print(f"[RESET] Failed to recreate: {recreate_error}")
                raise

        return self._get_obs(obs, 0.0, info, is_first=True)

    def _get_obs(self, obs, reward, info, done=False, is_first=False):
        vehicle = self._env.agent

        # Extract image
        if 'image' in obs:
            image = obs['image']
            if len(image.shape) == 4:
                image = image[..., -1]
            if len(image.shape) == 3 and image.shape[-1] == 4:
                image = image[..., :3]
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            try:
                camera = self._env.agent.get_camera("rgb_camera")
                if camera is not None:
                    image = camera.perceive(False)
                    if len(image.shape) == 4:
                        image = image[..., -1]
                    if len(image.shape) == 3 and image.shape[-1] == 4:
                        image = image[..., :3]
                else:
                    image = self._env.render(mode='rgb_array')
                    image = np.array(image) if image is not None else np.zeros((*self._size, 3), dtype=np.uint8)
            except Exception as e:
                print(f"[Warning] Failed to get image from camera: {e}, using fallback render")
                image = self._env.render(mode='rgb_array')
                image = np.array(image) if image is not None else np.zeros((*self._size, 3), dtype=np.uint8)

        # Resize if needed
        if image.shape[:2] != self._size:
            try:
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(image)
                pil_image = pil_image.resize(self._size, PILImage.LANCZOS)
                image = np.array(pil_image)
            except ImportError:
                # Simple fallback
                if image.shape[:2] != self._size:
                    h_ratio = image.shape[0] // self._size[0]
                    w_ratio = image.shape[1] // self._size[1]
                    if h_ratio > 0 and w_ratio > 0:
                        image = image[::h_ratio, ::w_ratio]
                    if image.shape[0] > self._size[0]:
                        image = image[:self._size[0]]
                    if image.shape[1] > self._size[1]:
                        image = image[:, :self._size[1]]
                    if image.shape[0] < self._size[0] or image.shape[1] < self._size[1]:
                        new_image = np.zeros((*self._size, 3), dtype=image.dtype)
                        new_image[:image.shape[0], :image.shape[1]] = image
                        image = new_image

        image = image.astype(np.uint8)

        velocity = getattr(vehicle, 'velocity', [0, 0, 0])
        speed = np.linalg.norm(velocity[:2])
        angular_vel = getattr(vehicle, 'angular_velocity', [0, 0, 0])
        if not isinstance(angular_vel, np.ndarray):
            angular_vel = np.array(angular_vel, dtype=np.float32)

        steering = getattr(vehicle, 'steering', 0.0)
        throttle_brake = getattr(vehicle, 'throttle_brake_action', 0.0)

        route_completion = info.get('route_completion', 0.0)
        if hasattr(vehicle, 'lane') and vehicle.lane is not None:
            try:
                long, lat = vehicle.lane.local_coordinates(vehicle.position)
                distance_to_route = abs(lat)
            except Exception:
                distance_to_route = 0.0
        else:
            distance_to_route = 0.0

        if hasattr(self, '_last_speed'):
            acceleration = (speed - self._last_speed) * 10.0
        else:
            acceleration = 0.0
        self._last_speed = speed

        observation = {
            'image': image,
            'speed': np.float32(speed),
            'acceleration': np.float32(acceleration),
            'angular_velocity': angular_vel.astype(np.float32),
            'current_steering': np.float32(steering),
            'current_throttle_brake': np.float32(throttle_brake),
            'distance_to_route': np.float32(distance_to_route),
            'route_completion': np.float32(route_completion),
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': done,
            'is_terminal': done and info.get('crash', False),
        }
        # Save rollout frames to result folder (aligned with metadrive_lane_keeping.py behavior)
        if not is_first and not done:
            try:
                from PIL import Image
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
                save_dir = os.path.join(project_root, 'result-on-ramp')
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f'imagine_{self._total_steps:06d}.png')
                Image.fromarray(image).save(fname)
            except Exception as e:
                print(f"[Save Image] Failed: {e}")
        return observation

    def render(self):
        return self._env.render(mode='rgb_array')

    def close(self):
        if hasattr(self, '_env'):
            self._env.close()


