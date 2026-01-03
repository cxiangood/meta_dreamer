import functools
import os
import numpy as np
import elements
import embodied

try:
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False


class MetaDriveLaneReduction(embodied.Env):
    """
    MetaDrive Lane Reduction Environment for DreamerV3
    Scenario: Vehicle starts on a multi-lane road (e.g., 3 lanes) and must navigate
    through a section where lanes reduce (e.g., to 2 or 1 lane). The agent needs
    to learn to change lanes appropriately before the reduction point.
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

        # 检查是否需要渲染（通过环境变量控制，默认禁用渲染以兼容无显示环境）
        # 可以通过设置 METADRIVE_RENDER=1 来启用渲染
        enable_render = os.environ.get('METADRIVE_RENDER', '0') != '0'
        
        # 检查是否有显示设备（DISPLAY 环境变量）
        has_display = os.environ.get('DISPLAY') is not None
        
        # 如果没有显示设备但启用了渲染，使用 offscreen 渲染（headless）
        # 这样仍然可以获取图像观察，但不会显示窗口
        if enable_render and not has_display:
            print("[MetaDrive] 警告: 未检测到 DISPLAY 环境变量，将使用 offscreen 渲染（无窗口模式）")
            print("[MetaDrive] 提示: 如果需要显示窗口，请设置 DISPLAY 环境变量或使用 X11 forwarding")
            # 使用 offscreen 渲染，仍然可以获取图像
            use_render = False
            image_obs_enabled = True
        else:
            use_render = enable_render
            image_obs_enabled = True

        # Base configuration for a lane reduction scenario
        # Start with 3 lanes, then reduce to 2 lanes, then back to 3 lanes
        # The config "SSrSS" creates: Straight-Straight-ramp-Straight-Straight
        # We'll use a custom block sequence that creates lane reduction
        config = dict(
            # 根据显示设备情况决定是否渲染
            use_render=use_render,
            start_seed=42, # 固定种子
            horizon=length,
            map_config=dict(
                type="block_sequence",
                config="SSySS",  # S=Straight, y=Merge (车道合并/减少), 3车道→2车道
                # 配置：2个直道(3车道) → Merge块(减少到2车道) → 2个直道(2车道)
                lane_num=3,  # Start with 3 lanes, Merge块会减少到2车道
                lane_width=3.5,
            ),
            traffic_density=0.1,
            random_traffic=True,
            random_spawn_lane_index=False,
            vehicle_config=dict(
                # Spawn on the main road (not on ramp)
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

    def _create_env_headless(self):
        print("[MetaDrive] Creating headless environment...")
        config = self._base_config.copy()
        # 无显示环境下使用 offscreen 渲染
        config['use_render'] = False
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

    def _place_agent_on_main_road(self):
        """Place agent on the rightmost lane before the merge point (lane reduction scenario).
        
        Ego车应该放置在车道即将合并前的最右侧车道（即将被合并掉的车道）。
        在3车道→2车道的场景中，最右侧车道会被合并。
        """
        try:
            from metadrive.utils.pg.utils import get_all_lanes
            roadnet = self._env.engine.current_map.road_network
            all_lanes = get_all_lanes(roadnet)
            
            # 找到Merge块之前的3车道路段，选择最右侧车道
            main_road_lanes = []
            for lane in all_lanes:
                try:
                    # 查找主路lane（速度限制>=25，排除匝道）
                    if hasattr(lane, 'speed_limit') and lane.speed_limit >= 25:
                        main_road_lanes.append(lane)
                except Exception:
                    continue
            
            if not main_road_lanes and all_lanes:
                # 如果没有找到，使用所有lane
                main_road_lanes = all_lanes
            
            if main_road_lanes:
                # 找到最右侧车道：在3车道配置中，最右侧车道是lane index最大的
                # 方法1: 通过road network的graph找到同一道路段的所有车道，选择index最大的
                rightmost_lane = None
                try:
                    # 遍历road network的graph，找到3车道路段
                    graph = roadnet.graph
                    max_lane_index = -1
                    
                    # 遍历所有道路段，找到3车道路段中最右侧的车道
                    for from_node in graph:
                        for to_node in graph[from_node]:
                            lanes = graph[from_node][to_node]
                            if lanes and len(lanes) >= 3:  # 找到3车道或以上的道路段
                                # 在这个道路段中，最右侧车道的index通常是最大的
                                # 在MetaDrive中，lane index通常是 (from_node, to_node, lane_index)
                                # 其中lane_index从0开始，0是最左侧，最大的index是最右侧
                                for i, lane in enumerate(lanes):
                                    try:
                                        # 检查是否是主路lane（速度限制>=25）
                                        if hasattr(lane, 'speed_limit') and lane.speed_limit >= 25:
                                            # 在同一道路段中，index最大的就是最右侧的
                                            if rightmost_lane is None or i > max_lane_index:
                                                max_lane_index = i
                                                rightmost_lane = lane
                                    except Exception:
                                        continue
                except Exception:
                    pass
                
                # 方法2: 如果方法1失败，通过位置判断（y坐标最大的）
                if rightmost_lane is None:
                    max_lateral = float('-inf')
                    for lane in main_road_lanes:
                        try:
                            test_pos = lane.position(120.0, 0.0)  # 在Merge块之前的位置测试
                            if test_pos is not None:
                                lateral_pos = test_pos[1] if len(test_pos) > 1 else 0.0
                                if lateral_pos > max_lateral:
                                    max_lateral = lateral_pos
                                    rightmost_lane = lane
                        except Exception:
                            continue
                
                # 方法3: 如果还是没找到，使用最后一个lane（通常是最右侧的）
                if rightmost_lane is None:
                    rightmost_lane = main_road_lanes[-1]
                
                if rightmost_lane is not None:
                    agent = self._env.agent
                    # 放置在Merge块之前的位置（距离起点较远，但还没到Merge块）
                    # 根据"SSySS"配置，Merge块大约在第2个Straight块之后
                    # 每个Straight块大约50-100米，所以放在80-100米处比较合适
                    vcfg = getattr(agent, 'config', {}) if hasattr(agent, 'config') else {}
                    spawn_longitude = vcfg.get('spawn_longitude', 80.0)  # Merge块之前的位置
                    spawn_lateral = vcfg.get('spawn_lateral', 0.0)  # 车道中心
                    new_position = rightmost_lane.position(spawn_longitude, spawn_lateral)
                    new_heading = rightmost_lane.heading_theta_at(spawn_longitude)
                    if hasattr(agent, 'set_position'):
                        agent.set_position(new_position, height=agent.HEIGHT / 2)
                    if hasattr(agent, 'set_heading_theta'):
                        agent.set_heading_theta(new_heading)
                    agent.spawn_place = new_position
                    if hasattr(agent, 'set_static'):
                        agent.set_static(False)
                    print("[MetaDrive] Agent placed on rightmost lane before merge point (lane reduction scenario).")
        except Exception as e:
            print(f"[MetaDrive] Failed to place agent on rightmost lane: {e}")

    def step(self, action):
        if action['reset'] or self._done:
            return self._reset()

        steering = np.float32(action['steering'])
        throttle_brake = np.float32(action['throttle_brake'])

        steering = np.clip(steering, -1.0, 1.0)
        throttle_brake = np.clip(throttle_brake, -1.0, 1.0)

        md_action = [float(steering), float(throttle_brake)]

        total_reward = 0.0
        info = {}
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self._env.step(md_action)

            # Light reward shaping tailored for lane reduction:
            # Encourage forward progress and proper lane positioning
            time_penalty = -0.01
            reward += time_penalty
            vehicle = self._env.agent
            velocity = getattr(vehicle, 'velocity', [0, 0, 0])
            speed = np.linalg.norm(velocity[:2])
            reward += 0.5 * speed
            
            # Reward for staying on lane (important for lane reduction)
            if info.get('on_lane', False):
                reward += 0.2
            
            # Check if vehicle is in a valid lane (not about to be forced off road)
            if hasattr(vehicle, 'lane') and vehicle.lane is not None:
                try:
                    long, lat = vehicle.lane.local_coordinates(vehicle.position)
                    # Reward for staying centered in lane
                    lane_center_reward = 0.1 * (1.0 - abs(lat) / 1.75)  # Normalize by half lane width
                    reward += lane_center_reward
                except Exception:
                    pass
            
            # Penalties
            if info.get('crash_vehicle', False) or info.get('crash', False):
                reward -= 20.0
            if info.get('out_of_road', False):
                reward -= 10.0
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
        if hasattr(self, '_last_step_speed'):
            delattr(self, '_last_step_speed')
        if hasattr(self, '_last_speed'):
            delattr(self, '_last_speed')

        seed = 42
        try:
            obs, info = self._env.reset(seed=seed)
            # After reset, put agent on main road (not on ramp)
            self._place_agent_on_main_road()
        except Exception as e:
            print(f"[RESET] Reset failed: {type(e).__name__}, recreating environment...")
            try:
                self._env.close()
            except Exception:
                pass
            try:
                self._create_env_headless()
                obs, info = self._env.reset(seed=seed)
                self._place_agent_on_main_road()
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
        # Save rollout frames to result folder
        if not is_first and not done:
            try:
                from PIL import Image
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
                save_dir = os.path.join(project_root, 'result-lane-reduction')
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

