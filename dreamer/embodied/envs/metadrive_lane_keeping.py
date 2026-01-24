import functools
import os
import shutil
import gym
import numpy as np
import elements
import embodied

try:
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.component.sensors.depth_camera import DepthCamera
    from metadrive.constants import DEFAULT_AGENT
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class MetaDriveLaneKeeping(embodied.Env):
    """
    MetaDrive Lane Keeping Environment for DreamerV3
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
        
        # 检查是否需要渲染（通过环境变量控制，默认启用渲染）
        # 可以通过设置 METADRIVE_RENDER=0 来禁用渲染
        enable_render = os.environ.get('METADRIVE_RENDER', '1') != '0'
        
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
        
        # MetaDrive environment configuration for lane keeping
        # Use the correct MetaDrive configuration approach
        config = dict(
            # Basic environment settings
            use_render=use_render,  # 根据显示设备情况决定是否渲染
            num_scenarios=1000,
            start_seed=0,
            horizon=100000,  # Increased to 10^5 steps per rollout
            
            # Map configuration - 使用更长的地图
            map=10,  # 使用10号地图，比较长（1-99可选，数字越大路段越长）
            
            # Traffic settings
            traffic_density=0.1,
            random_traffic=True,
            
            # 起点设置：禁用随机起点，使用固定位置
            random_spawn_lane_index=False,  # 关键：禁用随机起点
            
            # ⭐⭐⭐ 顶层vehicle_config - 这是所有车辆的默认配置
            vehicle_config=dict(
                navigation_module=NodeNetworkNavigation,  # ⭐ 关键：启用导航模块
                show_navi_mark=True,  # 显示导航waypoints（蓝色方块）
                show_dest_mark=True,  # 显示目的地标记（红色标记）
                show_line_to_navi_mark=True,  # 显示到waypoint的连线
                show_line_to_dest=True,  # 显示到终点的连线
                show_lidar=False,
                enable_reverse=False,
                image_source="rgb_camera",  # ⭐ 指定使用哪个 camera 作为图像观察源
            ),
            
            # ⭐⭐⭐ Agent配置 - 特定agent的配置（包括spawn位置）
            agent_configs={
                DEFAULT_AGENT: dict(
                    spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),  # 最右侧车道
                )
            },
            
            # Use sensors dictionary for camera
            sensors=dict(
                rgb_camera=(RGBCamera, 64, 64),  # 修改分辨率为64x64
            ),
            
            # ⭐ 关键：启用图像观察模式
            image_observation=True,  # 告诉 MetaDrive 使用图像作为观察
            norm_pixel=False,  # 不归一化像素值，保持 0-255
            stack_size=1,  # 不使用图像堆叠，只使用单帧
            
            # Reward configuration
            success_reward=100.0,
            out_of_road_penalty=20.0,  # 提高出界惩罚
            crash_vehicle_penalty=50.0,  # 提高车辆碰撞惩罚
            crash_object_penalty=30.0,  # 提高物体碰撞惩罚
            driving_reward=5.0,  # 降低基础驾驶奖励
            use_lateral_reward=True,  # 启用车道保持奖励
            
            # Termination settings - 使用悬崖模型：出界立即终止
            out_of_road_done=True,  # 出界立即终止episode（悬崖模型）
            crash_vehicle_done=False,  # 车辆碰撞继续（允许学习从错误中恢复）
            crash_object_done=False,  # 物体碰撞继续（允许学习从错误中恢复）
            
        )
        
        # Update config with user provided kwargs
        # Only allow keys that exist in the default config to avoid MetaDrive KeyError
        for k, v in kwargs.items():
            if k in config or k in ('agent_configs', 'sensors'):
                config[k] = v
            else:
                # silently ignore keys that MetaDrive doesn't accept (e.g., 'logdir' from training config)
                print(f"[MetaDrive] Ignoring unknown config key: {k}")

        # Ensure manual_control is OFF by default unless explicitly enabled
        # either via kwargs or the METADRIVE_MANUAL_CONTROL environment variable.
        if 'manual_control' not in config:
            config['manual_control'] = False
        if os.environ.get('METADRIVE_MANUAL_CONTROL') == '1':
            config['manual_control'] = True
        
        # 保存配置以便在需要时重新创建环境
        self._base_config = config.copy()
        
        self._env = MetaDriveEnv(config)
        self._md_action_mode = 'agent_dict' # 显式指定使用字典模式，最稳健
        self._md_agent_ids = [DEFAULT_AGENT] # 单智能体固定 ID
        self._done = True
        self._step_count = 0
        self._last_throttle_brake = 0.0  # 记录上一步的油门/刹车动作
        self._last_steering = 0.0  # 记录上一步的转向动作
        self._total_steps = 0  # 记录总步数，用于探索偏置衰减
        # Deterministic evaluation: use fixed seed if provided via env var
        fixed_seed = os.environ.get('METADRIVE_FIXED_SEED')
        self._fixed_seed = int(fixed_seed) if fixed_seed is not None else None
        # Reward logging path (per-episode CSV)
        self._reward_log_path = os.path.expanduser(os.environ.get('METADRIVE_REWARD_LOG', '~/metadrive_eval_rewards.csv'))
        # Episode bookkeeping
        self._episode_total_reward = 0.0
        self._episode_length = 0
        self._episode_count = 0
        self._current_seed = None
        self._last_waypoint_idx = None
        self._last_waypoint_reached = False
        self._last_waypoint_dist = float('inf')
        # Ensure log file exists with header (best-effort)
        try:
            if not os.path.exists(self._reward_log_path):
                with open(self._reward_log_path, 'w') as f:
                    f.write('episode,total_reward,length,seed,reason\n')
        except Exception:
            pass

        # Temporary frames root (store per-episode frames until user/export)
        self._frames_tmp_root = os.path.expanduser(os.environ.get('METADRIVE_FRAMES_TMP', os.path.join(os.getcwd(), 'logs', 'frames_tmp')))
        os.makedirs(self._frames_tmp_root, exist_ok=True)
        self._current_episode_dir = None
        # how many recent temp episodes to keep
        self._keep_temp_episodes = int(os.environ.get('METADRIVE_KEEP_EPISODES', '10'))
    
    def _recreate_env(self):
        """重新创建环境（当渲染窗口被关闭时使用）"""
        print("[MetaDrive] Recreating environment...")
        # 复制配置，保持原有的渲染设置
        config = self._base_config.copy()
        config['manual_control'] = False
        
        # 创建新的环境实例
        self._env = MetaDriveEnv(config)
        print("[MetaDrive] Environment recreated successfully")
        
    @functools.cached_property
    def obs_space(self):
        spaces = {}
        
        # Image observation
        spaces['image'] = elements.Space(np.uint8, (*self._size, 3), 0, 255)
        
        # Vehicle state observations
        spaces['speed'] = elements.Space(np.float32, (), -np.inf, np.inf)
        spaces['acceleration'] = elements.Space(np.float32, (), -np.inf, np.inf)
        spaces['angular_velocity'] = elements.Space(np.float32, (3,), -np.inf, np.inf)  # Roll, pitch, yaw rates
        spaces['current_steering'] = elements.Space(np.float32, (), -1.0, 1.0)
        spaces['current_throttle_brake'] = elements.Space(np.float32, (), -1.0, 1.0)
        
        # Navigation information
        spaces['distance_to_route'] = elements.Space(np.float32, (), -np.inf, np.inf)
        spaces['route_completion'] = elements.Space(np.float32, (), 0.0, 1.0)
        
        # Standard RL observations
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

    def step(self, action):
        if action['reset'] or self._done:
            return self._reset()
        
        # 动作处理
        steering = np.clip(np.float32(action['steering']), -1.0, 1.0)
        throttle_brake = np.clip(np.float32(action['throttle_brake']), -1.0, 1.0)
        
        self._total_steps += 1
        
        md_action = [float(steering), float(throttle_brake)]
        
        # 执行环境步进
        total_reward = 0.0
        obs = None
        info = {}
        terminated = False
        truncated = False
        
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self._env.step(md_action)
            
            vehicle = self._env.agent
            
            # ---------------------------------------------------------
            # 奖励函数设计 - 参考 MetaDrive ScenarioEnv.reward_function
            # 主要组件：
            # 1. driving_reward: 前进奖励（纵向位移）
            # 2. lateral_penalty: 横向偏移惩罚
            # 3. heading_penalty: 航向偏差惩罚
            # 4. steering_range_penalty: 转向范围惩罚（基于速度）
            # 5. 碰撞/出界惩罚
            # 6. 成功奖励
            # ---------------------------------------------------------
            
            # 奖励配置参数（与 ScenarioEnv 保持一致）
            driving_reward_weight = 1.0
            lateral_penalty_weight = 0.5
            heading_penalty_weight = 1.0
            steering_range_penalty_weight = 0.5
            max_lateral_dist = 4.0
            success_reward = 5.0
            out_of_road_penalty = 5.0
            on_lane_line_penalty = 1.0
            crash_vehicle_penalty = 1.0
            crash_object_penalty = 1.0
            no_negative_reward = True  # 是否允许负奖励
            
            components = {}
            
            # 获取车辆状态
            velocity = getattr(vehicle, 'velocity', [0, 0, 0])
            speed = float(np.linalg.norm(velocity[:2]))
            
            # 获取导航信息
            nav = getattr(vehicle, 'navigation', None) or getattr(vehicle, 'navigation_module', None)
            
            # 获取纵向和横向位置
            current_long = 0.0
            current_lat = 0.0
            last_long = getattr(self, '_last_long', 0.0)
            
            try:
                if nav is not None:
                    # 优先使用导航模块的属性
                    if hasattr(nav, 'current_longitude'):
                        current_long = float(nav.current_longitude)
                    if hasattr(nav, 'current_lateral'):
                        current_lat = float(nav.current_lateral)
                    if hasattr(nav, 'last_longitude'):
                        last_long = float(nav.last_longitude)
                    else:
                        last_long = getattr(self, '_last_long', current_long)
                
                # 如果导航模块没有这些属性，从车道获取
                if current_long == 0.0 and hasattr(vehicle, 'lane') and vehicle.lane is not None:
                    current_long, current_lat = vehicle.lane.local_coordinates(vehicle.position)
                    current_long = float(current_long)
                    current_lat = float(current_lat)
            except Exception:
                pass
            
            # 1. 前进奖励 (driving_reward)
            long_progress = current_long - last_long
            driving_reward = driving_reward_weight * long_progress
            components['driving_reward'] = driving_reward
            
            # 2. 横向偏移惩罚 (lateral_penalty)
            lateral_factor = abs(current_lat) / max_lateral_dist
            lateral_penalty = -lateral_factor * lateral_penalty_weight
            components['lateral_penalty'] = lateral_penalty
            
            # 3. 航向偏差惩罚 (heading_penalty)
            heading_penalty = 0.0
            try:
                if nav is not None and hasattr(nav, 'current_heading_theta_at_long'):
                    ref_line_heading = nav.current_heading_theta_at_long
                    heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi
                    heading_penalty = -heading_diff * heading_penalty_weight
                elif hasattr(vehicle, 'lane') and vehicle.lane is not None:
                    # 备选：使用车道方向
                    lane_heading = vehicle.lane.heading_theta_at(current_long)
                    heading_diff = wrap_to_pi(abs(vehicle.heading_theta - lane_heading)) / np.pi
                    heading_penalty = -heading_diff * heading_penalty_weight
            except Exception:
                pass
            components['heading_penalty'] = heading_penalty
            
            # 4. 转向范围惩罚 (steering_range_penalty)
            # 基于速度的转向限制：速度越快，允许的转向角度越小
            steering_range_penalty = 0.0
            try:
                current_steering = abs(steering)
                allowed_steering = 1.0 / max(speed, 1e-2)  # 速度越快，允许转向越小
                overflowed_steering = min(allowed_steering - current_steering, 0)
                steering_range_penalty = overflowed_steering * steering_range_penalty_weight
            except Exception:
                pass
            components['steering_range_penalty'] = steering_range_penalty
            
            # 计算步骤奖励
            step_reward = driving_reward + lateral_penalty + heading_penalty + steering_range_penalty
            
            # 是否禁止负奖励
            if no_negative_reward:
                step_reward = max(step_reward, 0)
            
            reward = step_reward
            
            # 5. 碰撞惩罚（覆盖步骤奖励）
            if vehicle.crash_vehicle:
                reward = -crash_vehicle_penalty
                components['crash_vehicle'] = -crash_vehicle_penalty
            if vehicle.crash_object:
                reward = -crash_object_penalty
                components['crash_object'] = -crash_object_penalty
            
            # 车道线惩罚
            if getattr(vehicle, 'on_yellow_continuous_line', False) or \
               getattr(vehicle, 'crash_sidewalk', False) or \
               getattr(vehicle, 'on_white_continuous_line', False):
                reward = -on_lane_line_penalty
                components['on_lane_line'] = -on_lane_line_penalty
            
            # 6. 终止奖励/惩罚
            if terminated or truncated:
                if info.get('arrive_dest', False):
                    reward = success_reward
                    components['success'] = success_reward
                    print(f"[Step {self._step_count}] *** ARRIVED! ***")
                elif info.get('out_of_road', False):
                    reward = -out_of_road_penalty
                    components['out_of_road'] = -out_of_road_penalty
                    print(f"[Step {self._step_count}] *** OUT OF ROAD! ***")
                elif info.get('crash', False) or info.get('crash_vehicle', False):
                    print(f"[Step {self._step_count}] *** CRASH! ***")
            
            # 更新状态记录
            self._last_long = current_long
            self._last_throttle_brake = throttle_brake
            self._last_steering = steering

            # 打印详细的奖励信息（每50步打印一次以避免刷屏）
            if self._step_count % 50 == 0:
                comp_items = ', '.join([f"{k}={v:.3f}" for k, v in components.items()])
                print(f"[Step {self._step_count}] Speed: {speed:.2f}m/s, Reward: {reward:.4f}")
                print(f"  Components: {comp_items}")
                print(f"  Position: long={current_long:.2f}, lat={current_lat:.2f}")
            
            total_reward += reward
            self._step_count += 1
            # episode bookkeeping
            try:
                self._episode_total_reward += reward
                self._episode_length += 1
            except Exception:
                # episode bookkeeping may not be initialized in some call paths
                pass
            
            if terminated or truncated:
                self._done = True
                # 打印episode总结
                print(f"\n{'='*60}")
                print(f"[Episode End] Steps: {self._step_count}, Total Reward: {total_reward:.2f}")
                reason = "SUCCESS" if info.get('arrive_dest', False) else \
                         ("CRASH" if (info.get('crash', False) or info.get('crash_vehicle', False)) else \
                         ("OUT_OF_ROAD" if info.get('out_of_road', False) else "OTHER"))
                route_completion = info.get('route_completion', 0.0)
                print(f"[Episode End] Reason: {reason}, Route Completion: {route_completion:.2%}")
                print(f"{'='*60}\n")
                # Log to CSV (best-effort)
                try:
                    ep_idx = self._episode_count
                    with open(self._reward_log_path, 'a') as f:
                        f.write(f"{ep_idx},{total_reward:.6f},{self._episode_length},{self._current_seed},{reason}\n")
                except Exception:
                    pass

                self._episode_count += 1
                break
                
        final_obs = self._get_obs(obs, total_reward, info, terminated or truncated)
        return final_obs

    def _reset(self):
        """Reset the environment"""
        print(f"\n{'>'*60}")
        print(f"[RESET] Starting new episode (total episodes so far: ~{self._total_steps // 100})...")
        print(f"{'>'*60}")
        
        self._done = False
        self._step_count = 0
        self._last_throttle_brake = 0.0  # 重置上一步油门值
        self._last_steering = 0.0  # 重置上一步转向值
        self._last_long = 0.0  # 重置上一步的纵向位置
        
        # 重置速度追踪（用于加速度计算）
        if hasattr(self, '_last_step_speed'):
            delattr(self, '_last_step_speed')
        if hasattr(self, '_last_speed'):
            delattr(self, '_last_speed')
        
        # Choose seed: fixed seed if provided, otherwise random
        if self._fixed_seed is not None:
            seed = self._fixed_seed
        else:
            seed = int(self._random.randint(0, 1000))  # MetaDrive requires seed in [0:1000)

        # remember current seed for logging
        self._current_seed = seed

        try:
            obs, info = self._env.reset(seed=seed)
            # reset episode bookkeeping
            self._episode_total_reward = 0.0
            self._episode_length = 0
            print(f"[RESET] ✓ Episode started with map seed {seed}")
            # create per-episode temp dir for frames (index = current completed episodes)
            ep_idx = self._episode_count
            self._current_episode_dir = os.path.join(self._frames_tmp_root, f'episode_{ep_idx:06d}')
            try:
                os.makedirs(self._current_episode_dir, exist_ok=True)
            except Exception:
                self._current_episode_dir = self._frames_tmp_root
            # prune older temp episodes beyond keep limit
            try:
                dirs = sorted([d for d in os.listdir(self._frames_tmp_root) if d.startswith('episode_')])
                if len(dirs) > self._keep_temp_episodes:
                    to_remove = dirs[:len(dirs) - self._keep_temp_episodes]
                    for dname in to_remove:
                        full = os.path.join(self._frames_tmp_root, dname)
                        try:
                            if os.path.isdir(full):
                                shutil.rmtree(full)
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception as e:
            error_msg = str(e)
            
            # 如果重置失败（例如窗口被关闭），重新创建环境
            print(f"[RESET] ✗ Reset failed: {type(e).__name__}, recreating environment...")
            
            # 关闭旧环境（如果还存在）
            try:
                self._env.close()
            except Exception as close_error:
                pass
            
            # 重新创建环境
            try:
                self._recreate_env()
                obs, info = self._env.reset(seed=seed)
                print(f"[RESET] ✓ Environment recreated, seed {seed}")
            except Exception as recreate_error:
                print(f"[RESET] ✗ Failed to recreate: {recreate_error}")
                raise
        
        return self._get_obs(obs, 0.0, info, is_first=True)

    def _get_obs(self, obs, reward, info, done=False, is_first=False):
        """Convert MetaDrive observation to DreamerV3 format"""
        # Get vehicle state
        vehicle = self._env.agent
        
        # Extract image observation from MetaDrive
        # MetaDrive 在 image_observation=True 时返回 {'image': ..., 'state': ...}
        if 'image' in obs:
            image = obs['image']
            # 处理图像堆叠：如果 shape 是 (H, W, C, stack_size)，取最后一帧
            if len(image.shape) == 4:  # (H, W, C, stack_size)
                image = image[..., -1]  # 取最后一帧
            # 处理 RGBA 转 RGB
            if len(image.shape) == 3 and image.shape[-1] == 4:  # RGBA to RGB
                image = image[..., :3]
            # 确保是 uint8 类型
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            # Fallback: 直接从 camera sensor 获取图像
            try:
                camera = self._env.agent.get_camera("rgb_camera")
                if camera is not None:
                    camera.RESOLUTION = tuple(self._size)  # 按配置分辨率输出
                    image = camera.perceive(False)  # False = 不归一化，返回 uint8
                    if len(image.shape) == 4:  # 如果有堆叠维度
                        image = image[..., -1]
                    if len(image.shape) == 3 and image.shape[-1] == 4:  # RGBA to RGB
                        image = image[..., :3]
                else:
                    # 最后的 fallback: render
                    image = self._env.render(mode='rgb_array')
                    if image is not None:
                        image = np.array(image)
                    else:
                        image = np.zeros((*self._size, 3), dtype=np.uint8)
            except Exception as e:
                print(f"[Warning] Failed to get image from camera: {e}, using fallback render")
                image = self._env.render(mode='rgb_array')
                if image is not None:
                    image = np.array(image)
                else:
                    image = np.zeros((*self._size, 3), dtype=np.uint8)
        
        # Resize image if needed using numpy-based resizing
        if image.shape[:2] != self._size:
            try:
                # Try to use PIL for resizing
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(image)
                # Pillow>=9 uses PILImage.Resampling; fall back to PILImage.LANCZOS if missing
                try:
                    resample = PILImage.Resampling.LANCZOS
                except Exception:
                    # fallback: try to get attribute or use integer constant 1 which maps to LANCZOS
                    resample = getattr(PILImage, 'LANCZOS', 1)
                pil_image = pil_image.resize(self._size, resample)
                image = np.array(pil_image)
            except ImportError:
                # Simple nearest neighbor resize fallback
                if image.shape[:2] != self._size:
                    # Simple downsampling - take every nth pixel
                    h_ratio = image.shape[0] // self._size[0]
                    w_ratio = image.shape[1] // self._size[1]
                    if h_ratio > 0 and w_ratio > 0:
                        image = image[::h_ratio, ::w_ratio]
                    # Crop or pad to exact size
                    if image.shape[0] > self._size[0]:
                        image = image[:self._size[0]]
                    if image.shape[1] > self._size[1]:
                        image = image[:, :self._size[1]]
                    # Pad if too small
                    if image.shape[0] < self._size[0] or image.shape[1] < self._size[1]:
                        new_image = np.zeros((*self._size, 3), dtype=image.dtype)
                        new_image[:image.shape[0], :image.shape[1]] = image
                        image = new_image
        
        image = image.astype(np.uint8)
        
        # Vehicle dynamics
        velocity = getattr(vehicle, 'velocity', [0, 0, 0])
        speed = np.linalg.norm(velocity[:2])  # 2D speed
        
        # Angular velocity (roll, pitch, yaw rates)
        angular_vel = getattr(vehicle, 'angular_velocity', [0, 0, 0])
        if not isinstance(angular_vel, np.ndarray):
            angular_vel = np.array(angular_vel, dtype=np.float32)
        
        # Current action state - 使用环境实际执行的动作（加偏置后）
        # 这样可以保证模型学习到正确的因果关系：实际执行的动作 → 观察到的结果
        steering = getattr(vehicle, 'steering', 0.0)
        throttle_brake = getattr(vehicle, 'throttle_brake_action', 0.0)
        
        # Navigation information
        route_completion = info.get('route_completion', 0.0)
        route_completion = np.clip(route_completion, 0.0, 1.0).astype(np.float32)
        
        # Calculate distance to route center (lane keeping metric)
        if hasattr(vehicle, 'lane') and vehicle.lane is not None:
            try:
                long, lat = vehicle.lane.local_coordinates(vehicle.position)
                distance_to_route = abs(lat)  # Lateral distance from lane center
            except:
                distance_to_route = 0.0
        else:
            distance_to_route = 0.0
            
        # Acceleration (approximate from speed change if not directly available)
        if hasattr(self, '_last_speed'):
            acceleration = (speed - self._last_speed) * 10.0  # Assuming 10Hz
        else:
            acceleration = 0.0
        self._last_speed = speed
        
        # Create observation dictionary
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
            'is_terminal': done and (info.get('crash', False) or info.get('out_of_road', False)),
        }
        # 仿照atari.py保存“想象帧”到result文件夹
        if not is_first and not done:
            try:
                from PIL import Image
                # target dir: current episode dir, fallback to frames_tmp_root
                save_dir = self._current_episode_dir if self._current_episode_dir is not None else self._frames_tmp_root
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f'imagine_{self._total_steps:06d}.png')
                Image.fromarray(image).save(fname)
            except Exception as e:
                print(f"[Save Image] Failed: {e}")
        return observation


    def _find_last_episode_dir(self):
        last_idx = max(0, self._episode_count - 1)
        src = os.path.join(self._frames_tmp_root, f'episode_{last_idx:06d}')
        if os.path.exists(src):
            return src, last_idx
        # fallback: find the latest by listing
        try:
            dirs = sorted([d for d in os.listdir(self._frames_tmp_root) if d.startswith('episode_')])
            if not dirs:
                return None, None
            last = dirs[-1]
            idx = int(last.split('_')[-1])
            return os.path.join(self._frames_tmp_root, last), idx
        except Exception:
            return None, None

    def render(self):
        """Render the environment"""
        return self._env.render(mode='rgb_array')

    def close(self):
        """Close the environment"""
        if hasattr(self, '_env'):
            self._env.close()