import functools
import os
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
        
        # 检查是否需要渲染（通过环境变量控制）
        # 只有第一个环境会渲染，通过检查是否已经有渲染实例来判断
        enable_render = os.environ.get('METADRIVE_RENDER', '0') == '1'
        
        # 使用类变量来跟踪是否已经创建了渲染实例
        if not hasattr(MetaDriveLaneKeeping, '_render_instance_created'):
            MetaDriveLaneKeeping._render_instance_created = False
        
        # 只允许第一个实例渲染
        if enable_render and not MetaDriveLaneKeeping._render_instance_created:
            use_render = True
            MetaDriveLaneKeeping._render_instance_created = True
            print(f"[MetaDrive] Enabling rendering for this environment instance")
        else:
            use_render = False
            if enable_render:
                print(f"[MetaDrive] Rendering disabled for this instance (only first instance renders)")
        
        # MetaDrive environment configuration for lane keeping
        # Use the correct MetaDrive configuration approach
        config = dict(
            # Basic environment settings
            use_render=use_render,  # 根据上面的逻辑决定是否渲染
            num_scenarios=1000,
            start_seed=0,
            horizon=length,
            
            # Map configuration - 使用更长的地图
            map=30,  # 使用30号地图，比较长（1-99可选，数字越大路段越长）
            
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
            ),
            
            # ⭐⭐⭐ Agent配置 - 特定agent的配置（包括spawn位置）
            agent_configs={
                DEFAULT_AGENT: dict(
                    spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),  # 最右侧车道
                )
            },
            
            # Use sensors dictionary for camera
            sensors=dict(
                rgb_camera=(RGBCamera, size[0], size[1]),
            ),
            
            # Reward configuration
            success_reward=100.0,
            out_of_road_penalty=10.0,
            crash_vehicle_penalty=20.0,
            crash_object_penalty=10.0,
            driving_reward=10.0,  # 提高以鼓励前进
            speed_reward=3.0,  # 降低避免行为震荡
            use_lateral_reward=True,
            
            # Termination settings
            out_of_road_done=True,
            crash_vehicle_done=True,
            crash_object_done=True,
            
        )
        
        # Update config with user provided kwargs
        config.update(kwargs)

        # Ensure manual_control is OFF by default unless explicitly enabled
        # either via kwargs or the METADRIVE_MANUAL_CONTROL environment variable.
        if 'manual_control' not in config:
            config['manual_control'] = False
        if os.environ.get('METADRIVE_MANUAL_CONTROL') == '1':
            config['manual_control'] = True
        
        # 保存配置以便在需要时重新创建环境
        self._base_config = config.copy()
        
        self._env = MetaDriveEnv(config)
        self._done = True
        self._step_count = 0
        self._last_throttle_brake = 0.0  # 记录上一步的油门/刹车动作
        self._last_steering = 0.0  # 记录上一步的转向动作
        self._total_steps = 0  # 记录总步数，用于探索偏置衰减
    
    def _create_env_headless(self):
        """创建无渲染的环境（当渲染窗口被关闭时使用）"""
        print("[MetaDrive] Creating headless environment...")
        # 复制配置并禁用渲染
        config = self._base_config.copy()
        config['use_render'] = False
        config['manual_control'] = False
        
        # 创建新的环境实例
        self._env = MetaDriveEnv(config)
        print("[MetaDrive] Headless environment created successfully")
        
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
        # 只在重置时打印日志，减少输出
        if action['reset'] or self._done:
            print(f"[RESET TRIGGER] Episode ended, resetting... (reset_flag={action['reset']}, done={self._done})")
            return self._reset()
        
        # Convert action to MetaDrive format，强制转换为float32类型
        steering = np.float32(action['steering'])
        throttle_brake = np.float32(action['throttle_brake'])
        
        # 探索偏置：在训练早期阶段，将动作偏向加速
        # 前10000步逐渐衰减这个偏置
        exploration_bias = max(0.0, 1.0 - self._total_steps / 10000.0)
        if exploration_bias > 0.01:  # 如果还在探索阶段
            # 将油门/刹车值向正方向（加速）偏移
            # 偏移量会随着训练进行而逐渐减小
            bias_strength = 1 * exploration_bias  # 最大偏移1，逐渐衰减到0
            throttle_brake = throttle_brake + bias_strength
            
            # 如果偏移后仍然是负值（刹车），进一步减弱刹车力度
            if throttle_brake < 0:
                throttle_brake = throttle_brake * (1.0 - exploration_bias * 0.5)
        
        # Clip values to action space bounds（这是实际执行的动作，也是应该记录的动作）
        steering = np.clip(steering, -1.0, 1.0)
        throttle_brake = np.clip(throttle_brake, -1.0, 1.0)
        
        # 每1000步打印一次探索偏置状态
        if self._total_steps % 1000 == 0 and exploration_bias > 0.01:
            print(f"[Exploration Bias] Total steps: {self._total_steps}, "
                  f"Bias strength: {exploration_bias:.3f}, "
                  f"Current throttle: {throttle_brake:.3f}")
        
        self._total_steps += 1
        
        md_action = [float(steering), float(throttle_brake)]
        
        # Execute action with frame repeat
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self._env.step(md_action)
            
            # 记录原始奖励
            original_reward = reward
            
            # 添加自定义奖励修正
            # 1. 时间惩罚：减小以避免过于焦虑
            time_penalty = -0.02
            reward += time_penalty
            
            # 2. 速度激励：大幅提高，让高速的好处更明显
            vehicle = self._env.agent
            velocity = getattr(vehicle, 'velocity', [0, 0, 0])
            speed = np.linalg.norm(velocity[:2])
            speed_bonus = speed * 2.0  # 提高速度奖励系数
            reward += speed_bonus
            
            # 3. 油门激励：鼓励踩油门
            throttle_bonus = 0.0
            if throttle_brake > 0:  # 踩油门（正值）
                throttle_bonus = throttle_brake * 1.0  # 提高油门奖励
                reward += throttle_bonus
            
            # 4. 动作一致性奖励：惩罚频繁改变油门/刹车
            action_consistency_bonus = 0.0
            throttle_change = abs(throttle_brake - self._last_throttle_brake)
            if throttle_change < 0.3:  # 如果动作变化小，给予奖励
                action_consistency_bonus = 0.2
                reward += action_consistency_bonus
            else:  # 动作变化大，给予惩罚
                action_consistency_penalty = -0.3
                reward += action_consistency_penalty
                action_consistency_bonus = action_consistency_penalty
            
            # 4.5. 剧烈转向惩罚：防止车辆急转弯导致失控
            sharp_steering_penalty = 0.0
            steering_change = abs(steering - self._last_steering)
            if steering_change > 0.5:  # 转向变化超过0.5视为剧烈转向
                sharp_steering_penalty = -1.0 * steering_change  # 惩罚力度与转向幅度成正比
                reward += sharp_steering_penalty
            elif abs(steering) > 0.7:  # 持续大角度转向也给予惩罚
                sharp_steering_penalty = -0.5 * abs(steering)
                reward += sharp_steering_penalty
            
            # 5. 持续加速奖励：如果连续踩油门，额外奖励
            acceleration_bonus = 0.0
            if throttle_brake > 0.5 and self._last_throttle_brake > 0.5:
                acceleration_bonus = 0.5
                reward += acceleration_bonus
            
            # 更新上一步的动作值
            self._last_throttle_brake = throttle_brake
            self._last_steering = steering
            
            # 6. 终止惩罚：确保智能体能学到"出界/碰撞很糟糕"
            termination_penalty = 0.0
            if terminated or truncated:
                # 根据终止原因给予不同的惩罚
                if info.get('crash', False) or info.get('crash_vehicle', False):
                    termination_penalty = -50.0  # 碰撞的严重惩罚
                    print(f"[Step {self._step_count}] *** CRASH! Penalty: {termination_penalty} ***")
                elif info.get('out_of_road', False):
                    termination_penalty = -30.0  # 出界的严重惩罚
                    print(f"[Step {self._step_count}] *** OUT OF ROAD! Penalty: {termination_penalty} ***")
                else:
                    termination_penalty = -20.0  # 其他异常终止
                    print(f"[Step {self._step_count}] *** EPISODE ENDED! Penalty: {termination_penalty} ***")
                
                reward += termination_penalty
            
            # 打印详细的奖励信息（每50步打印一次以避免刷屏）
            if self._step_count % 50 == 0:
                # 计算真实加速度（速度变化率）
                if hasattr(self, '_last_step_speed'):
                    real_acceleration = (speed - self._last_step_speed) / (1.0 / 10.0)  # 假设10Hz
                else:
                    real_acceleration = 0.0
                self._last_step_speed = speed
                
                print(f"[Step {self._step_count}] Speed: {speed:.2f}m/s, Reward: {reward:.2f}")
            
            total_reward += reward
            self._step_count += 1
            
            if terminated or truncated:
                self._done = True
                # 打印episode总结（简化版本）
                print(f"\n{'='*60}")
                print(f"[Episode End] Steps: {self._step_count}, Reward: {total_reward:.2f}")
                reason = "CRASH" if info.get('crash', False) else ("OUT_OF_ROAD" if info.get('out_of_road', False) else "OTHER")
                print(f"[Episode End] Reason: {reason}, Speed: {speed:.2f}m/s")
                print(f"{'='*60}\n")
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
        
        # 重置速度追踪（用于加速度计算）
        if hasattr(self, '_last_step_speed'):
            delattr(self, '_last_step_speed')
        if hasattr(self, '_last_speed'):
            delattr(self, '_last_speed')
        
        seed = self._random.randint(0, 1000)  # MetaDrive requires seed in [0:1000)
        
        try:
            obs, info = self._env.reset(seed=seed)
            print(f"[RESET] ✓ Episode started with map seed {seed}")
        except Exception as e:
            # 如果重置失败（例如窗口被关闭），重新创建环境
            print(f"[RESET] ✗ Reset failed: {type(e).__name__}, recreating environment...")
            
            # 关闭旧环境（如果还存在）
            try:
                self._env.close()
            except Exception as close_error:
                pass
            
            # 重新创建环境，使用 headless 模式（无渲染）
            try:
                self._create_env_headless()
                obs, info = self._env.reset(seed=seed)
                print(f"[RESET] ✓ Environment recreated in headless mode, seed {seed}")
            except Exception as recreate_error:
                print(f"[RESET] ✗ Failed to recreate: {recreate_error}")
                raise
        
        return self._get_obs(obs, 0.0, info, is_first=True)

    def _get_obs(self, obs, reward, info, done=False, is_first=False):
        """Convert MetaDrive observation to DreamerV3 format"""
        # Get vehicle state
        vehicle = self._env.agent
        
        # Extract image observation
        if 'image' in obs:
            image = obs['image']
            if len(image.shape) == 3 and image.shape[-1] == 4:  # RGBA to RGB
                image = image[..., :3]
        else:
            # Fallback: render the environment if no camera sensor
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
                pil_image = pil_image.resize(self._size, PILImage.LANCZOS)
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
            'is_terminal': done and info.get('crash', False),
        }
        
        return observation

    def render(self):
        """Render the environment"""
        return self._env.render(mode='rgb_array')

    def close(self):
        """Close the environment"""
        if hasattr(self, '_env'):
            self._env.close()