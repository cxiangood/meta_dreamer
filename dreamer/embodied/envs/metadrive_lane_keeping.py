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
            speed_reward=1.0,  # 降低速度奖励
            use_lateral_reward=True,  # 启用车道保持奖励
            
            # Termination settings - 使用悬崖模型：出界立即终止
            out_of_road_done=True,  # 出界立即终止episode（悬崖模型）
            crash_vehicle_done=False,  # 车辆碰撞继续（允许学习从错误中恢复）
            crash_object_done=False,  # 物体碰撞继续（允许学习从错误中恢复）
            
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
        # Ensure log file exists with header (best-effort)
        try:
            if not os.path.exists(self._reward_log_path):
                with open(self._reward_log_path, 'w') as f:
                    f.write('episode,total_reward,length,seed,reason\n')
        except Exception:
            pass
    
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
        
        # 1. 动作处理
        steering = np.clip(np.float32(action['steering']), -1.0, 1.0)
        throttle_brake = np.clip(np.float32(action['throttle_brake']), -1.0, 1.0)
        
        self._total_steps += 1
        
        md_action = [float(steering), float(throttle_brake)]
        # md_action_dict = {DEFAULT_AGENT: md_action}
        
        # # ⭐【核心修复】：直接构造 Numpy 数组，不带 ID 键
        # # 这样 MetaDrive 内部就不会去查找 'default_agent'，而是直接应用动作
        # md_action = np.array([steering, throttle_brake], dtype=np.float32)
        # 3. 执行环境步进
        total_reward = 0.0
        obs = None
        info = {}
        terminated = False
        truncated = False
        reward = 0.0
        time_penalty = 0.0
        speed_bonus = 0.0
        lane_keeping_reward = 0.0
        lane_penalty = 0.0
        waypoint_reward = 0.0
        throttle_bonus = 0.0
        steering_penalty = 0.0
        sharp_steering_penalty = 0.0
        action_consistency_bonus = 0.0
        
        # for _ in range(self._repeat):
        #     obs, reward, terminated, truncated, info = self._env.step(md_action)
        
        #     # 记录原始奖励
        #     original_reward = reward
            
        #     # 添加自定义奖励修正
        #     # 1. 时间惩罚：减小以避免过于焦虑
        #     time_penalty = -0.001  # 减小时间惩罚以避免策略追求快速终止
        #     reward += time_penalty
            
        #     # 2. 速度激励：适度奖励速度，但不要过强
        #     vehicle = self._env.agent
        #     velocity = getattr(vehicle, 'velocity', [0, 0, 0])
        #     speed = np.linalg.norm(velocity[:2])
        #     speed_bonus = max(0.0, min(1.0, speed / 10.0)) * 0.5  # 归一化速度并缩放
        #     reward += speed_bonus
            
        #     # 3. 平滑车道保持奖励（高斯衰减）和超出惩罚
        #     lane_keeping_reward = 0.0
        #     lane_penalty = 0.0
        #     if hasattr(vehicle, 'lane') and vehicle.lane is not None:
        #         try:
        #             long, lat = vehicle.lane.local_coordinates(vehicle.position)
        #             lateral_distance = abs(lat)
        #             # Smooth positive reward near center (Gaussian decay)
        #             lane_keeping_reward = 2.0 * np.exp(- (lateral_distance / 0.5) ** 2)
        #             # Mild penalty when leaving preferred band (>0.5)
        #             if lateral_distance > 0.5:
        #                 lane_penalty = -1.0 * (lateral_distance - 0.5)
        #             # Larger linear penalty for severe deviations
        #             if lateral_distance > 1.0:
        #                 lane_penalty += -1.5 * (lateral_distance - 1.0)
        #             reward += lane_keeping_reward + lane_penalty
        #         except:
        #             pass
            
        #     # 4. waypoint 到达奖励（基于 route_completion 的里程碑触达）
        #     waypoint_reward = 0.0
        #     try:
        #         route_completion = info.get('route_completion', 0.0)
        #         route_completion = np.clip(route_completion, 0.0, 1.0).astype(np.float32)
        #         prev_rc = getattr(self, '_last_route_completion', 0.0)
        #         prev_milestone = int(prev_rc * 10)
        #         cur_milestone = int(route_completion * 10)
        #         if cur_milestone > prev_milestone:
        #             waypoint_reward = 5.0 * (cur_milestone - prev_milestone)
        #             reward += waypoint_reward
        #         self._last_route_completion = route_completion
        #     except:
        #         pass

        #     # 5. 油门激励：轻微鼓励踩油门，但不可过强
        #     throttle_bonus = 0.0
        #     if throttle_brake > 0.2:
        #         throttle_bonus = throttle_brake * 0.2
        #         reward += throttle_bonus
            
        #     # 5. 转向惩罚：大幅加强转向惩罚，特别是大角度转向
        #     steering_penalty = 0.0
        #     # if abs(steering) > 0.3:  # 转向角度超过0.3
        #     #     steering_penalty = -2.0 * abs(steering)  # 强惩罚大角度转向
        #     #     reward += steering_penalty
        #     if abs(steering) > 0.1:  # 中等转向
        #         steering_penalty = -0.5 * (abs(steering)-0.1)  # 中等惩罚
        #         reward += steering_penalty
            
        #     # 6. 剧烈转向变化惩罚：防止急转弯
        #     sharp_steering_penalty = 0.0
        #     steering_change = abs(steering - self._last_steering)
        #     if steering_change > 0.3:  # 转向变化超过0.3视为剧烈转向
        #         sharp_steering_penalty = -2.0 * steering_change  # 强惩罚剧烈转向变化
        #         reward += sharp_steering_penalty
            
        #     # 7. 动作一致性奖励：轻微奖励动作平滑
        #     action_consistency_bonus = 0.0
        #     throttle_change = abs(throttle_brake - self._last_throttle_brake)
        #     if throttle_change < 0.2:  # 如果动作变化小，给予小奖励
        #         action_consistency_bonus = 0.1
        #         reward += action_consistency_bonus
        #     elif throttle_change > 0.5:  # 动作变化大，给予惩罚
        #         action_consistency_penalty = -0.2
        #         reward += action_consistency_penalty
            
        #     # 更新上一步的动作值
        #     self._last_throttle_brake = throttle_brake
        #     self._last_steering = steering
            
        #     # 6. 终止惩罚：确保智能体能学到"出界/碰撞很糟糕"
        #     termination_penalty = 0.0
        #     if terminated or truncated:
        #         # 根据终止原因给予不同的惩罚
        #         if info.get('crash', False) or info.get('crash_vehicle', False):
        #             termination_penalty = -200.0  # 碰撞的严重惩罚（增大）
        #             print(f"[Step {self._step_count}] *** CRASH! Penalty: {termination_penalty} ***")
        #         elif info.get('out_of_road', False):
        #             termination_penalty = -200.0  # 出界的严重惩罚（增大）
        #             print(f"[Step {self._step_count}] *** OUT OF ROAD! Penalty: {termination_penalty} ***")
        #         else:
        #             termination_penalty = -20.0  # 其他异常终止
        #             print(f"[Step {self._step_count}] *** EPISODE ENDED! Penalty: {termination_penalty} ***")
                
        #         reward += termination_penalty
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self._env.step(md_action)
            
            # 记录原始奖励
            original_reward = reward
            vehicle = self._env.agent
            
            # ---------------------------------------------------------
            # 自定义奖励修正逻辑
            # ---------------------------------------------------------
            
            # 1. 基础时间惩罚：维持极小值
            reward += -0.001 
            
            # 2. 进度与速度激励 (核心：驱动智能体往前走)
            velocity = getattr(vehicle, 'velocity', [0, 0, 0])
            speed = np.linalg.norm(velocity[:2])
            
            # A. 进度奖：根据本步沿车道线前进的距离（纵向位移）给奖
            try:
                current_long, current_lat = vehicle.lane.local_coordinates(vehicle.position)
                prev_long = getattr(self, '_last_long', current_long)
                # 只有向前走才给奖，dist_moved 单位通常是米
                dist_moved = current_long - prev_long
                if dist_moved > 0:
                    reward += dist_moved * 2.0  # 稠密进度奖励，权重可调
                self._last_long = current_long
            except:
                pass

            # B. 速度奖：鼓励维持在 5m/s - 15m/s 之间
            # 基础速度奖励
            reward += min(speed, 10.0) * 0.1 
            # 极低速惩罚：防止原地不动或过慢导致的控制失效
            if speed < 1.0:
                reward -= 0.1

            # 3. 改进的车道保持 (平滑中心带 + 边缘惩罚)
            if hasattr(vehicle, 'lane') and vehicle.lane is not None:
                try:
                    lateral_distance = abs(current_lat)
                    # 中心 0.2米内给予固定高分，不产生梯度，减少抖动
                    if lateral_distance < 0.2:
                        lane_keeping_reward = 2.0
                    else:
                        # 0.2米外开始高斯衰减
                        lane_keeping_reward = 2.0 * np.exp(- ((lateral_distance - 0.2) / 0.5) ** 2)
                    
                    # 只有真正快压线了（>1.0m）才开始线性惩罚
                    lane_penalty = 0.0
                    if lateral_distance > 1.0:
                        lane_penalty = -2.0 * (lateral_distance - 1.0)
                    
                    reward += lane_keeping_reward + lane_penalty
                except:
                    pass

            # 4. 转向惩罚 (大幅软化)
            steering_penalty = 0.0
            # 允许 0.1 以内的自然修正，不予惩罚
            if abs(steering) > 0.1:
                # 使用平方项，角度越大惩罚增长越快，但小角度惩罚很轻
                steering_penalty = -0.5 * ((abs(steering) - 0.1) ** 2)
                reward += steering_penalty

            # 5. 转向平滑度 (惩罚抖动)
            steering_change = abs(steering - self._last_steering)
            if steering_change > 0.2:
                # 只有动作突变时才重罚
                reward += -1.0 * steering_change

            # 6. 油门激励：仅在低速时鼓励踩油门
            if speed < 5.0 and throttle_brake > 0.2:
                reward += 0.1 * throttle_brake

            # ---------------------------------------------------------
            # 更新状态记录
            self._last_throttle_brake = throttle_brake
            self._last_steering = steering
            
            # 7. 终止惩罚 (保持高压，但需要进度奖励来对冲)
            if terminated or truncated:
                if info.get('crash', False) or info.get('crash_vehicle', False):
                    termination_penalty = -200.0
                    print(f"[Step {self._step_count}] *** CRASH! ***")
                elif info.get('out_of_road', False):
                    termination_penalty = -200.0
                    print(f"[Step {self._step_count}] *** OUT OF ROAD! ***")
                elif info.get('arrive_dest', False):
                    termination_penalty = 50.0 # 到达终点给大奖
                    print(f"[Step {self._step_count}] *** ARRIVED! ***")
                else:
                    termination_penalty = -10.0
                
                reward += termination_penalty

            # 打印详细的奖励信息（每50步打印一次以避免刷屏）
            if self._step_count % 50 == 0:
                print(f"[Step {self._step_count}] Speed: {speed:.2f}m/s, Reward: {reward:.2f}")
                print(f"  Components: time={time_penalty:.3f}, speed={speed_bonus:.3f}, lane={lane_keeping_reward:.3f}, lane_penalty={lane_penalty:.3f}, waypoint={waypoint_reward:.3f}, throttle={throttle_bonus:.3f}")
                print(f"  Penalties: steering={steering_penalty:.2f}, sharp_turn={sharp_steering_penalty:.2f}, consistency={action_consistency_bonus:.2f}")
                if hasattr(vehicle, 'lane') and vehicle.lane is not None:
                    try:
                        long, lat = vehicle.lane.local_coordinates(vehicle.position)
                        print(f"  Lane info: lateral_dist={abs(lat):.2f}")
                    except:
                        pass
            
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
                # 打印episode总结（简化版本）
                print(f"\n{'='*60}")
                print(f"[Episode End] Steps: {self._step_count}, Reward: {total_reward:.2f}")
                reason = "CRASH" if info.get('crash', False) else ("OUT_OF_ROAD" if info.get('out_of_road', False) else "OTHER")
                print(f"[Episode End] Reason: {reason}, Speed: {speed:.2f}m/s")
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
        except Exception as e:
            error_msg = str(e)
            # 检查是否是资源下载失败
            if "Connection timed out" in error_msg or "URLError" in str(type(e).__name__):
                print(f"\n{'='*60}")
                print("[ERROR] MetaDrive 资源下载失败（网络连接超时）")
                print(f"{'='*60}")
                print("解决方案：")
                print("1. 手动下载资源文件：")
                print("   cd /home/yajiao-xu/meta_dreamer/metadrive-main")
                print("   python -m metadrive.pull_asset")
                print("")
                print("2. 或者检查网络连接和防火墙设置")
                print("3. 如果使用代理，请配置代理环境变量")
                print(f"{'='*60}\n")
                raise RuntimeError(
                    "MetaDrive 资源下载失败。请手动运行 'python -m metadrive.pull_asset' 下载资源文件。"
                ) from e
            
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
                save_dir = r'/share/home/u23516/code/meta_dreamer-main/logs/result'
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f'imagine_{self._total_steps:06d}.png')
                Image.fromarray(image).save(fname)
            except Exception as e:
                print(f"[Save Image] Failed: {e}")
        return observation

    def render(self):
        """Render the environment"""
        return self._env.render(mode='rgb_array')

    def close(self):
        """Close the environment"""
        if hasattr(self, '_env'):
            self._env.close()