# dreamerv3/embodied/envs/carla.py
import os
import time
import threading
import collections
from dataclasses import dataclass

import carla
import numpy as np
import elements
import embodied

from PIL import Image
import cv2

@dataclass
class CarlaConfig:
    # 环境配置
    town: str = "Town03"
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    
    # 车辆和传感器配置
    vehicle_model: str = "model3"
    camera_width: int = 800
    camera_height: int = 600
    camera_fov: int = 90
    show_sensor_data: bool = False
    
    # 任务配置
    max_steps: int = 1000
    target_speed: float = 50.0  # km/h
    waypoint_distance: float = 2.0
    
    # 观测配置
    image_size: tuple = (84, 84)
    gray_scale: bool = True
    include_state: bool = True
    
    # 动作配置
    continuous: bool = True
    action_repeat: int = 2
    
    # 奖励配置
    reward_scale: float = 1.0
    collision_penalty: float = -10.0
    off_road_penalty: float = -5.0
    speed_reward_weight: float = 0.1
    steering_penalty_weight: float = 0.01

class CarlaEnv(embodied.Env):
    LOCK = threading.Lock()
    
    def __init__(self, config: CarlaConfig = None, seed=None):
        self.config = config or CarlaConfig()
        self.rng = np.random.default_rng(seed)
        
        # 初始化CARLA客户端和世界
        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.camera = None
        self.camera_image = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
        # 环境状态
        self.waypoints = None
        self.current_waypoint_idx = 0
        self.collision_occurred = False
        self.off_road = False
        self.step_count = 0
        self.episode_reward = 0.0
        self.done = True
        
        # 连接到CARLA服务器
        self._connect()
        
        # 动作空间映射
        if self.config.continuous:
            self.action_mapping = self._continuous_action_mapping
        else:
            self.action_mapping = self._discrete_action_mapping
            self._define_discrete_actions()
    
    @property
    def obs_space(self):
        spaces = {
            'image': elements.Space(
                np.uint8, 
                (*self.config.image_size, 1 if self.config.gray_scale else 3)
            ),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
        
        if self.config.include_state:
            spaces.update({
                'speed': elements.Space(np.float32),
                'steering': elements.Space(np.float32),
                'throttle': elements.Space(np.float32),
                'distance_to_waypoint': elements.Space(np.float32),
                'angle_to_waypoint': elements.Space(np.float32),
            })
        
        return spaces
    
    @property
    def act_space(self):
        if self.config.continuous:
            return {
                'action': elements.Space(
                    np.float32, 
                    (2,),  # 油门/刹车和转向
                    -1.0, 1.0
                ),
                'reset': elements.Space(bool),
            }
        else:
            return {
                'action': elements.Space(
                    np.int32, 
                    (), 
                    0, len(self.discrete_actions)
                ),
                'reset': elements.Space(bool),
            }
    
    def _connect(self):
        """连接到CARLA服务器并初始化环境"""
        with self.LOCK:
            self.client = carla.Client(self.config.host, self.config.port)
            self.client.set_timeout(self.config.timeout)
            
            # 加载指定的地图
            self.world = self.client.load_world(self.config.town)
            self.map = self.world.get_map()
            
            # 设置同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            
            # 获取初始航点
            self.waypoints = self._generate_waypoints()
    
    def _generate_waypoints(self):
        """生成导航路径的航点"""
        spawn_points = self.map.get_spawn_points()
        start_point = spawn_points[self.rng.integers(len(spawn_points))]
        
        waypoints = []
        current_waypoint = self.map.get_waypoint(start_point.location)
        
        # 生成一系列航点
        for _ in range(100):  # 生成100个航点
            waypoints.append(current_waypoint)
            next_waypoints = current_waypoint.next(self.config.waypoint_distance)
            if not next_waypoints:
                break
            current_waypoint = next_waypoints[0]
            
        return waypoints
    
    def _setup_vehicle_and_sensors(self):
        """设置车辆和传感器"""
        # 清除现有车辆
        if self.vehicle:
            self.vehicle.destroy()
        
        # 清除现有传感器
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        
        # 生成车辆
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self.config.vehicle_model)[0]
        spawn_point = self.waypoints[0].transform
        spawn_point.location.z += 0.5  # 稍微抬高一点避免碰撞
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # 设置相机传感器
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.config.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.config.camera_height))
        camera_bp.set_attribute('fov', str(self.config.camera_fov))
        
        # 相机位置（车辆前方）
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4), 
            carla.Rotation(pitch=-15)
        )
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        
        # 监听相机图像
        self.camera.listen(lambda image: self._process_camera_image(image))
        
        # 设置碰撞传感器
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        # 设置车道入侵传感器
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.lane_invasion_sensor.listen(
            lambda event: self._on_lane_invasion(event)
        )
        
        # 设置初始状态
        self.current_waypoint_idx = 0
        self.collision_occurred = False
        self.off_road = False
        self.step_count = 0
        self.episode_reward = 0.0
    
    def _process_camera_image(self, image):
        """处理相机图像"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 移除alpha通道
        array = array[:, :, ::-1]  # 转换为RGB格式
        self.camera_image = array
    
    def _on_collision(self, event):
        """处理碰撞事件"""
        self.collision_occurred = True
    
    def _on_lane_invasion(self, event):
        """处理车道入侵事件"""
        for marking in event.crossed_lane_markings:
            if str(marking.type).startswith("OffRoad"):
                self.off_road = True
                break
    
    def _define_discrete_actions(self):
        """定义离散动作集"""
        self.discrete_actions = [
            (0.0, 0.0),   # 不动
            (0.5, 0.0),   # 轻微加速
            (1.0, 0.0),   # 全力加速
            (0.0, -0.5),  # 轻微左转
            (0.0, 0.5),   # 轻微右转
            (0.0, -1.0),  # 大幅左转
            (0.0, 1.0),   # 大幅右转
            (-0.5, 0.0),  # 轻微刹车
            (-1.0, 0.0)   # 全力刹车
        ]
    
    def _continuous_action_mapping(self, action):
        """将连续动作映射到车辆控制"""
        # 动作格式: [油门/刹车, 转向]
        # 油门/刹车: [-1, 1] -> 刹车到油门
        # 转向: [-1, 1] -> 左转到右转
        
        throttle = max(0.0, action[0])
        brake = max(0.0, -action[0])
        steer = action[1]
        
        return carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer,
            hand_brake=False,
            reverse=False
        )
    
    def _discrete_action_mapping(self, action_idx):
        """将离散动作映射到车辆控制"""
        throttle_brake, steer = self.discrete_actions[action_idx]
        
        throttle = max(0.0, throttle_brake)
        brake = max(0.0, -throttle_brake)
        
        return carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer,
            hand_brake=False,
            reverse=False
        )
    
    def step(self, action):
        if action['reset'] or self.done:
            self._reset()
            return self._get_observation(0.0, is_first=True)
        
        # 处理动作
        control = self.action_mapping(action['action'])
        
        # 应用动作并更新世界
        self.vehicle.apply_control(control)
        self.world.tick()
        
        # 计算奖励
        reward = self._calculate_reward(control)
        
        # 检查终止条件
        self.step_count += 1
        self.episode_reward += reward
        
        terminal = False
        is_last = False
        
        if self.collision_occurred:
            terminal = True
            is_last = True
        elif self.off_road:
            terminal = True
            is_last = True
        elif self.step_count >= self.config.max_steps:
            is_last = True
        elif self.current_waypoint_idx >= len(self.waypoints) - 1:
            # 到达最后一个航点
            reward += 10.0  # 到达目标奖励
            is_last = True
        
        self.done = is_last
        
        return self._get_observation(reward, is_last=is_last, is_terminal=terminal)
    
    def _reset(self):
        """重置环境"""
        with self.LOCK:
            # 重新生成航点
            self.waypoints = self._generate_waypoints()
            
            # 设置车辆和传感器
            self._setup_vehicle_and_sensors()
            
            # 确保有相机图像
            while self.camera_image is None:
                self.world.tick()
                time.sleep(0.1)
            
            self.done = False
            return True
    
    def _calculate_reward(self, control):
        """计算奖励"""
        # 获取车辆状态
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed * 3.6  # 转换为km/h
        
        # 获取当前位置和目标航点
        vehicle_transform = self.vehicle.get_transform()
        current_location = vehicle_transform.location
        current_waypoint = self.map.get_waypoint(current_location)
        
        # 更新当前航点索引
        while (self.current_waypoint_idx < len(self.waypoints) - 1 and
               current_waypoint.distance(self.waypoints[self.current_waypoint_idx].transform.location) < 
               self.config.waypoint_distance / 2):
            self.current_waypoint_idx += 1
        
        target_waypoint = self.waypoints[self.current_waypoint_idx]
        
        # 计算到目标航点的距离和角度
        distance = current_location.distance(target_waypoint.transform.location)
        waypoint_direction = target_waypoint.transform.get_forward_vector()
        vehicle_direction = vehicle_transform.get_forward_vector()
        
        # 计算方向角差异（奖励沿着航点方向行驶）
        dot_product = (waypoint_direction.x * vehicle_direction.x +
                      waypoint_direction.y * vehicle_direction.y +
                      waypoint_direction.z * vehicle_direction.z)
        angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # 基础奖励：跟随航点
        reward = (1.0 - distance / self.config.waypoint_distance) * 0.5
        reward += (1.0 - angle_diff / np.pi) * 0.5
        
        # 速度奖励：接近目标速度
        speed_reward = 1.0 - abs(speed_kmh - self.config.target_speed) / self.config.target_speed
        reward += speed_reward * self.config.speed_reward_weight
        
        # 惩罚过度转向
        reward -= abs(control.steer) * self.config.steering_penalty_weight
        
        # 碰撞惩罚
        if self.collision_occurred:
            reward += self.config.collision_penalty
        
        # 偏离道路惩罚
        if self.off_road:
            reward += self.config.off_road_penalty
        
        # 缩放奖励
        return reward * self.config.reward_scale
    
    def _get_observation(self, reward, is_first=False, is_last=False, is_terminal=False):
        """获取观测数据"""
        # 处理图像
        image = self.camera_image.copy()
        
        # 调整图像大小
        if self.config.resize == 'opencv':
            image = cv2.resize(image, self.config.image_size, interpolation=cv2.INTER_AREA)
        else:
            image = Image.fromarray(image)
            image = image.resize(self.config.image_size, Image.BILINEAR)
            image = np.array(image)
        
        # 转换为灰度图
        if self.config.gray_scale:
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            image = image.astype(np.uint8)
            image = np.expand_dims(image, -1)
        
        # 构建观测字典
        observation = {
            'image': image,
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }
        
        # 添加车辆状态信息
        if self.config.include_state:
            velocity = self.vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed * 3.6  # 转换为km/h
            
            target_waypoint = self.waypoints[self.current_waypoint_idx]
            distance = self.vehicle.get_location().distance(target_waypoint.transform.location)
            
            # 计算与目标航点的角度差
            vehicle_transform = self.vehicle.get_transform()
            waypoint_direction = target_waypoint.transform.get_forward_vector()
            vehicle_direction = vehicle_transform.get_forward_vector()
            dot_product = (waypoint_direction.x * vehicle_direction.x +
                          waypoint_direction.y * vehicle_direction.y +
                          waypoint_direction.z * vehicle_direction.z)
            angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            observation.update({
                'speed': np.float32(speed_kmh),
                'steering': np.float32(self.vehicle.get_control().steer),
                'throttle': np.float32(self.vehicle.get_control().throttle),
                'distance_to_waypoint': np.float32(distance),
                'angle_to_waypoint': np.float32(angle_diff),
            })
        
        return observation
    
    def close(self):
        """关闭环境并清理资源"""
        with self.LOCK:
            if self.camera:
                self.camera.destroy()
            if self.collision_sensor:
                self.collision_sensor.destroy()
            if self.lane_invasion_sensor:
                self.lane_invasion_sensor.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            
            # 恢复世界设置
            if self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

