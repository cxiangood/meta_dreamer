import socket
import json
import time
import numpy as np
from typing import Optional, Any

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    print("Warning: CARLA not installed. This will cause runtime errors.")
    # 创建一个假的carla模块以避免类型检查错误
    class FakeCarla:
        def __getattr__(self, name):
            raise ImportError(f"CARLA not available. Cannot access carla.{name}")
    carla = FakeCarla()  # type: ignore
    CARLA_AVAILABLE = False

HOST = '127.0.0.1'
PORT = 50037
IMAGE_SIZE = (64, 64)
FPS = 10

class CarlaLaneKeepingServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen(1)
        
        # CARLA对象 - 使用Any类型避免类型检查错误
        self.vehicle: Any = None
        self.world: Any = None
        self.camera_sensor: Any = None
        self.collision_sensor: Any = None
        self.client_carla: Any = None
        self.blueprint_library: Any = None
        
        self.last_image: Optional[list] = None
        self.done = True
        
        # 奖励函数相关变量
        self.collision_flag = False
        self.last_location: Any = None
        self.episode_distance = 0.0
        self.max_speed = 30.0  # km/h
        self.target_speed = 20.0  # km/h
        
        self._connect_to_carla()
        print(f"[Carla0915] Server started at {host}:{port}")

    def _connect_to_carla(self):
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA is not available. Please install CARLA first.")
        self.client_carla = carla.Client('localhost', 2000)  # type: ignore
        self.client_carla.set_timeout(10.0)
        self.world = self.client_carla.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

    def _spawn_vehicle(self):
        vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            print(f"[Carla0915] ERROR: Vehicle spawn failed! spawn_point={spawn_point}, vehicle_bp={vehicle_bp}")
            raise Exception("Carla vehicle spawn failed. Check Carla server, map, and spawn point availability.")
        self.vehicle.set_autopilot(False)
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        back_vec = carla.Location(x=-8, y=0)
        cam_location = transform.location + back_vec + carla.Location(z=8)
        cam_rotation = carla.Rotation(pitch=-60, yaw=transform.rotation.yaw)
        spectator.set_transform(carla.Transform(cam_location, cam_rotation))

    def _setup_camera(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMAGE_SIZE[1]))
        camera_bp.set_attribute('image_size_y', str(IMAGE_SIZE[0]))
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(self._on_camera_data)
        
    def _setup_collision_sensor(self):
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)
        
    def _on_collision(self, event):
        self.collision_flag = True
        print(f"[Carla0915] Collision detected with {event.other_actor.type_id}")

    def _on_camera_data(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.last_image = array.tolist()

    def reset(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
            
        # 重置奖励相关变量
        self.collision_flag = False
        self.last_location = None
        self.episode_distance = 0.0
        
        self._spawn_vehicle()
        self._setup_camera()
        self._setup_collision_sensor()
        self.done = False
        time.sleep(0.2)
        return self._get_obs()

    def step(self, action):
        steer = float(np.clip(action.get('steer', 0.0), -1.0, 1.0))
        acc = float(np.clip(action.get('acc', 0.0), -1.0, 1.0))
        throttle = acc if acc > 0 else 0.0
        brake = -acc if acc < 0 else 0.0
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        self.vehicle.apply_control(control)
        self.world.tick()
        
        # 更新位置用于计算前进奖励
        current_location = self.vehicle.get_location()
        if self.last_location is not None:
            distance = current_location.distance(self.last_location)
            self.episode_distance += distance
        self.last_location = current_location
        
        # 检查是否需要终止episode (除了碰撞外的其他条件)
        self._check_termination_conditions()
        
        time.sleep(1.0 / FPS)
        return self._get_obs()
    
    def _check_termination_conditions(self):
        """检查episode终止条件"""
        try:
            # 检查是否偏离道路太远
            vehicle_location = self.vehicle.get_location()
            waypoint = self.world.get_map().get_waypoint(
                vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            
            if waypoint:
                distance_to_road = vehicle_location.distance(waypoint.transform.location)
                if distance_to_road > 5.0:  # 偏离道路5米以上
                    self.done = True
                    print("[Carla0915] Episode terminated: vehicle too far from road")
            
            # 检查是否长时间停止不动
            velocity = self.vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            if speed < 0.5:  # 速度小于0.5 m/s
                if not hasattr(self, '_stationary_steps'):
                    self._stationary_steps = 0
                self._stationary_steps += 1
                if self._stationary_steps > 100:  # 停止超过10秒 (100 steps * 0.1s)
                    self.done = True
                    print("[Carla0915] Episode terminated: vehicle stationary too long")
            else:
                self._stationary_steps = 0
                
        except Exception as e:
            print(f"[Carla0915] Error in termination check: {e}")

    def _get_obs(self):
        image = self.last_image if self.last_image is not None else [[0]*3]*IMAGE_SIZE[0]*IMAGE_SIZE[1]
        velocity = self.vehicle.get_velocity()
        speed_ms = float(np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        speed_kmh = speed_ms * 3.6  # Convert to km/h
        
        # 计算奖励
        reward = self._calculate_reward(speed_kmh)
        
        obs = {
            'image': image,
            'speed': speed_kmh,
            'reward': reward,
            'done': self.done
        }
        return obs
    
    def _calculate_reward(self, speed_kmh):
        """
        计算基于多个因素的综合奖励函数
        """
        reward = 0.0
        
        # 1. 碰撞惩罚 (最高优先级)
        if self.collision_flag:
            reward -= 100.0
            self.done = True
            print("[Carla0915] Episode terminated due to collision")
            return reward
        
        # 2. 速度奖励 - 鼓励维持目标速度
        if speed_kmh > 0:
            # 速度在目标范围内给予奖励，超出或过低则惩罚
            if 15 <= speed_kmh <= 25:  # 目标速度范围
                reward += 1.0
            elif speed_kmh < 5:  # 停车或过慢惩罚
                reward -= 0.5
            elif speed_kmh > 35:  # 超速惩罚
                reward -= 0.3
            else:
                # 渐变奖励，距离目标速度越近奖励越高
                speed_diff = abs(speed_kmh - self.target_speed)
                reward += max(0, 1.0 - speed_diff / 10.0)
        
        # 3. 车道保持奖励
        lane_reward = self._calculate_lane_keeping_reward()
        reward += lane_reward
        
        # 4. 前进奖励 - 鼓励持续前进
        progress_reward = self._calculate_progress_reward()
        reward += progress_reward
        
        # 5. 方向一致性奖励
        heading_reward = self._calculate_heading_reward()
        reward += heading_reward
        
        # 6. 生存奖励 - 每步基础奖励
        reward += 0.1
        
        return float(reward)
    
    def _calculate_lane_keeping_reward(self):
        """计算车道保持奖励"""
        try:
            vehicle_location = self.vehicle.get_location()
            waypoint = self.world.get_map().get_waypoint(
                vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            
            if waypoint:
                # 计算车辆到车道中心的距离
                lane_center = waypoint.transform.location
                distance_to_center = vehicle_location.distance(lane_center)
                
                # 车道保持奖励，距离中心越近奖励越高
                if distance_to_center < 1.0:  # 在车道中心1米内
                    return 0.5
                elif distance_to_center < 2.0:  # 在车道中心2米内
                    return 0.2
                elif distance_to_center > 3.5:  # 偏离车道太远
                    return -1.0
                else:
                    return -0.2
            return 0.0
        except:
            return 0.0
    
    def _calculate_progress_reward(self):
        """计算前进奖励"""
        try:
            current_location = self.vehicle.get_location()
            if self.last_location is not None:
                distance = current_location.distance(self.last_location)
                self.episode_distance += distance
                # 前进距离奖励
                return min(distance * 0.1, 0.5)  # 最大0.5的前进奖励
            else:
                self.last_location = current_location
                return 0.0
        except:
            return 0.0
    
    def _calculate_heading_reward(self):
        """计算方向一致性奖励"""
        try:
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            
            waypoint = self.world.get_map().get_waypoint(
                vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            
            if waypoint:
                # 计算车辆朝向与道路方向的差异
                vehicle_yaw = vehicle_transform.rotation.yaw
                road_yaw = waypoint.transform.rotation.yaw
                
                # 计算角度差异 (-180 到 180)
                yaw_diff = ((vehicle_yaw - road_yaw + 180) % 360) - 180
                yaw_diff = abs(yaw_diff)
                
                # 方向一致性奖励
                if yaw_diff < 10:  # 角度差小于10度
                    return 0.3
                elif yaw_diff < 30:  # 角度差小于30度
                    return 0.1
                elif yaw_diff > 90:  # 方向完全错误
                    return -0.5
                else:
                    return -0.1
            return 0.0
        except:
            return 0.0

    def serve(self):
        while True:
            print("[Carla0915] Waiting for DreamerV3 client...")
            conn, addr = self.server.accept()
            print(f"[Carla0915] Connected by {addr}")
            try:
                while True:
                    data = conn.recv(40960)
                    if not data:
                        break
                    msg = json.loads(data.decode())
                    if msg['cmd'] == 'reset':
                        obs = self.reset()
                        conn.sendall(json.dumps({'obs': obs}).encode())
                    elif msg['cmd'] == 'step':
                        obs = self.step(msg['action'])
                        conn.sendall(json.dumps({'obs': obs}).encode())
                    else:
                        conn.sendall(json.dumps({'error': 'Unknown command'}).encode())
            except Exception as e:
                print(f"[Carla0915] Client error: {e}")
            finally:
                conn.close()

if __name__ == '__main__':
    server = CarlaLaneKeepingServer(HOST, PORT)
    server.serve()
