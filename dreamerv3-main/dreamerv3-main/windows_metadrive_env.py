
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

class WindowsMetaDriveLaneKeeping:
    """
    Simplified MetaDrive Lane Keeping Environment for Windows
    """
    
    def __init__(self, size=(64, 64), length=1000):
        self._size = size
        self._length = length
        self._step_count = 0
        self._done = True
        
        config = dict(
            use_render=False,
            num_scenarios=1000,
            start_seed=0,
            horizon=length,
            map=7,  # Use integer for block number
            # map_config will be auto-generated for integer maps
            traffic_density=0.1,
            image_observation=True,  # Move to top level  
            vehicle_config=dict(
                show_lidar=False,
                show_navi_mark=True,
            ),
            sensors=dict(
                rgb_camera=(RGBCamera, size[0], size[1]),
            ),
            success_reward=50.0,
            out_of_road_penalty=20.0,
            crash_vehicle_penalty=30.0,
            driving_reward=2.0,
            speed_reward=0.5,
            use_lateral_reward=True,
        )
        
        self._env = MetaDriveEnv(config)
    
    def reset(self):
        """Reset environment"""
        self._done = False
        self._step_count = 0
        obs, info = self._env.reset()
        return self._format_obs(obs, 0.0, info, True)
    
    def step(self, action):
        """Step environment"""
        if self._done:
            return self.reset()
            
        # Convert action format if needed
        if isinstance(action, dict):
            md_action = [action.get('steering', 0), action.get('throttle_brake', 0)]
        else:
            md_action = action
            
        obs, reward, terminated, truncated, info = self._env.step(md_action)
        self._step_count += 1
        self._done = terminated or truncated
        
        return self._format_obs(obs, reward, info, self._done)
    
    def _format_obs(self, obs, reward, info, done, is_first=False):
        """Format observation for consistency"""
        vehicle = self._env.agent
        
        # Extract image
        if 'image' in obs:
            image = obs['image']
            if len(image.shape) == 3 and image.shape[-1] == 4:
                image = image[..., :3]
        else:
            image = np.zeros((*self._size, 3), dtype=np.uint8)
        
        # Ensure correct image size
        if image.shape[:2] != self._size:
            try:
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(image)
                pil_image = pil_image.resize(self._size, PILImage.LANCZOS)
                image = np.array(pil_image)
            except ImportError:
                # Simple fallback
                image = np.zeros((*self._size, 3), dtype=np.uint8)
        
        # Vehicle state
        velocity = getattr(vehicle, 'velocity', [0, 0, 0])
        speed = np.linalg.norm(velocity[:2])
        
        # Navigation info
        route_completion = info.get('route_completion', 0.0)
        
        return {
            'image': image.astype(np.uint8),
            'speed': np.float32(speed),
            'route_completion': np.float32(route_completion),
            'reward': np.float32(reward),
            'done': done,
            'is_first': is_first,
            'info': info
        }
    
    def close(self):
        """Close environment"""
        if hasattr(self, '_env'):
            self._env.close()
    
    def render(self):
        """Render environment"""
        return self._env.render(mode='rgb_array')
