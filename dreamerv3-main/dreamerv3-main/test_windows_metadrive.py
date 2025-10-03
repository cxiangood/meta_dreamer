#!/usr/bin/env python3
"""
Simplified test script for MetaDrive environment on Windows
"""

import sys
import os
import numpy as np

def test_metadrive_basic():
    """Basic test of MetaDrive without DreamerV3 dependencies"""
    try:
        # Test MetaDrive installation
        from metadrive import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
        print("✓ Successfully imported MetaDrive")
        
        # Create a basic MetaDrive environment
        config = dict(
            use_render=False,
            num_scenarios=10,
            start_seed=0,
            horizon=200,
            map=1,
            traffic_density=0.05,
            image_observation=True,  # Move to top level
            vehicle_config=dict(
                show_lidar=False,
            ),
            sensors=dict(
                rgb_camera=(RGBCamera, 64, 64),
            ),
        )
        
        env = MetaDriveEnv(config)
        print("✓ Successfully created MetaDrive environment")
        
        # Test basic functionality
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"✓ Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'numpy array'}")
        
        # Test a few steps
        for i in range(5):
            action = [np.random.uniform(-0.1, 0.1), np.random.uniform(0.0, 0.3)]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✓ Step {i+1}: Reward={reward:.3f}, Done={terminated or truncated}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("✓ Environment closed successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_windows_compatible_env():
    """Create a Windows-compatible MetaDrive wrapper"""
    try:
        # Create a simplified environment wrapper that works on Windows
        wrapper_code = '''
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
'''
        
        # Write the wrapper to a file
        with open('windows_metadrive_env.py', 'w') as f:
            f.write(wrapper_code)
        
        print("✓ Created Windows-compatible environment wrapper: windows_metadrive_env.py")
        return True
        
    except Exception as e:
        print(f"✗ Error creating wrapper: {e}")
        return False

def test_windows_wrapper():
    """Test the Windows wrapper"""
    try:
        # Import the wrapper
        sys.path.insert(0, '.')
        from windows_metadrive_env import WindowsMetaDriveLaneKeeping
        
        # Test the wrapper
        env = WindowsMetaDriveLaneKeeping(size=(32, 32), length=100)
        print("✓ Created Windows wrapper environment")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Reset successful, obs keys: {list(obs.keys())}")
        print(f"✓ Image shape: {obs['image'].shape}")
        
        # Test steps
        for i in range(3):
            action = [np.random.uniform(-0.1, 0.1), np.random.uniform(0.0, 0.2)]
            obs = env.step(action)
            print(f"✓ Step {i+1}: Speed={obs['speed']:.2f}, Reward={obs['reward']:.3f}")
            
            if obs['done']:
                break
        
        env.close()
        print("✓ Windows wrapper test successful")
        return True
        
    except Exception as e:
        print(f"✗ Windows wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing MetaDrive on Windows ===\\n")
    
    success1 = test_metadrive_basic()
    print("\\n" + "="*50 + "\\n")
    
    success2 = create_windows_compatible_env()
    print("\\n" + "="*30 + "\\n")
    
    success3 = test_windows_wrapper()
    print("\\n" + "="*50)
    
    if success1 and success2 and success3:
        print("\\n🎉 MetaDrive works on Windows!")
        print("\\nNext steps:")
        print("1. For full DreamerV3 integration, consider using WSL2")
        print("2. Or use the Windows wrapper for basic RL training")
        print("\\nTo install WSL2 (recommended):")
        print("wsl --install -d Ubuntu")
    else:
        print("\\n❌ Some tests failed. Check the errors above.")