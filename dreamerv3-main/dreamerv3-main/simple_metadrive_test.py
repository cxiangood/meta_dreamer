#!/usr/bin/env python3
"""
Simple MetaDrive test for Linux environment
"""

import sys
import numpy as np

def test_basic_metadrive():
    """Test basic MetaDrive functionality"""
    try:
        from metadrive import MetaDriveEnv  
        from metadrive.component.sensors.rgb_camera import RGBCamera
        print("✓ MetaDrive imported successfully")
        
        # Use the simplest possible configuration
        config = dict(
            use_render=False,
            num_scenarios=10,
            start_seed=0,
            horizon=100,
            map=1,  # Simplest map
            traffic_density=0.0,  # No traffic to avoid complications
            sensors=dict(
                rgb_camera=(RGBCamera, 64, 64),
            ),
        )
        
        env = MetaDriveEnv(config)
        print("✓ MetaDrive environment created")
        
        # Test basic functionality
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"✓ Observation type: {type(obs)}")
        if isinstance(obs, dict):
            print(f"✓ Observation keys: {list(obs.keys())}")
            if 'image' in obs:
                print(f"✓ Image shape: {obs['image'].shape}")
        
        # Test a few steps
        for i in range(3):
            action = [0.0, 0.1]  # Small forward action
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✓ Step {i+1}: Reward={reward:.3f}")
            if terminated or truncated:
                break
                
        env.close()
        print("✓ Basic MetaDrive test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic MetaDrive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dreamerv3_metadrive():
    """Test our DreamerV3 MetaDrive wrapper"""
    try:
        sys.path.insert(0, '.')
        from embodied.envs.metadrive_lane_keeping import MetaDriveLaneKeeping
        print("✓ MetaDriveLaneKeeping imported")
        
        # Test with minimal configuration
        env = MetaDriveLaneKeeping('lane_keeping', size=(32, 32), length=50)
        print("✓ MetaDriveLaneKeeping created")
        
        # Test reset
        obs = env.step({'steering': 0.0, 'throttle_brake': 0.0, 'reset': True})
        print(f"✓ Reset successful")
        print(f"✓ Observation keys: {list(obs.keys())}")
        print(f"✓ Image shape: {obs['image'].shape}")
        
        # Test a few steps
        for i in range(3):
            action = {'steering': 0.0, 'throttle_brake': 0.1, 'reset': False}
            obs = env.step(action)
            print(f"✓ Step {i+1}: Speed={obs['speed']:.2f}, Reward={obs['reward']:.3f}")
            if obs['is_last']:
                break
                
        env.close()
        print("✓ DreamerV3 MetaDrive test passed!")
        return True
        
    except Exception as e:
        print(f"✗ DreamerV3 MetaDrive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing MetaDrive on Linux ===\n")
    
    success1 = test_basic_metadrive()
    print("\n" + "="*40 + "\n")
    
    success2 = test_dreamerv3_metadrive()
    print("\n" + "="*40)
    
    if success1 and success2:
        print("\n🎉 All tests passed! Ready for DreamerV3 training!")
    else:
        print("\n❌ Some tests failed.")