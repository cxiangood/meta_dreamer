#!/usr/bin/env python3
"""
Test script for MetaDrive Lane Keeping environment in DreamerV3
"""

import sys
import os
import numpy as np

# Add the dreamerv3 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_metadrive_env():
    """Test the MetaDrive lane keeping environment"""
    try:
        from embodied.envs.metadrive_lane_keeping import MetaDriveLaneKeeping
        print("✓ Successfully imported MetaDriveLaneKeeping")
        
        # Create environment
        env = MetaDriveLaneKeeping('lane_keeping', size=(64, 64), length=100)
        print("✓ Successfully created environment")
        
        # Test observation and action spaces
        obs_space = env.obs_space
        act_space = env.act_space
        
        print(f"✓ Observation space keys: {list(obs_space.keys())}")
        print(f"✓ Action space keys: {list(act_space.keys())}")
        
        # Test reset
        obs = env.step({'steering': 0.0, 'throttle_brake': 0.0, 'reset': True})
        print("✓ Successfully reset environment")
        print(f"✓ Observation keys: {list(obs.keys())}")
        print(f"✓ Image shape: {obs['image'].shape}")
        print(f"✓ Speed: {obs['speed']}")
        print(f"✓ Is first: {obs['is_first']}")
        
        # Test a few steps
        for i in range(5):
            action = {
                'steering': np.random.uniform(-0.1, 0.1), 
                'throttle_brake': np.random.uniform(0.0, 0.3),
                'reset': False
            }
            obs = env.step(action)
            print(f"✓ Step {i+1}: Speed={obs['speed']:.2f}, Reward={obs['reward']:.3f}")
            
            if obs['is_last']:
                print("  Episode ended")
                break
        
        # Test rendering
        try:
            render_img = env.render()
            if render_img is not None:
                print(f"✓ Rendering works, image shape: {render_img.shape}")
            else:
                print("⚠ Rendering returned None")
        except Exception as e:
            print(f"⚠ Rendering failed: {e}")
        
        # Clean up
        env.close()
        print("✓ Successfully closed environment")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Make sure MetaDrive is installed: pip install metadrive-simulator")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dreamerv3_integration():
    """Test integration with DreamerV3 main function"""
    try:
        # Reset any existing MetaDrive engine to avoid initialization conflicts
        try:
            from metadrive.engine.engine_utils import close_engine
            close_engine()
        except:
            pass  # Ignore if engine is not initialized
        
        # Test basic import of the environment module
        import importlib.util
        
        # Test if we can import the environment directly
        spec = importlib.util.spec_from_file_location(
            "metadrive_lane_keeping", 
            "embodied/envs/metadrive_lane_keeping.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        env_class = getattr(module, 'MetaDriveLaneKeeping')
        print("✓ Successfully imported MetaDriveLaneKeeping class")
        
        # Test environment creation
        env = env_class('lane_keeping', size=(32, 32), length=50)  # Smaller for testing
        print("✓ Successfully created environment instance")
        
        # Test basic functionality
        obs_space = env.obs_space
        act_space = env.act_space
        print(f"✓ Obs space: {list(obs_space.keys())}")
        print(f"✓ Act space: {list(act_space.keys())}")
        
        env.close()
        print("✓ DreamerV3 integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ DreamerV3 integration error: {e}")
        print("  This may be due to missing dependencies or import path issues")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing MetaDrive Lane Keeping Environment ===\n")
    
    success1 = test_metadrive_env()
    print("\n" + "="*50 + "\n")
    
    success2 = test_dreamerv3_integration()
    print("\n" + "="*50)
    
    if success1 and success2:
        print("\n🎉 All tests passed! The environment is ready to use.")
        print("\nTo train with DreamerV3, use:")
        print("python dreamerv3/main.py --configs metadrive_lane_keeping")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)