#!/usr/bin/env python3
"""
Simple RL training script for MetaDrive on Windows using stable-baselines3
This provides an alternative to DreamerV3 that works natively on Windows
"""

import os
import sys
import numpy as np
from typing import Dict, Any
import gym
from gym import spaces

def install_requirements():
    """Install required packages for Windows training"""
    try:
        import stable_baselines3
        print("✓ stable-baselines3 already installed")
    except ImportError:
        print("Installing stable-baselines3...")
        os.system("pip install stable-baselines3[extra]")
    
    try:
        import tensorboard
        print("✓ tensorboard already installed")
    except ImportError:
        print("Installing tensorboard...")
        os.system("pip install tensorboard")

class MetaDriveGymWrapper(gym.Env):
    """
    Gym wrapper for MetaDrive to use with stable-baselines3
    """
    
    def __init__(self, render_mode=None, **kwargs):
        super().__init__()
        
        # Import our Windows wrapper
        sys.path.insert(0, '.')
        from windows_metadrive_env import WindowsMetaDriveLaneKeeping
        
        self.env = WindowsMetaDriveLaneKeeping(**kwargs)
        self._render_mode = render_mode
        
        # Define action space: [steering, throttle_brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space
        # We'll use a dictionary space that includes image and vehicle state
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255, 
                shape=(64, 64, 3), 
                dtype=np.uint8
            ),
            'speed': spaces.Box(
                low=0.0, high=50.0, 
                shape=(), 
                dtype=np.float32
            ),
            'route_completion': spaces.Box(
                low=0.0, high=1.0, 
                shape=(), 
                dtype=np.float32
            )
        })
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        obs = self.env.reset()
        gym_obs = self._convert_obs(obs)
        return gym_obs, {}
    
    def step(self, action):
        # Convert action to dictionary format
        action_dict = {
            'steering': float(action[0]),
            'throttle_brake': float(action[1])
        }
        
        obs = self.env.step(action_dict)
        gym_obs = self._convert_obs(obs)
        
        reward = float(obs['reward'])
        terminated = obs['done']
        truncated = False  # We'll handle this in the environment
        info = obs.get('info', {})
        
        return gym_obs, reward, terminated, truncated, info
    
    def _convert_obs(self, obs):
        """Convert our custom observation format to gym format"""
        # Ensure image is the right shape
        image = obs['image']
        
        # Fix image dimensions
        while len(image.shape) > 3:
            image = image[..., 0]  # Remove extra dimensions
        
        if len(image.shape) == 3 and image.shape[-1] > 3:
            image = image[..., :3]  # Keep only RGB channels
            
        # Ensure it's uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        return {
            'image': image,
            'speed': np.float32(obs['speed']),
            'route_completion': np.float32(obs['route_completion'])
        }
    
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.env.render()
        return None
    
    def close(self):
        self.env.close()

def train_with_stable_baselines3():
    """Train using stable-baselines3 (works on Windows)"""
    print("Starting training with stable-baselines3...")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
        
        # Create environment
        def make_env():
            env = MetaDriveGymWrapper(size=(64, 64), length=1000)
            env = Monitor(env)
            return env
        
        # Create vectorized environment for parallel training
        env = make_vec_env(make_env, n_envs=4)
        eval_env = make_vec_env(make_env, n_envs=1)
        
        print("✓ Created vectorized environments")
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path='./models/',
            log_path='./logs/', 
            eval_freq=10000,
            deterministic=True, 
            render=False
        )
        
        # Create PPO model
        model = PPO(
            "MultiInputPolicy",  # For dictionary observations
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./tensorboard_logs/",
            verbose=1
        )
        
        print("✓ Created PPO model")
        print("Starting training...")
        print("  - Total timesteps: 1,000,000")
        print("  - Parallel environments: 4")
        print("  - Model saves: ./models/")
        print("  - Logs: ./logs/")
        print("  - TensorBoard: ./tensorboard_logs/")
        print("  - Monitor training: tensorboard --logdir=./tensorboard_logs/")
        
        # Train the model
        model.learn(
            total_timesteps=1_000_000,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save final model
        model.save("./models/metadrive_ppo_final")
        print("✓ Training completed! Model saved to ./models/metadrive_ppo_final")
        
        # Test the trained model
        print("\nTesting trained model...")
        obs, _ = eval_env.reset()
        for i in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            if i % 20 == 0:
                print(f"  Step {i}: Reward={rewards[0]:.3f}")
        
        env.close()
        eval_env.close()
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== MetaDrive Windows Training ===\n")
    
    # Install requirements
    print("1. Checking requirements...")
    install_requirements()
    print()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True) 
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Test environment first
    print("2. Testing environment...")
    try:
        env = MetaDriveGymWrapper(size=(32, 32), length=100)
        obs, _ = env.reset()
        print(f"✓ Environment test passed")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Image shape: {obs['image'].shape}")
        env.close()
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return
    print()
    
    # Start training
    print("3. Starting training...")
    success = train_with_stable_baselines3()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("\nNext steps:")
        print("1. Monitor training: tensorboard --logdir=./tensorboard_logs/")
        print("2. Load model: model = PPO.load('./models/metadrive_ppo_final')")
        print("3. Evaluate: python eval_trained_model.py")
    else:
        print("\n❌ Training failed. Check the errors above.")

if __name__ == "__main__":
    main()