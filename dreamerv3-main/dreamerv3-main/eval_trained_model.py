#!/usr/bin/env python3
"""
Evaluation script for trained MetaDrive model on Windows
"""

import sys
import numpy as np
import time

def evaluate_model(model_path="./models/metadrive_ppo_final", episodes=10, render=False):
    """Evaluate a trained model"""
    try:
        from stable_baselines3 import PPO
        sys.path.insert(0, '.')
        from train_windows_metadrive import MetaDriveGymWrapper
        
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path)
        
        # Create environment
        env = MetaDriveGymWrapper(size=(64, 64), length=2000, render_mode='rgb_array' if render else None)
        
        print(f"Starting evaluation with {episodes} episodes...")
        
        stats = {
            'total_reward': 0,
            'total_steps': 0,
            'success_count': 0,
            'episodes': 0
        }
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            
            while not done:
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                done = terminated or truncated
                
                # Print progress occasionally
                if episode_steps % 100 == 0:
                    print(f"  Step {episode_steps}: Speed={obs['speed']:.1f}, "
                          f"Route completion={obs['route_completion']:.2%}, "
                          f"Reward={reward:.3f}")
                
                # Render if requested
                if render:
                    img = env.render()
                    if img is not None:
                        # Here you could display the image using matplotlib or save it
                        pass
                    time.sleep(0.05)  # Slow down for visualization
                
                # Safety break
                if episode_steps > 5000:
                    break
            
            # Episode summary
            route_completion = obs['route_completion']
            print(f"  Final reward: {episode_reward:.2f}")
            print(f"  Episode steps: {episode_steps}")
            print(f"  Route completion: {route_completion:.2%}")
            
            if route_completion > 0.8:
                stats['success_count'] += 1
                print("  Result: SUCCESS ✓")
            else:
                print("  Result: INCOMPLETE ⚠")
            
            stats['total_reward'] += episode_reward
            stats['total_steps'] += episode_steps
            stats['episodes'] += 1
        
        # Final statistics
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Episodes: {stats['episodes']}")
        print(f"Average Reward: {stats['total_reward']/stats['episodes']:.2f}")
        print(f"Average Steps: {stats['total_steps']/stats['episodes']:.1f}")
        print(f"Success Rate: {stats['success_count']/stats['episodes']:.1%}")
        print(f"Average Route Completion: {obs['route_completion']:.1%}")
        
        env.close()
        return stats
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MetaDrive Model')
    parser.add_argument('--model', type=str, default='./models/metadrive_ppo_final',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering (slower)')
    
    args = parser.parse_args()
    
    print("=== MetaDrive Model Evaluation ===\n")
    
    stats = evaluate_model(args.model, args.episodes, args.render)
    
    if stats:
        print(f"\n🎉 Evaluation completed!")
    else:
        print(f"\n❌ Evaluation failed!")

if __name__ == "__main__":
    main()