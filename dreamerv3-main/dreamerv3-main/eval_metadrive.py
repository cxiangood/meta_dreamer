#!/usr/bin/env python3
"""
Evaluation and visualization script for trained MetaDrive Lane Keeping agent
"""

import os
import sys
import argparse
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Evaluate MetaDrive Lane Keeping Agent')
    parser.add_argument('--logdir', type=str, required=True,
                       help='Path to the trained model directory')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Enable visual rendering')
    parser.add_argument('--save_video', type=str, default=None,
                       help='Save video to specified path')
    
    args = parser.parse_args()
    
    # Add paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    
    try:
        from embodied.envs.metadrive_lane_keeping import MetaDriveLaneKeeping
        from dreamerv3.main import make_agent
        from types import SimpleNamespace
        import elements
        
        print(f"Loading model from: {args.logdir}")
        
        # Create config
        config = SimpleNamespace()
        config.task = 'metadrive_lane_keeping'
        config.logdir = args.logdir
        config.env = {
            'metadrive': {
                'size': [64, 64],
                'repeat': 1,  # No repeat for evaluation
                'length': 2000
            }
        }
        config.seed = 42
        config.batch_size = 1
        config.batch_length = 64
        config.replay_context = 1
        config.report_length = 32
        config.replica = 0
        config.replicas = 1
        config.random_agent = False
        config.jax = {
            'platform': 'cpu',
            'policy_devices': [0],
            'train_devices': [0]
        }
        config.agent = {}  # Will be loaded from checkpoint
        
        # Create environment
        env_config = config.env.get('metadrive', {})
        if args.render:
            env_config['use_render'] = True
            
        env = MetaDriveLaneKeeping('lane_keeping', **env_config)
        print("✓ Environment created")
        
        # Load agent
        agent = make_agent(config)
        print("✓ Agent loaded")
        
        # Evaluation loop
        stats = {
            'episodes': 0,
            'total_reward': 0,
            'total_length': 0,
            'success_rate': 0,
            'crash_rate': 0,
            'out_of_road_rate': 0
        }
        
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            
            # Reset environment
            obs = env.step({'steering': 0, 'throttle_brake': 0, 'reset': True})
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Initialize agent state
            agent_state = None
            
            while not done:
                # Get action from agent
                if agent_state is None:
                    action, agent_state = agent.act(obs, mode='eval')
                else:
                    action, agent_state = agent.act(obs, agent_state, mode='eval')
                
                # Step environment
                obs = env.step(action)
                
                episode_reward += obs['reward']
                episode_length += 1
                done = obs['is_last']
                
                # Render if requested
                if args.render:
                    img = env.render()
                    if img is not None:
                        # Simple display using print (could be enhanced with matplotlib)
                        print(f"  Step {episode_length}: Reward={obs['reward']:.3f}, "
                              f"Speed={obs['speed']:.1f}, "
                              f"Distance_to_route={obs['distance_to_route']:.2f}")
                    time.sleep(0.1)  # Slow down for visualization
                
                # Safety break for very long episodes
                if episode_length > 5000:
                    break
            
            # Episode statistics
            print(f"  Final reward: {episode_reward:.2f}")
            print(f"  Episode length: {episode_length}")
            print(f"  Route completion: {obs['route_completion']:.2%}")
            
            # Update stats
            stats['episodes'] += 1
            stats['total_reward'] += episode_reward
            stats['total_length'] += episode_length
            
            # Determine episode outcome (simplified)
            if obs['route_completion'] > 0.9:
                stats['success_rate'] += 1
                print("  Result: SUCCESS ✓")
            elif obs.get('is_terminal', False):
                stats['crash_rate'] += 1  
                print("  Result: CRASH ✗")
            else:
                stats['out_of_road_rate'] += 1
                print("  Result: OUT OF ROAD ⚠")
        
        # Final statistics
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Episodes: {stats['episodes']}")
        print(f"Average Reward: {stats['total_reward']/stats['episodes']:.2f}")
        print(f"Average Length: {stats['total_length']/stats['episodes']:.1f}")
        print(f"Success Rate: {stats['success_rate']/stats['episodes']:.1%}")
        print(f"Crash Rate: {stats['crash_rate']/stats['episodes']:.1%}")
        print(f"Out-of-Road Rate: {stats['out_of_road_rate']/stats['episodes']:.1%}")
        
        env.close()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()