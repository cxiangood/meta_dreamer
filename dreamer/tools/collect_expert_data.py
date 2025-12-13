#!/usr/bin/env python3
"""
Collect expert trajectories using simple rule-based expert for supervised pre-training.
"""

import os
import csv
import numpy as np
from metadrive import MetaDriveEnv


class ExpertDataCollector:
    def __init__(self, num_episodes=10, max_steps=10000, seed=42):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.seed = seed

        # Setup environment (use default policy, we'll control manually)
        self.env = MetaDriveEnv({
            "num_scenarios": 100,
            "horizon": max_steps,
            "start_seed": seed,
            "use_render": False,
            "vehicle_config": dict(
                show_navi_mark=False,
                show_dest_mark=False,
                show_line_to_dest=False,
                show_line_to_navi_mark=False,
            ),
            "out_of_road_done": True,  # 出界立即终止（悬崖模型）
            "crash_vehicle_done": False,
            "crash_object_done": False,
        })

        # Data storage
        self.trajectories = []

    def get_expert_action(self, obs):
        """Simple expert action: maintain lane and speed."""
        # Simple rule-based expert: steer towards lane center, maintain speed
        # This is a basic approximation - real expert would use IDM/path tracking
        return np.array([0.0, 0.5])  # [steering, throttle] - go straight with moderate speed

    def collect_trajectory(self):
        """Collect one expert trajectory."""
        obs = self.env.reset()
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }

        done = False
        step = 0
        while not done and step < self.max_steps:
            # Get expert action (simple rule-based for now)
            action = self.get_expert_action(obs)

            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Store transition
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            trajectory['infos'].append(info)

            obs = next_obs
            step += 1

        return trajectory

    def collect_all(self):
        """Collect all expert trajectories."""
        print(f"Collecting {self.num_episodes} expert trajectories...")
        for i in range(self.num_episodes):
            if i % 5 == 0:
                print(f"Episode {i}/{self.num_episodes}")
            trajectory = self.collect_trajectory()
            self.trajectories.append(trajectory)

        print("Collection complete!")
        return self.trajectories

    def save_to_csv(self, filename='expert_trajectories.csv'):
        """Save trajectories to CSV for later use."""
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['episode', 'step', 'action', 'reward', 'done']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ep_idx, trajectory in enumerate(self.trajectories):
                for step_idx, (obs, action, reward, done, info) in enumerate(zip(
                    trajectory['observations'],
                    trajectory['actions'],
                    trajectory['rewards'],
                    trajectory['dones'],
                    trajectory['infos']
                )):
                    writer.writerow({
                        'episode': ep_idx,
                        'step': step_idx,
                        'action': action.tolist() if hasattr(action, 'tolist') else list(action) if hasattr(action, '__iter__') and not isinstance(action, str) else str(action),
                        'reward': reward,
                        'done': done
                    })

        print(f"Saved trajectories to {filename}")


if __name__ == "__main__":
    collector = ExpertDataCollector(num_episodes=10, max_steps=10000, seed=42)
    trajectories = collector.collect_all()
    collector.save_to_csv('expert_trajectories.csv')