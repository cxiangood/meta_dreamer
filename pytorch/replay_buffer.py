"""
Replay Buffer for Dreamer-style training.
Stores (obs, action, reward, continue, is_first) transitions and samples sequences.
"""

import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Circular replay buffer that stores episodes and samples consecutive sequences.
    Protected slots preserve successful / GT-warmup transitions from overwrite.
    """
    def __init__(self, capacity, obs_shape, action_dim=2):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        self.episodes = deque(maxlen=10000)
        self.total_steps = 0
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._gt_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._continues = np.zeros(capacity, dtype=np.float32)
        self._is_first = np.zeros(capacity, dtype=np.bool_)
        self._protected = set()

        self._ptr = 0
        self._filled = 0

    def add(self, obs, action, reward, done, is_first=False, gt_action=None):
        """Add a transition; skip protected slots when possible."""
        idx = self._ptr % self.capacity
        if idx in self._protected and len(self._protected) < self._filled:
            self._ptr += 1
            idx = self._ptr % self.capacity
            if idx in self._protected:
                self._ptr += 1
                idx = self._ptr % self.capacity

        self._obs[idx] = obs
        self._actions[idx] = action
        self._gt_actions[idx] = (
            gt_action if gt_action is not None
            else np.zeros(self.action_dim, dtype=np.float32))
        self._rewards[idx] = reward
        self._continues[idx] = 0.0 if done else 1.0
        self._is_first[idx] = is_first
        self._ptr += 1
        self._filled = min(self._filled + 1, self.capacity)
        self.total_steps += 1

    def add_episode(self, observations, actions, rewards, dones):
        for t in range(len(observations)):
            self.add(
                observations[t], actions[t], rewards[t], dones[t],
                is_first=(t == 0),
            )

    def _valid_starts(self, batch_length, protected_only=False):
        starts = []
        for i in range(max(self._filled - batch_length, 0)):
            if self._is_first[i + 1: i + batch_length].any():
                continue
            if protected_only:
                if not any(
                    (i + j) % self.capacity in self._protected
                    for j in range(batch_length)
                ):
                    continue
            starts.append(i)
        return starts

    def sample(self, batch_size, batch_length, return_gt=False, prefer_protected=0.0):
        """Sample consecutive sequences; optionally bias toward protected (success/GT) data."""
        valid_starts = self._valid_starts(batch_length, protected_only=False)
        prot_starts = (
            self._valid_starts(batch_length, protected_only=True)
            if self._protected else []
        )

        pool = valid_starts
        if (prefer_protected > 0.0 and prot_starts
                and np.random.random() < prefer_protected):
            pool = prot_starts

        if len(pool) >= batch_size:
            indices = np.random.choice(pool, batch_size, replace=True)
        elif len(valid_starts) >= batch_size:
            indices = np.random.choice(valid_starts, batch_size, replace=True)
        else:
            indices = np.random.randint(
                0, max(self._filled - batch_length, 1), batch_size)

        self._last_sample_indices = indices

        obs = np.stack([self._obs[i:i + batch_length] for i in indices])
        actions = np.stack([self._actions[i:i + batch_length] for i in indices])
        rewards = np.stack([self._rewards[i:i + batch_length] for i in indices])
        continues = np.stack([self._continues[i:i + batch_length] for i in indices])

        if return_gt:
            gt_actions = np.stack([self._gt_actions[i:i + batch_length] for i in indices])
            return obs, actions, rewards, continues, gt_actions
        return obs, actions, rewards, continues

    def __len__(self):
        return self._filled

    def protect_last_episode(self, start_idx, end_idx):
        """Mark [start_idx, end_idx) protected (max 20% of capacity)."""
        for i in range(start_idx, end_idx):
            self._protected.add(i % self.capacity)
        max_protected = self.capacity // 5
        if len(self._protected) > max_protected:
            excess = len(self._protected) - max_protected
            for idx in sorted(self._protected)[:excess]:
                self._protected.discard(idx)

    def get_protected_ratio(self):
        return len(self._protected) / max(self._filled, 1)
