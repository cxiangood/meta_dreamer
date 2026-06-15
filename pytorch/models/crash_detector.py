"""
Online Crash Detector: binary classifier trained on RSSM features from real episodes.

- Positive samples: features from K frames before crash/out_of_road
- Negative samples: features from normal driving frames
- Used in imagination to replace continue_head-based crash penalty
"""

import torch
import torch.nn as nn


class CrashDetector(nn.Module):
    """MLP binary classifier: P(crash | RSSM feature).

    Input: RSSM feature (deter + stoch_flat, dim=3072)
    Output: crash probability [0, 1]
    """

    def __init__(self, feature_dim=3072, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, feature):
        return torch.sigmoid(self.net(feature)).squeeze(-1)


class CrashFeatureBuffer:
    """Balanced buffer of (feature, label) pairs for crash detector training.

    Maintains separate queues for positive (crash) and negative (normal) samples.
    """

    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.pos_features = []  # list of tensors (feat_dim,)
        self.neg_features = []

    def add_crash(self, features):
        """Add pre-crash features as positive samples."""
        for f in features:
            self.pos_features.append(f.detach().cpu())
        self._trim()

    def add_normal(self, features):
        """Add normal driving features as negative samples."""
        for f in features:
            self.neg_features.append(f.detach().cpu())
        self._trim()

    def _trim(self):
        max_each = self.max_samples // 2
        if len(self.pos_features) > max_each:
            self.pos_features = self.pos_features[-max_each:]
        if len(self.neg_features) > max_each:
            self.neg_features = self.neg_features[-max_each:]

    def size(self):
        return len(self.pos_features) + len(self.neg_features)

    def ready(self, min_samples=64):
        return len(self.pos_features) >= min_samples // 2 and len(self.neg_features) >= min_samples // 2

    def sample_batch(self, batch_size, device='cpu'):
        """Sample a balanced batch for training."""
        n_pos = batch_size // 2
        n_neg = batch_size - n_pos

        pos_idx = torch.randint(0, len(self.pos_features), (n_pos,))
        neg_idx = torch.randint(0, len(self.neg_features), (n_neg,))

        pos_batch = torch.stack([self.pos_features[i] for i in pos_idx])
        neg_batch = torch.stack([self.neg_features[i] for i in neg_idx])

        features = torch.cat([pos_batch, neg_batch], dim=0).to(device)
        labels = torch.cat([
            torch.ones(n_pos), torch.zeros(n_neg)
        ]).to(device)

        return features, labels
