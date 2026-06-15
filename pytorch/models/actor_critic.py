"""
Actor-Critic networks for DreamerV3-style imagination training.

Actor: policy network π(a|s) → continuous action distribution
  - Standard: single head
  - Phase-as-Input: concat [feature, phase_onehot]
  - MoE-Phase: 3 expert heads + classifier, hard routing (train) / soft routing (infer)
Critic: value network V(s) with symlog returns and two-hot encoded distribution

Trained entirely in world model imagination.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Symlog:
    @staticmethod
    def forward(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def inverse(x):
        return torch.sign(x) * torch.expm1(torch.abs(x))


class Actor(nn.Module):
    """
    Policy network: outputs tanh-squashed Gaussian residual actions (ResWM-style).

    action = tanh(prev_action + outscale * (Δμ + noise))
    Δμ, noise from Gaussian with learned std.  The prev_action prior prevents
    collapse / saturation and encourages smooth control with minimal hyperparameter
    changes.

    Modes (mutually exclusive):
      - Default: single mean_head
      - phase_as_input=True: concat [feature, phase_onehot] → MLP → single head
      - moe_phase=True: shared MLP → 3 expert heads + phase classifier
                        BC: hard routing (only correct expert trained)
                        Inference: soft routing via classifier
    """
    def __init__(self, feature_dim, action_dim=2, hidden_dim=512,
                 layers=3, act=nn.SiLU, init_std=5.0,
                 use_phase_head=False, phase_as_input=False,
                 moe_phase=False, residual_action=True):
        super().__init__()
        self.action_dim = action_dim
        self.use_phase_head = use_phase_head
        self.phase_as_input = phase_as_input
        self.moe_phase = moe_phase
        self.residual_action = residual_action

        input_dim = feature_dim

        # ResWM: condition on prev_action for residual delta prediction
        if residual_action:
            input_dim += action_dim

        if phase_as_input:
            input_dim += 3

        # Shared backbone
        net = []
        dim = input_dim
        for _ in range(layers):
            net.extend([nn.Linear(dim, hidden_dim), act()])
            dim = hidden_dim
        self.net = nn.Sequential(*net)

        # Output heads
        if moe_phase:
            # 3 expert heads (one per phase) + phase classifier for inference routing
            self.expert_heads = nn.ModuleList([
                nn.Linear(hidden_dim, action_dim) for _ in range(3)
            ])
            for head in self.expert_heads:
                nn.init.xavier_uniform_(head.weight, gain=0.01)
            self.phase_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 256), act(),
                nn.Linear(256, 3),
            )
            self.mean_head = None  # not used
        else:
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)

        # Outscale: DreamerV3 scales the mean by 0.01 to prevent tanh saturation.
        # Residual: small per-step adjustments (0.01).
        # Standard (no-residual): full-scale output (1.0) — tanh handles saturation.
        self.outscale = 0.01 if residual_action else 1.0

        # Shared log_std (per action dim)
        # In residual mode: effective_std = exp(log_std) * outscale, so we must
        # divide init_std by outscale to get the intended effective exploration.
        _init = max(init_std, 1e-4)
        if residual_action:
            _init = _init / self.outscale  # e.g. 0.3 / 0.01 = 30 → exp(ln(30)) * 0.01 = 0.3
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(_init)))

        # Legacy: phase as output (auxiliary loss)
        if use_phase_head:
            self.phase_head = nn.Sequential(
                nn.Linear(hidden_dim, 256), act(),
                nn.Linear(256, 3),
            )

    def forward(self, features, prev_action=None, phase=None, phase_labels=None,
                deterministic=False):
        """
        Args:
            features: (B, feat_dim)
            prev_action: (B, action_dim) previous action, used for residual update.
                         If None, zeros are used (cold start).
            phase: (B, 3) one-hot, used when phase_as_input=True
            phase_labels: (B,) int64 [0,1,2], used when moe_phase=True for hard routing
            deterministic: if True, return tanh(mean); else sample from distribution

        Returns:
            action: (B, action_dim) tanh-squashed
            log_prob: (B,) log probability (None when deterministic)

        MoE phase info stored as self.last_phase_{logits,probs} for trainer access.
        """
        B, D = features.shape
        if self.residual_action:
            if prev_action is None:
                prev_action = torch.zeros(B, self.action_dim, device=features.device)
            x = torch.cat([features, prev_action], dim=-1)
        else:
            x = features

        x = self._join_phase_input(x, phase)
        h = self.net(x)

        if self.moe_phase:
            delta, self.last_phase_logits, self.last_phase_probs = self._moe_mean(h, phase_labels)
        else:
            delta = self.mean_head(h)

        # Outscale: small delta per step → smooth residual control (ResWM)
        delta = delta * self.outscale

        if deterministic:
            if self.residual_action:
                return torch.tanh(prev_action + delta), None
            return torch.tanh(delta), None

        # ResWM: scale noise by outscale so residual update is truly incremental.
        # Without scaling, noise(0, 0.5) dominates the NN signal (0.01*Δμ) by 50×.
        # With residual: init_std=50 → effective std=50*0.01=0.5 (smooth exploration).
        if self.residual_action:
            std = torch.exp(self.log_std).expand_as(delta).clamp(min=1.0, max=80.0)
            std = std * self.outscale
        else:
            std = torch.exp(self.log_std).expand_as(delta).clamp(min=0.1, max=1.0)

        dist = torch.distributions.Normal(delta, std)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw).sum(dim=-1)

        if self.residual_action:
            pre_tanh = prev_action + raw
        else:
            pre_tanh = raw
        action = torch.tanh(pre_tanh)

        # Tanh correction (same formula for residual: ∂action/∂raw = 1 - action²)
        log_prob = log_prob - torch.log(1 - action.pow(2).clamp(max=1 - 1e-6)).sum(dim=-1)

        return action, log_prob

    def _moe_mean(self, h, phase_labels):
        """Compute action mean via MoE routing.

        Training (phase_labels given): hard routing → only correct expert fires.
        Inference (phase_labels=None): soft routing via classifier.
        """
        B = h.shape[0]

        # Phase classifier always computes logits (for CE loss in training)
        phase_logits = self.phase_classifier(h)  # (B, 3)
        phase_probs = F.softmax(phase_logits, dim=-1)

        if phase_labels is not None:
            # Hard routing: only the correct expert produces output
            mean = torch.zeros(B, self.action_dim, device=h.device)
            for i in range(3):
                mask = (phase_labels == i)
                if mask.any():
                    mean[mask] = self.expert_heads[i](h[mask])
        else:
            # Soft routing: weighted sum of expert outputs
            expert_outputs = torch.stack(
                [head(h) for head in self.expert_heads], dim=1
            )  # (B, 3, action_dim)
            mean = (phase_probs.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (B, action_dim)

        return mean, phase_logits, phase_probs

    def _join_phase_input(self, x, phase):
        if self.phase_as_input and phase is not None:
            x = torch.cat([x, phase], dim=-1)
        elif self.phase_as_input:
            p = torch.ones(x.shape[0], 3, device=x.device) / 3.0
            x = torch.cat([x, p], dim=-1)
        return x

    @torch.no_grad()
    def act(self, features, prev_action=None, deterministic=False, phase=None):
        """Deterministic/exploration action for online collection."""
        B, D = features.shape
        if self.residual_action:
            if prev_action is None:
                prev_action = torch.zeros(B, self.action_dim, device=features.device)
            x = torch.cat([features, prev_action], dim=-1)
        else:
            x = features

        x = self._join_phase_input(x, phase)
        h = self.net(x)

        if self.moe_phase:
            # Soft routing for inference
            phase_probs = F.softmax(self.phase_classifier(h), dim=-1)
            expert_outputs = torch.stack(
                [head(h) for head in self.expert_heads], dim=1
            )
            delta = (phase_probs.unsqueeze(-1) * expert_outputs).sum(dim=1)
        else:
            delta = self.mean_head(h)

        delta = delta * self.outscale

        if deterministic:
            if self.residual_action:
                return torch.tanh(prev_action + delta)
            return torch.tanh(delta)
        if self.residual_action:
            std = torch.exp(self.log_std).expand_as(delta).clamp(min=1.0, max=80.0)
            std = std * self.outscale
        else:
            std = torch.exp(self.log_std).expand_as(delta).clamp(min=0.1, max=1.0)
        raw = torch.distributions.Normal(delta, std).sample()
        if self.residual_action:
            return torch.tanh(prev_action + raw)
        return torch.tanh(raw)


class Critic(nn.Module):
    """
    Value network V(s) with two-hot encoded symlog distribution (DreamerV3).
    Predicts a distribution over returns rather than a point estimate.
    """
    def __init__(self, feature_dim, hidden_dim=512, layers=3,
                 bins=255, min_val=-20.0, max_val=20.0, act=nn.SiLU):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val

        net = []
        dim = feature_dim
        for _ in range(layers):
            net.extend([nn.Linear(dim, hidden_dim), act()])
            dim = hidden_dim
        net.append(nn.Linear(hidden_dim, bins))
        self.net = nn.Sequential(*net)

        # Exponentially-spaced bin centers in symlog space
        self.register_buffer("bin_centers", torch.linspace(min_val, max_val, bins))

    def forward(self, features):
        logits = self.net(features)
        probs = F.softmax(logits, dim=-1)
        value_symlog = (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1)
        value = Symlog.inverse(value_symlog)
        return value, logits

    def compute_target(self, returns):
        """Two-hot encode returns in symlog space."""
        symlog_r = Symlog.forward(returns).clamp(self.min_val, self.max_val)

        # Find bin positions
        pos = (symlog_r - self.min_val) / (self.max_val - self.min_val) * (self.bins - 1)
        below = pos.clamp(0, self.bins - 2).long()
        above = below + 1

        frac = (symlog_r - self.bin_centers[below]) / (
            self.bin_centers[above] - self.bin_centers[below] + 1e-8
        ).clamp(0, 1)

        target = torch.zeros(returns.shape[0], self.bins, device=returns.device)
        target.scatter_(1, below.unsqueeze(1), (1 - frac).unsqueeze(1))
        target.scatter_(1, above.unsqueeze(1), frac.unsqueeze(1))
        return target
