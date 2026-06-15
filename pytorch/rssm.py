"""
RSSM with SIGReg: Recurrent State Space Model replacing KL divergence with SIGReg.

SIGReg acts on continuous latent states (deter / stoch_logits) rather than
discrete one-hot samples, eliminating the discrete→continuous mismatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sigreg import SIGReg


class RSSM(nn.Module):
    def __init__(self, action_dim=2, deter_dim=1024, stoch_dim=32,
                 stoch_classes=64, hidden_dim=1024, embed_dim=512,
                 regularization="sigreg",
                 sigreg_lambda=0.1, sigreg_projections=1024,
                 sigreg_target="deter",
                 kl_beta=1.0, kl_free_bits=1.0,
                 gru_layers=1, act=nn.SiLU):
        """
        Args:
            regularization: "sigreg" | "kl" | "none"
            sigreg_target: which continuous state SIGReg acts on:
                "stoch"  - discrete one-hot (old, avoid)
                "deter"  - GRU deterministic state (1024-dim continuous)
                "logits" - categorical logits before sampling (2048-dim continuous)
                "deter+logits" - both concatenated
        """
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_flat = stoch_dim * stoch_classes
        self.regularization = regularization
        self.sigreg_lambda = sigreg_lambda
        self.sigreg_target = sigreg_target
        self.kl_beta = kl_beta
        self.kl_free_bits = kl_free_bits

        # GRU for deterministic state: h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])
        self.gru = nn.GRUCell(
            input_size=self.stoch_flat + action_dim,
            hidden_size=deter_dim
        )

        # Posterior: q(s_t | h_t, e_t) - uses observation
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )

        # Prior: p(s_t | h_t) - no observation, for imagination
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )

        # Regularizer: SIGReg on continuous states (deter / logits)
        self.sigreg = None       # for stoch_flat (2048-dim)
        self.sigreg_deter = None # for deter (1024-dim)
        if regularization == "sigreg":
            if "logits" in sigreg_target or sigreg_target == "stoch":
                self.sigreg = SIGReg(
                    embed_dim=self.stoch_flat,
                    num_projections=sigreg_projections,
                )
            if "deter" in sigreg_target:
                self.sigreg_deter = SIGReg(
                    embed_dim=deter_dim,
                    num_projections=sigreg_projections,
                )

    def initial_state(self, batch_size, device=None):
        """Return initial RSSM state (zeros)."""
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device),
            "stoch_logits": torch.zeros(batch_size, self.stoch_flat, device=device),
        }

    def _to_logits(self, x):
        """Reshape flat vector to (batch, stoch_dim, stoch_classes) logits."""
        return x.reshape(x.shape[0], self.stoch_dim, self.stoch_classes)

    def _sample(self, logits):
        """Gumbel-softmax sample from categorical logits."""
        # Straight-through Gumbel-softmax
        if self.training:
            one_hot = F.gumbel_softmax(logits, hard=True, tau=1.0)
        else:
            indices = logits.argmax(dim=-1)
            one_hot = F.one_hot(indices, self.stoch_classes).float()
        return one_hot

    def observe(self, embeds, actions, prev_state):
        """
        Posterior rollout: given observation embeddings and actions,
        compute the full state sequence using posterior (q).

        Args:
            embeds: (seq_len, batch, embed_dim) encoder outputs
            actions: (seq_len, batch, action_dim)
            prev_state: dict with 'deter', 'stoch'
        Returns:
            post_states: list of state dicts
            priors: list of prior state dicts (for SIGReg loss)
        """
        seq_len = embeds.shape[0]
        batch = embeds.shape[1]
        device = embeds.device

        h = prev_state["deter"]
        s_flat = prev_state["stoch"].reshape(batch, -1)

        post_states = []
        priors = []

        for t in range(seq_len):
            h, s_post, s_prior, post_logits, prior_logits = self._step(
                h, s_flat, actions[t], embeds[t]
            )
            post_states.append({
                "deter": h,
                "stoch": s_post,
                "stoch_logits": post_logits,
            })
            priors.append({
                "stoch": s_prior,
                "prior_logits": prior_logits,
            })
            s_flat = s_post.reshape(batch, -1)

        return post_states, priors

    def imagine(self, actions, prev_state):
        """
        Prior rollout: imagine future states without observations.
        Used for actor-critic training in imagination.

        Args:
            actions: (horizon, batch, action_dim)
            prev_state: dict with 'deter', 'stoch'
        Returns:
            states: list of state dicts
        """
        horizon = actions.shape[0]
        batch = actions.shape[1]
        device = actions.device

        h = prev_state["deter"]
        s_flat = prev_state["stoch"].reshape(batch, -1)

        states = []
        for t in range(horizon):
            h = self.gru(torch.cat([s_flat, actions[t]], dim=-1), h)

            # Prior only (no observation)
            prior_logits = self._to_logits(self.prior_net(h))
            s = self._sample(prior_logits)

            states.append({
                "deter": h,
                "stoch": s,
            })
            s_flat = s.reshape(batch, -1)

        return states

    def _step(self, h, s_flat, action, embed):
        """Single RSSM step: update deterministic state, compute posterior & prior."""
        # Deterministic update
        h = self.gru(torch.cat([s_flat, action], dim=-1), h)

        # Prior: p(s_t | h_t)
        prior_logits = self._to_logits(self.prior_net(h))
        s_prior = self._sample(prior_logits)

        # Posterior: q(s_t | h_t, e_t)
        post_logits = self._to_logits(self.posterior_net(torch.cat([h, embed], dim=-1)))
        s_post = self._sample(post_logits)

        return h, s_post, s_prior, post_logits, prior_logits

    def get_feature(self, state):
        """Concatenate deterministic + flattened stochastic for downstream use."""
        s_flat = state["stoch"].reshape(state["stoch"].shape[0], -1)
        return torch.cat([state["deter"], s_flat], dim=-1)

    def feature_dim(self):
        return self.deter_dim + self.stoch_flat

    def compute_loss(self, post_states, priors):
        """
        Compute dynamics regularization loss.
        - SIGReg mode: constrain continuous latent states to match N(0, I)
        - KL mode: constrain posterior q(s) close to prior p(s)

        SIGReg target is configurable (sigreg_target):
          "stoch"  → discrete one-hot (deprecated, mismatch with Gaussian)
          "deter"  → GRU deterministic state (continuous 1024-dim)
          "logits" → categorical logits before Gumbel-softmax (continuous 2048-dim)
          "deter+logits" → both, summed

        Returns:
            loss_dict
        """
        if self.regularization == "sigreg":
            total = torch.tensor(0.0, device=post_states[0]["deter"].device)
            reg_post = torch.tensor(0.0)
            reg_prior = torch.tensor(0.0)
            loss_type = self.sigreg_target

            # ---- Post SIGReg ----
            if "deter" in self.sigreg_target:
                post_deters = torch.stack([s["deter"] for s in post_states])
                post_deter = self.sigreg_deter(post_deters)
                total = total + self.sigreg_lambda * post_deter
                reg_post = post_deter

            if "logits" in self.sigreg_target or self.sigreg_target == "stoch":
                if self.sigreg_target == "stoch":
                    # Old: discrete one-hot (not recommended)
                    post_vals = torch.stack([
                        s["stoch"].reshape(s["stoch"].shape[0], -1) for s in post_states
                    ])
                else:
                    # logits: continuous, before Gumbel-softmax sampling
                    post_vals = torch.stack([
                        s["stoch_logits"].reshape(s["stoch_logits"].shape[0], -1)
                        for s in post_states
                    ])
                post_logits_reg = self.sigreg(post_vals)
                total = total + self.sigreg_lambda * post_logits_reg
                reg_post = reg_post + post_logits_reg

            # ---- Prior SIGReg (on prior_logits, continuous) ----
            if self.sigreg_target != "stoch" and self.sigreg is not None:
                prior_vals = torch.stack([
                    s["prior_logits"].reshape(s["prior_logits"].shape[0], -1)
                    for s in priors
                ])
                prior_logits_reg = self.sigreg(prior_vals)
                total = total + 0.5 * self.sigreg_lambda * prior_logits_reg
                reg_prior = prior_logits_reg

            return {
                "reg_type": f"sigreg_{loss_type}",
                "sigreg_loss": reg_post.item(),
                "sigreg_prior": reg_prior.item(),
                "dyn_loss": total,
                "total_dyn_loss": total,
            }

        elif self.regularization == "kl":
            # KL(q(s_t) || p(s_t)) for categorical distributions
            kl_losses = []
            for post_s, prior_s in zip(post_states, priors):
                post_logits = post_s["stoch_logits"].reshape(
                    -1, self.stoch_dim, self.stoch_classes
                )
                prior_logits = prior_s["prior_logits"].reshape(
                    -1, self.stoch_dim, self.stoch_classes
                )

                post_logprob = F.log_softmax(post_logits, dim=-1)
                prior_logprob = F.log_softmax(prior_logits, dim=-1)

                # KL(q || p) = sum_c q(c) * (log q(c) - log p(c))
                kl = (post_logprob.exp() * (post_logprob - prior_logprob)).sum(dim=-1).sum(dim=-1)
                if self.kl_free_bits > 0:
                    kl = torch.clamp(kl, min=self.kl_free_bits)
                kl_losses.append(kl.mean())

            kl_loss = torch.stack(kl_losses).mean()
            total = self.kl_beta * kl_loss
            return {
                "reg_type": "kl",
                "kl_loss": kl_loss.item(),
                "dyn_loss": total,
                "total_dyn_loss": total,
            }

        else:  # "none"
            device = post_states[0]["deter"].device
            return {
                "reg_type": "none",
                "dyn_loss": torch.tensor(0.0, device=device),
                "total_dyn_loss": torch.tensor(0.0, device=device),
            }


class MLPPredictor(nn.Module):
    """Simple MLP for reward prediction and continue (discount) prediction."""
    def __init__(self, input_dim, hidden_dim=512, output_dim=1, layers=2, act=nn.SiLU):
        super().__init__()
        net = []
        dim = input_dim
        for _ in range(layers):
            net.extend([nn.Linear(dim, hidden_dim), act()])
            dim = hidden_dim
        net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
