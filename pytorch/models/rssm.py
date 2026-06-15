"""
RSSM with SIGReg: Recurrent State Space Model replacing KL divergence with SIGReg.

SIGReg acts on continuous latent states (deter / stoch_logits) rather than
discrete one-hot samples, eliminating the discrete→continuous mismatch.

Extensions:
  - Phase-Conditional RSSM (Plan A): 3 independent dynamics heads gated by GT phase labels
  - MoE-Style RSSM (Plan B): Router + 3 expert dynamics heads with softmax gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sigreg import SIGReg


def _make_head(in_dim, hidden_dim, out_dim, act=nn.SiLU):
    """Create a 2-layer MLP head matching the original RSSM design."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        act(),
        nn.Linear(hidden_dim, out_dim),
    )


class RSSM(nn.Module):
    def __init__(self, action_dim=2, deter_dim=1024, stoch_dim=32,
                 stoch_classes=64, hidden_dim=1024, embed_dim=512,
                 regularization="sigreg",
                 sigreg_lambda=0.1, sigreg_projections=1024,
                 sigreg_target="deter",
                 kl_beta=1.0, kl_free_bits=1.0,
                 gru_layers=1, act=nn.SiLU,
                 # --- Plan A: Phase-Conditional ---
                 phase_conditional=False, num_phases=3,
                 # --- Plan B: MoE ---
                 moe=False, moe_experts=3, moe_lb_weight=0.01,
                 ):
        """
        Args:
            regularization: "sigreg" | "kl" | "none"
            sigreg_target: which continuous state SIGReg acts on:
                "stoch"  - discrete one-hot (old, avoid)
                "deter"  - GRU deterministic state (1024-dim continuous)
                "logits" - categorical logits before sampling (2048-dim continuous)
                "deter+logits" - both concatenated

            phase_conditional (Plan A): use 3 independent prior/posterior nets
                gated by ground-truth phase labels (0/1/2).
            moe (Plan B): router + 3 expert dynamics heads with softmax gating.
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
        self.phase_conditional = phase_conditional
        self.num_phases = num_phases
        self.moe = moe
        self.moe_experts = moe_experts
        self.moe_lb_weight = moe_lb_weight

        assert not (phase_conditional and moe), \
            "Phase-Conditional and MoE are mutually exclusive"

        # GRU for deterministic state: h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])
        self.gru = nn.GRUCell(
            input_size=self.stoch_flat + action_dim,
            hidden_size=deter_dim
        )

        if moe:
            # Plan B: Router + N expert prior/posterior nets
            self.prior_nets = nn.ModuleList([
                _make_head(deter_dim, hidden_dim, self.stoch_flat, act)
                for _ in range(moe_experts)
            ])
            self.posterior_nets = nn.ModuleList([
                _make_head(deter_dim + embed_dim, hidden_dim, self.stoch_flat, act)
                for _ in range(moe_experts)
            ])
            # Router: MLP(h) → softmax over experts
            self.router = nn.Sequential(
                nn.Linear(deter_dim, 256),
                act(),
                nn.Linear(256, moe_experts),
            )
            # For backward-compat: single prior/posterior reference
            self.prior_net = None
            self.posterior_net = None
        elif phase_conditional:
            # Plan A: 3 independent heads, gated by ground-truth phase labels
            self.prior_nets = nn.ModuleList([
                _make_head(deter_dim, hidden_dim, self.stoch_flat, act)
                for _ in range(num_phases)
            ])
            self.posterior_nets = nn.ModuleList([
                _make_head(deter_dim + embed_dim, hidden_dim, self.stoch_flat, act)
                for _ in range(num_phases)
            ])
            # Phase classifier for imagine() when we don't have GT labels
            self.phase_classifier = nn.Sequential(
                nn.Linear(deter_dim, 256),
                act(),
                nn.Linear(256, num_phases),
            )
            self.prior_net = None
            self.posterior_net = None
        else:
            # Original single-head design (backward compatible)
            self.prior_net = _make_head(deter_dim, hidden_dim, self.stoch_flat, act)
            self.posterior_net = _make_head(deter_dim + embed_dim, hidden_dim, self.stoch_flat, act)
            self.prior_nets = None
            self.posterior_nets = None

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
        if self.training:
            one_hot = F.gumbel_softmax(logits, hard=True, tau=1.0)
        else:
            indices = logits.argmax(dim=-1)
            one_hot = F.one_hot(indices, self.stoch_classes).float()
        return one_hot

    # ------------------------------------------------------------------
    #  Single-head methods (original, backward compatible)
    # ------------------------------------------------------------------

    def _step(self, h, s_flat, action, embed):
        """Single RSSM step: update deterministic state, compute posterior & prior."""
        h = self.gru(torch.cat([s_flat, action], dim=-1), h)
        prior_logits = self._to_logits(self.prior_net(h))
        s_prior = self._sample(prior_logits)
        post_logits = self._to_logits(self.posterior_net(torch.cat([h, embed], dim=-1)))
        s_post = self._sample(post_logits)
        return h, s_post, s_prior, post_logits, prior_logits

    # ------------------------------------------------------------------
    #  Plan A: Phase-Conditional helpers
    # ------------------------------------------------------------------

    def _step_phased(self, h, s_flat, action, embed, phase_idx):
        """Single step using phase-specific prior/posterior nets."""
        h = self.gru(torch.cat([s_flat, action], dim=-1), h)
        prior_logits = self._to_logits(self.prior_nets[phase_idx](h))
        s_prior = self._sample(prior_logits)
        post_logits = self._to_logits(self.posterior_nets[phase_idx](
            torch.cat([h, embed], dim=-1)))
        s_post = self._sample(post_logits)
        return h, s_post, s_prior, post_logits, prior_logits

    # ------------------------------------------------------------------
    #  Plan B: MoE helpers
    # ------------------------------------------------------------------

    def _step_moe(self, h, s_flat, action, embed):
        """Single step with router-weighted expert prior/posterior nets.

        Returns:
            (h, s_post, s_prior, post_logits, prior_logits, router_probs)
        """
        h = self.gru(torch.cat([s_flat, action], dim=-1), h)

        # Router: softmax over experts from deterministic state
        router_logits = self.router(h)                       # (B, N)
        router_probs = F.softmax(router_logits, dim=-1)      # (B, N)

        # Weighted sum of expert outputs
        prior_logits = torch.zeros_like(
            self.prior_nets[0](h)
        )
        for i in range(self.moe_experts):
            expert_out = self.prior_nets[i](h)
            prior_logits = prior_logits + router_probs[:, i:i+1] * expert_out

        post_cat = torch.cat([h, embed], dim=-1)
        post_logits = torch.zeros_like(
            self.posterior_nets[0](post_cat)
        )
        for i in range(self.moe_experts):
            expert_out = self.posterior_nets[i](post_cat)
            post_logits = post_logits + router_probs[:, i:i+1] * expert_out

        prior_logits = self._to_logits(prior_logits)
        s_prior = self._sample(prior_logits)
        post_logits = self._to_logits(post_logits)
        s_post = self._sample(post_logits)

        return h, s_post, s_prior, post_logits, prior_logits, router_probs

    # ------------------------------------------------------------------
    #  observe: posterior rollout (used during WM training)
    # ------------------------------------------------------------------

    def observe(self, embeds, actions, prev_state, phase_labels=None):
        """
        Posterior rollout: given observation embeddings and actions,
        compute the full state sequence using posterior (q).

        Args:
            embeds: (seq_len, batch, embed_dim) encoder outputs
            actions: (seq_len, batch, action_dim)
            prev_state: dict with 'deter', 'stoch'
            phase_labels: (seq_len, batch) int64, only used if phase_conditional=True
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
        router_probs_list = []  # only used for MoE

        for t in range(seq_len):
            if self.phase_conditional:
                # Plan A: route by ground-truth phase label
                assert phase_labels is not None, \
                    "phase_labels required for phase_conditional RSSM"
                phase_idx = int(phase_labels[t, 0].item())  # all batch items share phase
                h, s_post, s_prior, post_logits, prior_logits = self._step_phased(
                    h, s_flat, actions[t], embeds[t], phase_idx
                )
            elif self.moe:
                # Plan B: router-weighted expert outputs
                h, s_post, s_prior, post_logits, prior_logits, r_probs = self._step_moe(
                    h, s_flat, actions[t], embeds[t]
                )
                router_probs_list.append(r_probs)
            else:
                # Original: single-head
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

        if self.moe and router_probs_list:
            self._router_probs = torch.stack(router_probs_list)  # (T, B, N)

        return post_states, priors

    # ------------------------------------------------------------------
    #  imagine: prior rollout (used during imagination / actor training)
    # ------------------------------------------------------------------

    def imagine(self, actions, prev_state):
        """
        Prior rollout: imagine future states without observations.
        Used for actor-critic training in imagination.

        For Phase-Conditional RSSM: uses phase_classifier to predict phase from deter,
        then routes through the appropriate head.

        For MoE RSSM: uses router for expert gating (same as observe).

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

            if self.phase_conditional:
                # Use phase classifier to predict phase, then route
                phase_logits = self.phase_classifier(h)
                phase_pred = phase_logits.argmax(dim=-1)
                # Route each batch item through its predicted phase head
                # For efficiency: use mode (most common predicted phase)
                mode_phase = int(phase_pred.mode().values.item()) if batch > 1 else int(phase_pred.item())
                prior_logits = self._to_logits(self.prior_nets[mode_phase](h))
            elif self.moe:
                router_logits = self.router(h)
                router_probs = F.softmax(router_logits, dim=-1)
                prior_logits = torch.zeros_like(self.prior_nets[0](h))
                for i in range(self.moe_experts):
                    expert_out = self.prior_nets[i](h)
                    prior_logits = prior_logits + router_probs[:, i:i+1] * expert_out
                prior_logits = self._to_logits(prior_logits)
            else:
                prior_logits = self._to_logits(self.prior_net(h))

            s = self._sample(prior_logits)

            states.append({
                "deter": h,
                "stoch": s,
            })
            s_flat = s.reshape(batch, -1)

        return states

    # ------------------------------------------------------------------
    #  get_feature / feature_dim
    # ------------------------------------------------------------------

    def get_feature(self, state):
        """Concatenate deterministic + flattened stochastic for downstream use."""
        s_flat = state["stoch"].reshape(state["stoch"].shape[0], -1)
        return torch.cat([state["deter"], s_flat], dim=-1)

    def feature_dim(self):
        return self.deter_dim + self.stoch_flat

    # ------------------------------------------------------------------
    #  compute_loss: regularization + optional phase/MoE auxiliary losses
    # ------------------------------------------------------------------

    def compute_loss(self, post_states, priors, phase_labels=None):
        """
        Compute dynamics regularization loss.

        Returns dict with keys:
          reg_type, dyn_loss, total_dyn_loss
          + loss_phase_cond (if phase_conditional and phase_labels provided)
          + loss_moe_lb (if moe)
        """
        device = post_states[0]["deter"].device
        metrics = {}

        # ---- MoE load-balancing loss ----
        moe_lb_loss = torch.tensor(0.0, device=device)
        if self.moe and hasattr(self, '_router_probs') and self._router_probs is not None:
            # Compute entropy-based load balancing:
            # 1. Average router prob per expert across batch
            avg_probs = self._router_probs.mean(dim=0).mean(dim=0)  # (N,)
            # 2. Target: uniform distribution (1/N each)
            target = torch.full_like(avg_probs, 1.0 / self.moe_experts)
            # 3. KL(target || avg_probs) penalizes expert collapse
            moe_lb_loss = (target * (torch.log(target + 1e-8) - torch.log(avg_probs + 1e-8))).sum()
            metrics["loss_moe_lb"] = moe_lb_loss.item()

        # ---- Phase-conditional classification loss ----
        phase_cond_loss = torch.tensor(0.0, device=device)
        if self.phase_conditional and phase_labels is not None:
            # Train phase_classifier on post_states deters
            deters = torch.stack([s["deter"] for s in post_states])  # (T, B, F)
            phase_logits = self.phase_classifier(deters.reshape(-1, self.deter_dim))
            phase_targets = phase_labels.reshape(-1).long()
            phase_cond_loss = F.cross_entropy(phase_logits, phase_targets)
            metrics["loss_phase_cond"] = phase_cond_loss.item()

        # ---- SIGReg / KL ----
        if self.regularization == "sigreg":
            total = torch.tensor(0.0, device=device)
            reg_post = torch.tensor(0.0)
            reg_prior = torch.tensor(0.0)
            loss_type = self.sigreg_target

            if "deter" in self.sigreg_target:
                post_deters = torch.stack([s["deter"] for s in post_states])
                post_deter = self.sigreg_deter(post_deters)
                total = total + self.sigreg_lambda * post_deter
                reg_post = post_deter

            if "logits" in self.sigreg_target or self.sigreg_target == "stoch":
                if self.sigreg_target == "stoch":
                    post_vals = torch.stack([
                        s["stoch"].reshape(s["stoch"].shape[0], -1) for s in post_states
                    ])
                else:
                    post_vals = torch.stack([
                        s["stoch_logits"].reshape(s["stoch_logits"].shape[0], -1)
                        for s in post_states
                    ])
                post_logits_reg = self.sigreg(post_vals)
                total = total + self.sigreg_lambda * post_logits_reg
                reg_post = reg_post + post_logits_reg

            if self.sigreg_target != "stoch" and self.sigreg is not None:
                prior_vals = torch.stack([
                    s["prior_logits"].reshape(s["prior_logits"].shape[0], -1)
                    for s in priors
                ])
                prior_logits_reg = self.sigreg(prior_vals)
                total = total + 0.5 * self.sigreg_lambda * prior_logits_reg
                reg_prior = prior_logits_reg

            metrics.update({
                "reg_type": f"sigreg_{loss_type}",
                "sigreg_loss": reg_post.item(),
                "sigreg_prior": reg_prior.item(),
                "dyn_loss": total,
                "total_dyn_loss": total + moe_lb_loss * self.moe_lb_weight + phase_cond_loss,
            })

        elif self.regularization == "kl":
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
                kl = (post_logprob.exp() * (post_logprob - prior_logprob)).sum(dim=-1).sum(dim=-1)
                if self.kl_free_bits > 0:
                    kl = torch.clamp(kl, min=self.kl_free_bits)
                kl_losses.append(kl.mean())

            kl_loss = torch.stack(kl_losses).mean()
            total = self.kl_beta * kl_loss
            metrics.update({
                "reg_type": "kl",
                "kl_loss": kl_loss.item(),
                "dyn_loss": total,
                "total_dyn_loss": total + moe_lb_loss * self.moe_lb_weight + phase_cond_loss,
            })

        else:  # "none"
            metrics.update({
                "reg_type": "none",
                "dyn_loss": torch.tensor(0.0, device=device),
                "total_dyn_loss": moe_lb_loss * self.moe_lb_weight + phase_cond_loss,
            })

        return metrics


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
