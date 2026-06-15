"""
World Model: combines Encoder + RSSM + (optional) Decoder + Reward + Continue + Phase.

Training step:
1. Encode BEV observations -> embeddings
2. RSSM posterior rollout with SIGReg/KL regularization
3. Reconstruction loss OR Barlow Twins temporal alignment (decoder-free)
4. Predict reward and continue (discount)
5. (Optional) Predict merge phase from RSSM features (Topology-Guided WM)
6. Total loss = rec_or_barlow + dyn + reward + continue + phase

Decoder-free mode (use_decoder=False):
- Drops ~12M decoder parameters (~30% reduction)
- Uses Barlow Twins cross-correlation on time-shifted RSSM features
- Encourages temporal invariance + feature diversity without pixel reconstruction

Topology-Guided mode (use_phase_head=True):
- Adds lightweight 3-class phase prediction head on RSSM features
- Phases: 0=pre-merge (ramp), 1=merge-zone, 2=post-merge (main road)
- Forces WM to encode topological awareness of ramp→main transition
- Labels derived from lanelet ID transitions (merge_frame_idx in npz)

Imagination step:
1. Use prior (no observation) to imagine future states
2. Used by Actor-Critic for policy training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import BEVEncoder, BEVDecoder, CNNFrontend, PixelUnshuffleFrontend
from .rssm import RSSM, MLPPredictor


class JEPAPredictor(nn.Module):
    """Predicts future RSSM features from current features + actions.

    Core idea (AD-L-JEPA / BYOL-Drive): a good world model encodes
    dynamics-relevant information — feature[t+k] should be predictable
    from feature[t] + action[t:t+k]. Features that encode noise are
    unpredictable → gradient pushes WM toward physically meaningful
    representations.

    Uses stop-gradient on the target (BYOL-style) to prevent the
    trivial solution of collapsing all features to a constant.
    """

    def __init__(self, feat_dim, action_dim, k=1, hidden_dim=1024):
        super().__init__()
        self.k = k
        in_dim = feat_dim + action_dim * k
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, feat_t, actions_window):
        """Predict feat[t+k] from feat[t] + actions[t:t+k].

        Args:
            feat_t: (N, feat_dim) features at time t
            actions_window: (N, action_dim * k) flattened action window
        Returns:
            feat_pred: (N, feat_dim) predicted features at t+k
        """
        x = torch.cat([feat_t, actions_window], dim=-1)
        return self.net(x)


class WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.use_decoder = getattr(c, 'use_decoder', True)
        self.barlow_lambda = getattr(c, 'barlow_lambda', 0.005)
        self.barlow_k = getattr(c, 'barlow_k', 1)

        # JEPA: future-feature prediction for dense content supervision
        self.use_jepa = getattr(c, 'use_jepa', False)
        self.jepa_weight = getattr(c, 'jepa_weight', 0.1)
        self.jepa_k = getattr(c, 'jepa_k', 1)

        # Optional CNN frontend for learnable downsampling (ablation A8)
        self.bev_downsample = getattr(c, 'bev_downsample', 'bilinear')
        self.cnn_factor = getattr(c, 'cnn_factor', 2)
        if self.bev_downsample == 'cnn':
            self.cnn_frontend = CNNFrontend(
                in_channels=c.input_channels,
                out_channels=64,
                factor=self.cnn_factor,
            )
            enc_input_channels = 64
            enc_input_size = self.cnn_frontend.output_size
        elif self.bev_downsample == 'unshuffle':
            self.cnn_frontend = PixelUnshuffleFrontend(
                in_channels=c.input_channels,
                out_channels=64,
                factor=self.cnn_factor,
            )
            enc_input_channels = 64
            enc_input_size = self.cnn_frontend.output_size
        else:
            self.cnn_frontend = None
            enc_input_channels = c.input_channels
            enc_input_size = c.bev_size

        # Encoder: BEV image -> embedding
        self.encoder = BEVEncoder(
            input_channels=enc_input_channels,
            input_size=enc_input_size,
            embed_dim=c.embed_dim,
            depth=c.enc_depth,
        )

        # RSSM with configurable regularization (SIGReg or KL)
        # Supports Phase-Conditional (Plan A) and MoE (Plan B) variants
        rssm_kwargs = dict(
            action_dim=c.action_dim,
            deter_dim=c.deter_dim,
            stoch_dim=c.stoch_dim,
            stoch_classes=c.stoch_classes,
            hidden_dim=c.hidden_dim,
            embed_dim=c.embed_dim,
            regularization=c.regularization,
            sigreg_lambda=c.sigreg_lambda,
            sigreg_projections=c.sigreg_projections,
            sigreg_target=getattr(c, 'sigreg_target', 'deter'),
            kl_beta=c.kl_beta,
            kl_free_bits=c.kl_free_bits,
            phase_conditional=getattr(c, 'rssm_phase_conditional', False),
            num_phases=3,
            moe=getattr(c, 'rssm_moe', False),
            moe_experts=getattr(c, 'rssm_moe_experts', 3),
            moe_lb_weight=getattr(c, 'rssm_moe_load_balance_weight', 0.01),
        )
        self.rssm = RSSM(**rssm_kwargs)

        # Decoder: embedding -> BEV reconstruction (optional)
        if self.use_decoder:
            self.decoder = BEVDecoder(
                output_channels=c.input_channels,
                output_size=c.bev_size,
                embed_dim=self.rssm.feature_dim(),  # deter + stoch_flat
                depth=c.enc_depth,
            )
        else:
            self.decoder = None

        # Reward predictor
        self.reward_head = MLPPredictor(
            input_dim=self.rssm.feature_dim(),
            hidden_dim=c.hidden_dim,
            output_dim=1,
            layers=2,
        )

        # Continue predictor (discount / episode termination)
        self.continue_head = MLPPredictor(
            input_dim=self.rssm.feature_dim(),
            hidden_dim=c.hidden_dim,
            output_dim=1,
            layers=2,
        )

        # Phase prediction head (Topology-Guided WM: 3-class merge stage)
        self.use_phase_head = getattr(c, 'use_phase_head', False)
        self.phase_head_weight = getattr(c, 'phase_head_weight', 1.0)
        self.merge_zone_frames = getattr(c, 'merge_zone_frames', 20)
        if self.use_phase_head:
            self.phase_head = MLPPredictor(
                input_dim=self.rssm.feature_dim(),
                hidden_dim=256,
                output_dim=3,  # pre-merge / merge-zone / post-merge
                layers=2,
            )
        else:
            self.phase_head = None

        # Ego speed prediction head (strong supervised signal from position deltas)
        self.use_speed_head = getattr(c, 'use_speed_head', False)
        self.speed_head_weight = getattr(c, 'speed_head_weight', 1.0)
        if self.use_speed_head:
            self.speed_head = MLPPredictor(
                input_dim=self.rssm.feature_dim(),
                hidden_dim=256,
                output_dim=1,  # scalar speed (m/s)
                layers=2,
            )
        else:
            self.speed_head = None

        # Coarse spatial layout prediction head (low-res BEV, captures "where things are")
        self.use_spatial_head = getattr(c, 'use_spatial_head', False)
        self.spatial_head_weight = getattr(c, 'spatial_head_weight', 0.5)
        self.spatial_res = getattr(c, 'spatial_head_resolution', 32)
        if self.use_spatial_head:
            spatial_hidden = 512
            self.spatial_head = nn.Sequential(
                nn.Linear(self.rssm.feature_dim(), spatial_hidden),
                nn.SiLU(),
                nn.Linear(spatial_hidden, 3 * self.spatial_res * self.spatial_res),
            )
        else:
            self.spatial_head = None

        # Trajectory prediction head: ego-centric future displacements
        # Captures ~10m lateral shift during merge (100x stronger signal than steer)
        self.use_traj_head = getattr(c, 'use_traj_head', False)
        self.traj_head_weight = getattr(c, 'traj_head_weight', 0.1)
        self.traj_horizon = getattr(c, 'traj_horizon', 10)
        if self.use_traj_head:
            traj_hidden = 512
            self.traj_head = nn.Sequential(
                nn.Linear(self.rssm.feature_dim(), traj_hidden),
                nn.SiLU(),
                nn.Linear(traj_hidden, traj_hidden),
                nn.SiLU(),
                nn.Linear(traj_hidden, self.traj_horizon * 2),  # H×2 = (Δx,Δy) for H frames
            )
        else:
            self.traj_head = None

        # Curvature prediction head: road geometry from ego position heading changes.
        # Zero-cost label: curvature = Δheading / Δdistance (rad/m).
        # Forces WM to encode "is the road curving right (ramp) or straight (main)?".
        self.use_curvature_head = getattr(c, 'use_curvature_head', False)
        self.curvature_head_weight = getattr(c, 'curvature_head_weight', 0.1)
        if self.use_curvature_head:
            self.curvature_head = MLPPredictor(
                input_dim=self.rssm.feature_dim(),
                hidden_dim=256,
                output_dim=1,  # scalar curvature (rad/m)
                layers=2,
            )
        else:
            self.curvature_head = None

        # Surrounding vehicle prediction head: ego-centric (dx, dy, vx, vy) of N nearest cars.
        # Forces WM to encode "where are other vehicles and how fast are they moving?".
        self.use_vehicle_head = getattr(c, 'use_vehicle_head', False)
        self.vehicle_head_weight = getattr(c, 'vehicle_head_weight', 0.1)
        self.n_surrounding = getattr(c, 'n_surrounding_vehicles', 8)
        if self.use_vehicle_head:
            self.vehicle_head = nn.Sequential(
                nn.Linear(self.rssm.feature_dim(), 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, self.n_surrounding * 4),  # N × (dx, dy, vx, vy)
            )
        else:
            self.vehicle_head = None

        # JEPA predictor: feat[t] + action[t:t+k] → predicted feat[t+k]
        if self.use_jepa:
            feat_dim = self.rssm.feature_dim()
            self.jepa_predictor = JEPAPredictor(
                feat_dim=feat_dim,
                action_dim=c.action_dim,
                k=self.jepa_k,
                hidden_dim=getattr(c, 'jepa_hidden', 1024),
            )
        else:
            self.jepa_predictor = None

    def get_initial_state(self, batch_size, device=None):
        return self.rssm.initial_state(batch_size, device)

    def encode(self, obs):
        """obs: (batch, C, H, W) -> (batch, embed_dim)"""
        if self.cnn_frontend is not None:
            # CNNFrontend was trained on 300x300; during P4 eval the BEV
            # may arrive at 64x64, which would produce the wrong flat_size
            # for BEVEncoder's linear head (1024 vs 25600).
            if obs.shape[-1] != 300:
                obs = F.interpolate(
                    obs, size=(300, 300),
                    mode='bilinear', align_corners=False,
                )
            obs = self.cnn_frontend(obs)
        return self.encoder(obs)

    def observe(self, embeds, actions, prev_state):
        """Posterior rollout with observations."""
        return self.rssm.observe(embeds, actions, prev_state)

    def imagine(self, actions, prev_state):
        """Prior rollout without observations (for actor-critic)."""
        return self.rssm.imagine(actions, prev_state)

    def get_feature(self, state):
        """Get concatenated deterministic + stochastic features."""
        return self.rssm.get_feature(state)

    def feature_dim(self):
        return self.rssm.feature_dim()

    def compute_barlow_loss(self, features):
        """
        Decorrelation loss on time-shifted RSSM features.

        Driving-specific Barlow: only penalizes off-diagonal cross-correlation.
        - On-diagonal (temporal invariance) is NOT enforced because driving
          scenes genuinely change over time (ego motion, other vehicles).
        - Off-diagonal (decorrelation) forces each feature dimension to carry
          unique information, preventing dimensional collapse.

        Args:
            features: (seq_len, batch, feat_dim) RSSM features (deter + stoch_flat)
        Returns:
            loss: scalar tensor (mean squared off-diagonal cross-correlation)
        """
        k = self.barlow_k
        seq_len = features.shape[0]
        if seq_len <= k:
            return torch.tensor(0.0, device=features.device)

        # z_t and z_{t+k}: temporally shifted views
        z_t = features[:-k].reshape(-1, features.shape[-1])
        z_tk = features[k:].reshape(-1, features.shape[-1])

        # Normalize along batch dimension (zero mean, unit variance)
        z_t = (z_t - z_t.mean(dim=0, keepdim=True)) / (z_t.std(dim=0, keepdim=True) + 1e-5)
        z_tk = (z_tk - z_tk.mean(dim=0, keepdim=True)) / (z_tk.std(dim=0, keepdim=True) + 1e-5)

        # Cross-correlation matrix: (D, D)
        N = z_t.shape[0]
        c = (z_t.T @ z_tk) / N

        # Off-diagonal only: decorrelate feature dimensions across time
        diag = torch.diagonal(c)
        off_diag = c - torch.diag(diag)

        # Mean squared off-diagonal, normalized by feature dim
        off_loss = (off_diag ** 2).mean()

        return off_loss

    def compute_world_loss(self, obs_seq, actions, rewards, continues, prev_state,
                           phase_labels=None, speed_labels=None, position_labels=None,
                           surrounding_labels=None):
        """
        Compute all world model losses for one training step.

        Decoder mode:      reconstruction MSE in symlog space
        Decoder-free mode: Barlow Twins temporal alignment on RSSM features
        Topology-Guided:   + merge-phase classification on RSSM features
        Speed prediction:  + ego speed regression from RSSM features
        JEPA:              + future-feature prediction in latent space
        Trajectory:        + ego-centric future displacement prediction

        Args:
            obs_seq: (seq_len, batch, C, H, W) BEV observations
            actions: (seq_len, batch, action_dim)
            rewards: (seq_len, batch) rewards
            continues: (seq_len, batch) discount factors (1 - done)
            prev_state: initial RSSM state
            phase_labels: (seq_len, batch) int64, 0=pre-merge 1=merge 2=post-merge
            speed_labels: (seq_len, batch) float32, ego speed in m/s
            position_labels: (seq_len+H, batch, 2) float32, world-coord positions
        Returns:
            loss: total weighted loss
            metrics: dict of individual losses for logging
        """
        device = obs_seq.device
        seq_len, batch = obs_seq.shape[:2]

        # 1. Encode all observations (with optional CNN frontend)
        obs_flat = obs_seq.reshape(-1, *obs_seq.shape[2:])
        if self.cnn_frontend is not None:
            obs_flat = self.cnn_frontend(obs_flat)
        embeds = self.encoder(obs_flat)
        embeds = embeds.reshape(seq_len, batch, -1)

        # 2. RSSM posterior rollout (pass phase labels for Phase-Conditional / MoE)
        post_states, priors = self.rssm.observe(embeds, actions, prev_state,
                                                 phase_labels=phase_labels)

        # 3. Dynamics regularization loss (SIGReg or KL)
        dyn_metrics = self.rssm.compute_loss(post_states, priors,
                                              phase_labels=phase_labels)
        dyn_loss = dyn_metrics["total_dyn_loss"]

        # 4. Feature extraction (shared by both modes)
        features = torch.stack([self.rssm.get_feature(s) for s in post_states])  # (T, B, feat_dim)

        # 5. Reconstruction OR Barlow Twins loss
        if self.use_decoder:
            recon = self.decoder(features.reshape(-1, features.shape[-1]))
            recon = recon.reshape(seq_len, batch, *obs_seq.shape[2:])
            obs_symlog = torch.sign(obs_seq.float()) * torch.log1p(torch.abs(obs_seq.float()))
            rec_loss = F.mse_loss(recon, obs_symlog)
            barlow_loss = torch.tensor(0.0, device=device)
        else:
            rec_loss = torch.tensor(0.0, device=device)
            barlow_loss = self.compute_barlow_loss(features)

        # 6. Reward prediction loss
        reward_pred = self.reward_head(features.reshape(-1, features.shape[-1]))
        reward_target = Symlog.forward(rewards.reshape(-1))
        rew_loss = F.mse_loss(reward_pred.squeeze(), reward_target)

        # 7. Continue prediction loss (binary cross entropy)
        cont_pred = self.continue_head(features.reshape(-1, features.shape[-1]))
        cont_target = continues.reshape(-1).unsqueeze(-1)
        con_loss = F.binary_cross_entropy_with_logits(cont_pred, cont_target)

        # 8. Phase prediction loss (Topology-Guided WM: auxiliary 3-class classification)
        phase_loss = torch.tensor(0.0, device=device)
        phase_acc = torch.tensor(0.0, device=device)
        if self.use_phase_head and phase_labels is not None:
            phase_pred = self.phase_head(features.reshape(-1, features.shape[-1]))
            phase_target = phase_labels.reshape(-1).long()
            phase_loss = F.cross_entropy(phase_pred, phase_target)
            phase_acc = (phase_pred.argmax(-1) == phase_target).float().mean()

        # 9. Speed prediction loss (symlog, same as reward head)
        speed_loss = torch.tensor(0.0, device=device)
        if self.use_speed_head and speed_labels is not None:
            speed_pred = self.speed_head(features.reshape(-1, features.shape[-1]))
            speed_target = speed_labels.reshape(-1).clamp(0, 50)  # clamp outliers
            speed_target_symlog = Symlog.forward(speed_target)
            speed_loss = F.mse_loss(speed_pred.squeeze(), speed_target_symlog)

        # 10. Coarse spatial layout prediction loss (low-res BEV, captures "where things are")
        spatial_loss = torch.tensor(0.0, device=device)
        if self.use_spatial_head:
            # Downsample the last frame of obs_seq as spatial target
            with torch.no_grad():
                R = self.spatial_res
                obs_target = F.interpolate(
                    obs_seq[-1], size=(R, R), mode='bilinear', align_corners=False
                )  # (B, 3, R, R)
                obs_target_symlog = torch.sign(obs_target) * torch.log1p(torch.abs(obs_target))
                obs_target_flat = obs_target_symlog.reshape(batch, -1)  # (B, 3*R*R)

            spatial_pred = self.spatial_head(features[-1])  # (B, 3*R*R)
            spatial_loss = F.mse_loss(spatial_pred, obs_target_flat)

        # 11. JEPA future-feature prediction loss (dense 3072-dim self-supervised signal)
        jepa_loss = torch.tensor(0.0, device=device)
        if self.use_jepa:
            k = self.jepa_k
            if seq_len > k:
                # feature[t] and feature[t+k]
                feat_t = features[:-k].reshape(-1, features.shape[-1])    # ((T-k)*B, F)
                feat_tk = features[k:].reshape(-1, features.shape[-1])    # ((T-k)*B, F)

                # Action window: actions[t:t+k] concatenated
                # actions: (T, B, 2) → flatten window
                act_window = []
                for i in range(k):
                    act_window.append(actions[i:seq_len - k + i].reshape(-1, actions.shape[-1]))
                act_window = torch.cat(act_window, dim=-1)  # ((T-k)*B, 2*k)

                # Predict feature[t+k] from feature[t] + action window
                feat_pred = self.jepa_predictor(feat_t, act_window)
                # Stop-gradient on target: BYOL-style prevents collapse
                jepa_loss = F.mse_loss(feat_pred, feat_tk.detach())
            else:
                if not hasattr(self, '_jepa_k_warned'):
                    import warnings
                    warnings.warn(
                        f"JEPA disabled: k={k} >= seq_len={seq_len}. "
                        f"Use --batch-length >= {k+1} to enable JEPA."
                    )
                    self._jepa_k_warned = True

        # 12. Trajectory prediction loss (ego-centric future displacements)
        # Captures ~10m lateral shift during merge — 100x stronger than steer signal.
        # H=125 (5s) forces WM to encode ramp curvature and full merge trajectory.
        traj_loss = torch.tensor(0.0, device=device)
        curvature_loss = torch.tensor(0.0, device=device)
        traj_headings = None  # reused by curvature head
        if (self.use_traj_head or self.use_curvature_head) and position_labels is not None:
            H = max(self.traj_horizon, 1) if self.use_traj_head else 10
            pos_seq_len = position_labels.shape[0]
            if pos_seq_len >= seq_len + H:
                with torch.no_grad():
                    dp = position_labels[1:] - position_labels[:-1]
                    headings = torch.atan2(dp[..., 1], dp[..., 0])
                    headings = torch.cat([headings, headings[-1:]], dim=0)

                # --- TrajHead (if enabled) ---
                if self.use_traj_head:
                    T_eff = min(seq_len, pos_seq_len - H)
                    traj_targets = []
                    for t in range(T_eff):
                        pos_t = position_labels[t]
                        h_t = headings[t]
                        ego_disp = []
                        for k in range(1, H + 1):
                            delta = position_labels[t + k] - pos_t
                            cos_h = torch.cos(-h_t)
                            sin_h = torch.sin(-h_t)
                            ego_x = delta[..., 0] * cos_h - delta[..., 1] * sin_h
                            ego_y = delta[..., 0] * sin_h + delta[..., 1] * cos_h
                            ego_disp.append(ego_x)
                            ego_disp.append(ego_y)
                        traj_targets.append(torch.stack(ego_disp, dim=-1))
                    traj_targets = torch.stack(traj_targets)
                    feats_for_traj = features[:T_eff].reshape(-1, features.shape[-1])
                    traj_pred = self.traj_head(feats_for_traj)
                    traj_targets_flat = traj_targets.reshape(-1, H * 2)
                    traj_targets_symlog = Symlog.forward(traj_targets_flat)
                    traj_loss = F.mse_loss(traj_pred, traj_targets_symlog)

                # --- CurvatureHead (if enabled, zero-cost: computed from positions) ---
                if self.use_curvature_head:
                    # Smooth curvature via 15-frame Menger curvature (circle-fit discrete form).
                    # Window: ±7 frames (0.28s each side, ~3-5m @ highway speed).
                    # Signed: cross-product direction → κ>0=left turn, κ<0=right turn.
                    # Scaled ×100 for MSE visibility (raw κ ~1e-4 to 1e-2 rad/m).
                    CURV_SCALE = 100.0
                    curv_window = 15
                    curv_stride = 5
                    curv_labels = []
                    for t in range(0, seq_len, curv_stride):
                        w_start = max(0, t - curv_window // 2)
                        w_end = min(pos_seq_len, t + curv_window // 2 + 1)
                        window = position_labels[w_start:w_end]  # (W, B, 2)
                        # Menger curvature per consecutive triplet
                        p0 = window[:-2]   # (W-2, B, 2)
                        p1 = window[1:-1]
                        p2 = window[2:]
                        a = torch.norm(p1 - p0, dim=-1).clamp(min=1e-6)  # (W-2, B)
                        b = torch.norm(p2 - p1, dim=-1).clamp(min=1e-6)
                        c = torch.norm(p2 - p0, dim=-1).clamp(min=1e-6)
                        # Triangle area (×0.5 cancels in Menger formula)
                        v1 = p1 - p0
                        v2 = p2 - p0
                        cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
                        sign = torch.sign(cross)
                        area_half = 0.5 * torch.abs(cross)
                        # κ = 4 * area / (a * b * c), averaged over window
                        menger_curv = sign * (4.0 * area_half) / (a * b * c + 1e-6)  # (W-2, B)
                        curv_t = menger_curv.mean(dim=0)  # (B,)
                        curv_labels.append(curv_t * CURV_SCALE)
                    # Assemble and compute Huber loss (δ=1.0 suits κ×100 range)
                    curv_labels = torch.stack(curv_labels)  # (T_c, B)
                    T_c = curv_labels.shape[0]
                    feats_for_curv = features[:T_c * curv_stride:curv_stride].reshape(-1, features.shape[-1])
                    curv_pred = self.curvature_head(feats_for_curv).squeeze(-1)  # (T_c * B,)
                    curvature_loss = F.huber_loss(curv_pred, curv_labels.reshape(-1), delta=1.0)

        # 13. Surrounding vehicle prediction loss
        # Predicts ego-centric (dx, dy, vx, vy) of N nearest vehicles from RSSM features.
        # Labels from exiD CSV (pre-computed by add_surrounding_vehicles.py).
        vehicle_loss = torch.tensor(0.0, device=device)
        if self.use_vehicle_head and surrounding_labels is not None:
            # surrounding_labels: (B, seq_len, N, 4) or pre-permuted
            surr_labels = surrounding_labels  # (seq_len, B, N, 4) after permute
            surr_flat = surr_labels.reshape(-1, self.n_surrounding * 4)  # (seq_len*B, N*4)
            # Mask: exclude frames with no vehicles (all zeros)
            mask = (surr_flat.abs().sum(dim=-1) > 1e-6).float()
            if mask.sum() > 0:
                veh_pred = self.vehicle_head(features.reshape(-1, features.shape[-1]))
                veh_pred_symlog = Symlog.forward(veh_pred)
                surr_symlog = Symlog.forward(surr_flat)
                vehicle_loss = (F.mse_loss(veh_pred_symlog, surr_symlog, reduction='none')
                                .mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)

        # 14. Total weighted loss
        scales = self.config.loss_scales
        total_loss = (
            scales["rec"] * (rec_loss + self.barlow_lambda * barlow_loss)
            + scales["dyn"] * dyn_loss
            + scales["rew"] * rew_loss
            + scales["con"] * con_loss
            + self.phase_head_weight * phase_loss
            + self.speed_head_weight * speed_loss
            + self.spatial_head_weight * spatial_loss
            + self.jepa_weight * jepa_loss
            + self.traj_head_weight * traj_loss
            + self.curvature_head_weight * curvature_loss
            + self.vehicle_head_weight * vehicle_loss
        )

        # Find the primary reg loss value for logging
        dyn_keys = [k for k in dyn_metrics if k.endswith("_loss") and k not in ("dyn_loss", "total_dyn_loss")]
        primary_reg = dyn_metrics[dyn_keys[0]] if dyn_keys else dyn_loss.item()

        metrics = {
            "loss_total": total_loss.item(),
            "loss_rec": rec_loss.item() if self.use_decoder else barlow_loss.item(),
            "loss_barlow": barlow_loss.item(),
            "loss_dyn": dyn_loss.item(),
            "loss_rew": rew_loss.item(),
            "loss_con": con_loss.item(),
            f"{dyn_metrics.get('reg_type', 'dyn')}_loss": primary_reg,
        }
        if self.use_phase_head:
            metrics["loss_phase"] = phase_loss.item()
            metrics["phase_accuracy"] = phase_acc.item()
        if self.use_speed_head:
            metrics["loss_speed"] = speed_loss.item()
        if self.use_spatial_head:
            metrics["loss_spatial"] = spatial_loss.item()
        if self.use_jepa:
            metrics["loss_jepa"] = jepa_loss.item()
        if self.use_traj_head:
            metrics["loss_traj"] = traj_loss.item()
        if self.use_curvature_head:
            metrics["loss_curvature"] = curvature_loss.item()
        if self.use_vehicle_head:
            metrics["loss_vehicle"] = vehicle_loss.item()
        # RSSM MoE / Phase-Conditional auxiliary losses (already in total_dyn_loss)
        if "loss_moe_lb" in dyn_metrics:
            metrics["loss_moe_lb"] = dyn_metrics["loss_moe_lb"]
        if "loss_phase_cond" in dyn_metrics:
            metrics["loss_phase_cond"] = dyn_metrics["loss_phase_cond"]

        return total_loss, metrics


class Symlog:
    @staticmethod
    def forward(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
