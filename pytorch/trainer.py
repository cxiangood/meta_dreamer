"""
SIGReg-Dreamer Trainer: Three-phase training for Highway Merge-in Decision.

Phase 2: Offline world model pretraining on exiD data
Phase 3: Imagination policy training (frozen world model)
Phase 4: Online finetuning in MetaDrive simulator
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

try:
    import wandb
    _has_wandb = True
except (ImportError, AttributeError) as e:
    wandb = None
    _has_wandb = False
    print(f"[Trainer] wandb not available: {e}")

try:
    from torch.utils.tensorboard import SummaryWriter
    _has_tb = True
except (ImportError, AttributeError) as e:
    SummaryWriter = None
    _has_tb = False
    print(f"[Trainer] tensorboard not available: {e}")

from models import WorldModel, Actor, Critic
from models.crash_detector import CrashDetector, CrashFeatureBuffer


class Trainer:
    def __init__(self, config, env_factory=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")

        # Create models
        self.world_model = WorldModel(config).to(self.device)
        feat_dim = self.world_model.feature_dim()

        self.actor = Actor(
            feat_dim, config.action_dim,
            config.actor_hidden, config.actor_layers,
        ).to(self.device)
        self.critic = Critic(
            feat_dim, config.critic_hidden, config.critic_layers,
        ).to(self.device)

        # Slow (EMA) critic for stable targets
        self.slow_critic = Critic(
            feat_dim, config.critic_hidden, config.critic_layers,
        ).to(self.device)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        self.slow_tau = config.slow_value_tau

        # Optimizers
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=config.world_lr, eps=1e-5
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr, eps=1e-5
        )

        # Environment (optional, used in Phase 4)
        self.env_factory = env_factory

        # Logging
        self.logdir = config.logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.logdir, "tb")) if _has_tb else None
        self.metrics_history = defaultdict(list)
        self.global_step = 0

        # WandB
        self.wandb_run = None
        if _has_wandb and config.use_wandb:
            try:
                wandb.login(key=config.wandb_api_key)
                run_name = config.wandb_run_name or f"{config.regularization}_p2_s{config.seed}"
                self.wandb_run = wandb.init(
                    project="sigreg-dreamer",
                    name=run_name,
                    config=config.to_dict(),
                    dir=self.logdir,
                    sync_tensorboard=False,
                    settings=wandb.Settings(init_timeout=120),
                )
                print(f"[Trainer] WandB initialized: {self.wandb_run.name}")
            except Exception as e:
                print(f"[Trainer] WandB init failed: {e}")

        # Online crash detector: trained on real episode RSSM features
        self.crash_detector = CrashDetector(feat_dim).to(self.device)
        self.crash_buffer = CrashFeatureBuffer(max_samples=10000)
        self.crash_optimizer = torch.optim.Adam(
            self.crash_detector.parameters(), lr=1e-4, eps=1e-5
        )
        self.crash_ready = False  # True once buffer has enough positive samples

    # ========================================================================
    #  Main entry point
    # ========================================================================

    def train(self):
        cfg = self.config
        if cfg.train_phase == "phase2" and cfg.end_to_end:
            self._train_phase2_e2e()
        elif cfg.train_phase == "phase2":
            self._train_phase2()
        elif cfg.train_phase == "phase3":
            if cfg.use_dapo:
                self._train_phase3_dapo(K=cfg.dapo_group_size)
            else:
                self._train_phase3()
        elif cfg.train_phase == "phase4":
            self._train_phase4()
        elif cfg.train_phase == "bc":
            self._train_phase_bc()
        else:
            raise ValueError(f"Unknown phase: {cfg.train_phase}")

    # ========================================================================
    #  Phase 2: Offline world model pretraining
    # ========================================================================

    def _train_phase2(self):
        """Train world model on offline exiD data. No environment interaction."""
        cfg = self.config
        from training.offline_buffer import OfflineDataset

        use_speed = getattr(cfg, 'use_speed_head', False)
        use_traj = getattr(cfg, 'use_traj_head', False)
        use_curvature = getattr(cfg, 'use_curvature_head', False)
        use_vehicle = getattr(cfg, 'use_vehicle_head', False)
        need_positions = use_traj or use_curvature
        need_traj_horizon = max(getattr(cfg, 'traj_horizon', 125), 10)  # curvature needs >=10
        dataset = OfflineDataset(
            cfg.data_dir, bev_size=cfg.bev_size,
            seq_len=cfg.batch_length, device=str(self.device),
            cache_size=cfg.offline_cache_size,
            preload=cfg.preload_data,
            skip_resize=(getattr(cfg, 'bev_downsample', 'bilinear') in ('cnn', 'unshuffle')),
            return_phase_labels=(getattr(cfg, 'use_phase_head', False) or
                                  getattr(cfg, 'rssm_phase_conditional', False)),
            merge_zone_frames=getattr(cfg, 'merge_zone_frames', 20),
            return_speed_labels=use_speed,
            return_positions=need_positions,
            traj_horizon=need_traj_horizon,
            return_surrounding_labels=use_vehicle,
            n_surrounding=getattr(cfg, 'n_surrounding_vehicles', 5),
            merge_endpoints_path=getattr(cfg, 'merge_endpoints_path', ''),
            merge_zone_radius=getattr(cfg, 'merge_zone_radius', 25.0),
        )

        print(f"\n[Phase 2] Offline WM training | {cfg.total_wm_steps} steps")
        print(f"[Phase 2] Data: {len(dataset._file_list)} files, "
              f"{len(dataset)} sequences")
        print(f"[Phase 2] Reg: {cfg.regularization} "
              f"(sigreg_lambda={cfg.sigreg_lambda}, kl_beta={cfg.kl_beta})")
        if use_speed:
            print(f"[Phase 2] Aux heads: speed (λ={cfg.speed_head_weight})")
        if getattr(cfg, 'use_spatial_head', False):
            print(f"[Phase 2] Aux heads: spatial={cfg.spatial_head_resolution}x "
                  f"(λ={cfg.spatial_head_weight})")

        best_loss = float("inf")
        t_start = time.time()

        while self.global_step < cfg.total_wm_steps:
            # Sample batch from offline data
            sample_result = dataset.sample(cfg.batch_size)

            # Unpack: obs, obs_next, actions, rewards, continues [, phase_labels] [, speed_labels]
            idx = 0
            obs_r = sample_result[idx]; idx += 1
            _ = sample_result[idx]; idx += 1  # obs_next (unused in Phase 2)
            actions_r = sample_result[idx]; idx += 1
            rewards_r = sample_result[idx]; idx += 1
            continues_r = sample_result[idx]; idx += 1
            phase_labels = None
            speed_labels = None
            position_labels = None
            surrounding_labels = None
            if dataset.return_phase_labels:
                phase_labels = sample_result[idx]; idx += 1
                phase_labels = phase_labels.permute(1, 0).to(self.device)
            if dataset.return_speed_labels:
                speed_labels = sample_result[idx]; idx += 1
                speed_labels = speed_labels.permute(1, 0).to(self.device)
            if dataset.return_positions:
                position_labels = sample_result[idx]; idx += 1
                position_labels = position_labels.permute(1, 0, 2).to(self.device)
            if dataset.return_surrounding_labels:
                surrounding_labels = sample_result[idx]; idx += 1
                surrounding_labels = surrounding_labels.permute(1, 0, 2, 3).to(self.device)

            # Convert (B, T, ...) -> (T, B, ...)
            obs = obs_r.permute(1, 0, 2, 3, 4).to(self.device)
            actions = actions_r.permute(1, 0, 2).to(self.device)
            rewards = rewards_r.permute(1, 0).to(self.device)
            continues = continues_r.permute(1, 0).to(self.device)

            # Single training step
            self.wm_optimizer.zero_grad()
            prev_state = self.world_model.get_initial_state(cfg.batch_size, self.device)
            loss, metrics = self.world_model.compute_world_loss(
                obs, actions, rewards, continues, prev_state,
                phase_labels=phase_labels,
                speed_labels=speed_labels,
                position_labels=position_labels,
                surrounding_labels=surrounding_labels,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
            self.wm_optimizer.step()

            self.global_step += 1

            # Logging
            if self.global_step % cfg.log_every == 0:
                elapsed = time.time() - t_start
                fps = self.global_step / max(elapsed, 1)
                reg_key = f"loss_{cfg.regularization}"
                reg_val = metrics.get(reg_key, metrics.get("loss_dyn", 0))
                if cfg.use_decoder:
                    rec_label, rec_val_print = "rec", metrics['loss_rec']
                else:
                    rec_label, rec_val_print = "barlow", metrics.get('loss_barlow', metrics['loss_rec'])
                phase_str = ""
                if cfg.use_phase_head:
                    phase_str = (f" phase={metrics.get('loss_phase', 0):.4f} "
                                 f"acc={metrics.get('phase_accuracy', 0):.2f}")
                aux_str = ""
                if use_speed and 'loss_speed' in metrics:
                    aux_str += f" speed={metrics['loss_speed']:.4f}"
                if getattr(cfg, 'use_spatial_head', False) and 'loss_spatial' in metrics:
                    aux_str += f" spatial={metrics['loss_spatial']:.4f}"
                if getattr(cfg, 'use_jepa', False) and 'loss_jepa' in metrics:
                    aux_str += f" jepa={metrics['loss_jepa']:.4f}"
                if getattr(cfg, 'use_traj_head', False) and 'loss_traj' in metrics:
                    aux_str += f" traj={metrics['loss_traj']:.4f}"
                if getattr(cfg, 'use_curvature_head', False) and 'loss_curvature' in metrics:
                    aux_str += f" curv={metrics['loss_curvature']:.4f}"
                if getattr(cfg, 'use_vehicle_head', False) and 'loss_vehicle' in metrics:
                    aux_str += f" veh={metrics['loss_vehicle']:.4f}"
                if 'loss_moe_lb' in metrics:
                    aux_str += f" moe_lb={metrics['loss_moe_lb']:.4f}"
                if 'loss_phase_cond' in metrics:
                    aux_str += f" ph_cond={metrics['loss_phase_cond']:.4f}"
                print(f"[P2] step={self.global_step}/{cfg.total_wm_steps} "
                      f"| {rec_label}={rec_val_print:.4f} "
                      f"dyn={metrics['loss_dyn']:.4f} "
                      f"{cfg.regularization}={reg_val:.4f} "
                      f"rew={metrics['loss_rew']:.4f} "
                      f"con={metrics['loss_con']:.4f}"
                      f"{phase_str}{aux_str}"
                      f"| fps={fps:.1f}")

                self._write_log({
                    "phase": "phase2",
                    "global_step": self.global_step,
                    **{f"wm/{k}": v for k, v in metrics.items()},
                })
                if self.writer is not None:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"wm/{k}", v, self.global_step)
                    self.writer.add_scalar("wm/fps", fps, self.global_step)
                    # Report weighted SIGReg contribution
                    if cfg.regularization == "sigreg":
                        wm_lambda = cfg.sigreg_lambda
                        raw_reg = metrics.get("sigreg_loss", 0)
                        self.writer.add_scalar("wm/sigreg_weighted", wm_lambda * raw_reg, self.global_step)
                if self.wandb_run is not None:
                    wandb_log = {f"wm/{k}": v for k, v in metrics.items()}
                    wandb_log["wm/fps"] = fps
                    if cfg.regularization == "sigreg":
                        wandb_log["wm/sigreg_weighted"] = cfg.sigreg_lambda * metrics.get("sigreg_loss", 0)
                    self.wandb_run.log(wandb_log, step=self.global_step)

            # Save
            if self.global_step % cfg.save_every == 0:
                self._save(f"step{self.global_step}")
                if metrics["loss_total"] < best_loss:
                    best_loss = metrics["loss_total"]
                    self._save("best")

        self._save("final")
        elapsed = time.time() - t_start
        print(f"[Phase 2] Done in {elapsed:.0f}s. Best loss: {best_loss:.6f}")

    # ========================================================================
    #  Phase 2 E2E: Joint WM+AC training (end-to-end, skip Phase 3)
    # ========================================================================

    def _train_phase2_e2e(self):
        """Train world model AND actor-critic jointly on offline data.

        Each step:
          1. WM: encode observation sequence → RSSM observe → compute world loss
          2. AC: imagination rollout from first-frame state → actor-critic update
        """
        cfg = self.config
        from training.offline_buffer import OfflineDataset

        use_speed_e2e = getattr(cfg, 'use_speed_head', False)
        use_traj_e2e = getattr(cfg, 'use_traj_head', False)
        use_curvature_e2e = getattr(cfg, 'use_curvature_head', False)
        use_vehicle_e2e = getattr(cfg, 'use_vehicle_head', False)
        need_positions_e2e = use_traj_e2e or use_curvature_e2e
        need_traj_horizon_e2e = max(getattr(cfg, 'traj_horizon', 125), 10)
        dataset = OfflineDataset(
            cfg.data_dir, bev_size=cfg.bev_size,
            seq_len=cfg.batch_length, device=str(self.device),
            cache_size=cfg.offline_cache_size,
            preload=cfg.preload_data,
            skip_resize=(getattr(cfg, 'bev_downsample', 'bilinear') in ('cnn', 'unshuffle')),
            return_phase_labels=(getattr(cfg, 'use_phase_head', False) or
                                  getattr(cfg, 'rssm_phase_conditional', False)),
            merge_zone_frames=getattr(cfg, 'merge_zone_frames', 20),
            return_speed_labels=use_speed_e2e,
            return_positions=need_positions_e2e,
            traj_horizon=need_traj_horizon_e2e,
            return_surrounding_labels=use_vehicle_e2e,
            n_surrounding=getattr(cfg, 'n_surrounding_vehicles', 5),
            merge_endpoints_path=getattr(cfg, 'merge_endpoints_path', ''),
            merge_zone_radius=getattr(cfg, 'merge_zone_radius', 25.0),
        )

        print(f"\n[Phase 2 E2E] Joint WM+AC training | {cfg.total_wm_steps} steps")
        print(f"[Phase 2 E2E] Data: {len(dataset._file_list)} files, "
              f"{len(dataset)} sequences")
        print(f"[Phase 2 E2E] Reg: {cfg.regularization} "
              f"(sigreg_lambda={cfg.sigreg_lambda}, kl_beta={cfg.kl_beta})")
        print(f"[Phase 2 E2E] AC trained every {cfg.joint_ac_every} WM step(s) "
              f"| imagination_horizon={cfg.imagination_horizon}")

        best_loss = float("inf")
        t_start = time.time()

        while self.global_step < cfg.total_wm_steps:
            # ── 1. Sample batch ─────────────────────────────────────────
            sample_result = dataset.sample(cfg.batch_size)
            # Unpack (same logic as Phase 2)
            ridx = 0
            obs = sample_result[ridx]; ridx += 1
            _obs_next = sample_result[ridx]; ridx += 1
            actions = sample_result[ridx]; ridx += 1
            rewards = sample_result[ridx]; ridx += 1
            continues = sample_result[ridx]; ridx += 1
            phase_labels = None
            speed_labels = None
            if dataset.return_phase_labels:
                phase_labels = sample_result[ridx]; ridx += 1
                phase_labels = phase_labels.permute(1, 0).to(self.device)
            if dataset.return_speed_labels:
                speed_labels = sample_result[ridx]; ridx += 1
                speed_labels = speed_labels.permute(1, 0).to(self.device)
            position_labels_e2e = None
            surrounding_labels_e2e = None
            if dataset.return_positions:
                position_labels_e2e = sample_result[ridx]; ridx += 1
                position_labels_e2e = position_labels_e2e.permute(1, 0, 2).to(self.device)
            if dataset.return_surrounding_labels:
                surrounding_labels_e2e = sample_result[ridx]; ridx += 1
                surrounding_labels_e2e = surrounding_labels_e2e.permute(1, 0, 2, 3).to(self.device)

            # ── 2. World model training ──────────────────────────────────
            obs_wm = obs.permute(1, 0, 2, 3, 4).to(self.device)
            actions_wm = actions.permute(1, 0, 2).to(self.device)
            rewards_wm = rewards.permute(1, 0).to(self.device)
            continues_wm = continues.permute(1, 0).to(self.device)

            self.wm_optimizer.zero_grad()
            prev_state = self.world_model.get_initial_state(cfg.batch_size, self.device)
            loss, wm_metrics = self.world_model.compute_world_loss(
                obs_wm, actions_wm, rewards_wm, continues_wm, prev_state,
                phase_labels=phase_labels,
                speed_labels=speed_labels,
                position_labels=position_labels_e2e,
                surrounding_labels=surrounding_labels_e2e,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
            self.wm_optimizer.step()

            # ── 3. AC imagination training ───────────────────────────────
            ac_metrics = {}
            if self.global_step % cfg.joint_ac_every == 0:
                # Use first frame of each sequence as starting observation
                start_obs = obs[:, 0].to(self.device)  # (B, C, H, W)

                with torch.no_grad():
                    embed = self.world_model.encode(start_obs)
                    start_state_raw = self.world_model.get_initial_state(
                        cfg.batch_size, self.device)
                    dummy = torch.zeros(cfg.batch_size, cfg.action_dim, device=self.device)
                    states_obs, _ = self.world_model.rssm.observe(
                        embed.unsqueeze(0), dummy.unsqueeze(0), start_state_raw
                    )
                    start_state = states_obs[-1]

                ac_metrics = self._train_ac_imagination(start_state)

            self.global_step += 1

            # ── 4. Logging ──────────────────────────────────────────────
            if self.global_step % cfg.log_every == 0:
                elapsed = time.time() - t_start
                fps = self.global_step / max(elapsed, 1)
                reg_key = f"loss_{cfg.regularization}"
                reg_val = wm_metrics.get(reg_key, wm_metrics.get("loss_dyn", 0))
                if cfg.use_decoder:
                    rec_label, rec_val_print = "rec", wm_metrics['loss_rec']
                else:
                    rec_label, rec_val_print = "barlow", wm_metrics.get('loss_barlow', wm_metrics['loss_rec'])

                ac_str = ""
                if ac_metrics:
                    ac_str = (f"| actor={ac_metrics['actor_loss']:.4f} "
                              f"critic={ac_metrics['critic_loss']:.4f} "
                              f"rew_imag={ac_metrics['imag_reward_mean']:.4f}")

                phase_e2e = ""
                if cfg.use_phase_head:
                    phase_e2e = (f" phase={wm_metrics.get('loss_phase', 0):.4f} "
                                 f"acc={wm_metrics.get('phase_accuracy', 0):.2f}")
                print(f"[P2-E2E] step={self.global_step}/{cfg.total_wm_steps} "
                      f"| {rec_label}={rec_val_print:.4f} "
                      f"dyn={wm_metrics['loss_dyn']:.4f} "
                      f"{cfg.regularization}={reg_val:.4f} "
                      f"rew={wm_metrics['loss_rew']:.4f} "
                      f"con={wm_metrics['loss_con']:.4f}"
                      f"{phase_e2e}"
                      f"{ac_str}"
                      f"| fps={fps:.1f}")

                log_entry = {
                    "phase": "phase2_e2e",
                    "global_step": self.global_step,
                    **{f"wm/{k}": v for k, v in wm_metrics.items()},
                }
                if ac_metrics:
                    log_entry.update({f"ac/{k}": v for k, v in ac_metrics.items()})
                self._write_log(log_entry)

                if self.writer is not None:
                    for k, v in wm_metrics.items():
                        self.writer.add_scalar(f"wm/{k}", v, self.global_step)
                    self.writer.add_scalar("wm/fps", fps, self.global_step)
                    if ac_metrics:
                        for k, v in ac_metrics.items():
                            self.writer.add_scalar(f"ac/{k}", v, self.global_step)

                if self.wandb_run is not None:
                    wandb_log = {f"wm/{k}": v for k, v in wm_metrics.items()}
                    wandb_log["wm/fps"] = fps
                    if ac_metrics:
                        wandb_log.update({f"ac/{k}": v for k, v in ac_metrics.items()})
                    self.wandb_run.log(wandb_log, step=self.global_step)

            # ── 5. Save ─────────────────────────────────────────────────
            if self.global_step % cfg.save_every == 0:
                self._save(f"step{self.global_step}")
                if wm_metrics["loss_total"] < best_loss:
                    best_loss = wm_metrics["loss_total"]
                    self._save("best")

        self._save("final")
        elapsed = time.time() - t_start
        print(f"[Phase 2 E2E] Done in {elapsed:.0f}s. Best loss: {best_loss:.6f}")

    # ========================================================================
    #  Phase 3: Imagination policy training (frozen world model)
    # ========================================================================

    def _train_phase3(self):
        """Train actor-critic entirely in world model imagination."""
        cfg = self.config
        from training.offline_buffer import OfflineDataset

        dataset = OfflineDataset(
            cfg.data_dir, bev_size=cfg.bev_size,
            seq_len=1, device=str(self.device),
            preload=cfg.preload_data,
            skip_resize=(getattr(cfg, 'bev_downsample', 'bilinear') in ('cnn', 'unshuffle')),
        )

        # Freeze world model
        for p in self.world_model.parameters():
            p.requires_grad = False
        self.world_model.eval()

        self.global_step = 0  # Reset counter for Phase 3

        print(f"\n[Phase 3] Imagination AC training | {cfg.total_imagine_steps} steps")
        print(f"[Phase 3] Imagination horizon: {cfg.imagination_horizon}")

        t_start = time.time()

        while self.global_step < cfg.total_imagine_steps:
            # Sample starting observations + GT actions from offline data
            start_obs, start_action = dataset.sample_start_obs(cfg.batch_size)
            start_obs = start_obs.to(self.device)
            start_action = start_action.to(self.device)

            # Get starting RSSM state from real observation + GT action
            with torch.no_grad():
                embed = self.world_model.encode(start_obs)
                prev_state = self.world_model.get_initial_state(cfg.batch_size, self.device)
                states, _ = self.world_model.rssm.observe(
                    embed.unsqueeze(0), start_action.unsqueeze(0), prev_state
                )
                start_state = states[-1]

            # Train actor-critic from this starting state
            ac_metrics = self._train_ac_imagination(start_state)

            self.global_step += 1

            # Logging
            if self.global_step % cfg.log_every == 0:
                elapsed = time.time() - t_start
                fps = self.global_step / max(elapsed, 1)
                print(f"[P3] step={self.global_step}/{cfg.total_imagine_steps} "
                      f"| actor={ac_metrics['actor_loss']:.4f} "
                      f"critic={ac_metrics['critic_loss']:.4f} "
                      f"rew_imag={ac_metrics['imag_reward_mean']:.4f} "
                      f"| fps={fps:.1f}")

                self._write_log({
                    "phase": "phase3",
                    "global_step": self.global_step,
                    **{f"ac/{k}": v for k, v in ac_metrics.items()},
                })
                if self.writer is not None:
                    for k, v in ac_metrics.items():
                        self.writer.add_scalar(f"ac/{k}", v, self.global_step)
                    self.writer.add_scalar("ac/fps", fps, self.global_step)
                if self.wandb_run is not None:
                    wandb_log = {f"ac/{k}": v for k, v in ac_metrics.items()}
                    wandb_log["ac/fps"] = fps
                    self.wandb_run.log(wandb_log, step=self.global_step)

            # Save
            if self.global_step % cfg.save_every == 0:
                self._save(f"step{self.global_step}")

        self._save("final")
        elapsed = time.time() - t_start
        print(f"[Phase 3] Done in {elapsed:.0f}s.")

    def _train_ac_imagination(self, start_state):
        """One step of actor-critic training in imagination.

        Args:
            start_state: RSSM state dict from a real observation.
        Returns:
            dict of metrics.
        """
        cfg = self.config

        # Imagine rollout
        imag_rewards = []
        imag_continues = []
        imag_log_probs = []

        state = start_state
        for t in range(cfg.imagination_horizon):
            feature = self.world_model.get_feature(state)

            # Actor selects action
            action, log_prob = self.actor(feature.detach())

            # Predict reward and continue (world model provides these)
            with torch.no_grad():
                if cfg.use_phase_head and self.world_model.phase_head is not None:
                    phase_logits = self.world_model.phase_head(feature)
                    phase_probs = F.softmax(phase_logits, dim=-1)
                    phase_values = torch.tensor([-0.3, 0.5, 1.0],
                                                device=feature.device, dtype=feature.dtype)
                    base_reward = (phase_probs * phase_values).sum(-1)
                    # Action-dependent phase bonus (scale=0.3)
                    throttle = action[:, 1]
                    steer = action[:, 0]
                    p0_bonus = 0.3 * throttle.clamp(min=0) - 0.1 * steer.abs()
                    p1_bonus = 0.2 * steer.abs() - 0.1 * throttle.clamp(max=0).abs()
                    p2_bonus = 0.1 * throttle.clamp(min=0) - 0.1 * (steer ** 2)
                    action_bonus = 0.3 * (phase_probs[:, 0] * p0_bonus +
                                          phase_probs[:, 1] * p1_bonus +
                                          phase_probs[:, 2] * p2_bonus)
                    reward = base_reward + action_bonus
                    # Crash penalty: low continue + not yet post-merge
                    cont_logit = self.world_model.continue_head(feature).squeeze()
                    cont = torch.sigmoid(cont_logit)
                    # Crash penalty: online detector when ready, else continue_head heuristic
                    if self.crash_ready:
                        crash_penalty = self.crash_detector(feature)
                    else:
                        crash_penalty = F.relu(0.3 - cont)
                    reward = reward - 10.0 * crash_penalty
                else:
                    reward = self.world_model.reward_head(feature).squeeze()
                    cont_logit = self.world_model.continue_head(feature).squeeze()
                    cont = torch.sigmoid(cont_logit)

            # RSSM prior step (no observation)
            with torch.no_grad():
                states_imag = self.world_model.rssm.imagine(
                    action.unsqueeze(0), state
                )
                state = states_imag[0]

            imag_rewards.append(reward)
            imag_continues.append(cont)
            imag_log_probs.append(log_prob)

        # Stack: (H, B, ...) or (H, B)
        imag_rewards = torch.stack(imag_rewards)
        imag_continues = torch.stack(imag_continues)
        imag_log_probs = torch.stack(imag_log_probs)

        # Features for value estimation: re-run imagination to get features
        # (we need features detached from actor for critic training)
        with torch.no_grad():
            imag_features = []
            state = start_state
            for t in range(cfg.imagination_horizon):
                feature = self.world_model.get_feature(state)
                action, _ = self.actor(feature)  # sample action
                imag_features.append(feature)
                states_imag = self.world_model.rssm.imagine(
                    action.unsqueeze(0), state
                )
                state = states_imag[0]
            imag_features = torch.stack(imag_features)

        # Lambda returns via slow critic
        with torch.no_grad():
            slow_values, _ = self.slow_critic(
                imag_features.reshape(-1, imag_features.shape[-1])
            )
            slow_values = slow_values.reshape(imag_features.shape[:2])

        returns = self._compute_lambda_returns(
            imag_rewards, imag_continues, slow_values,
            gamma=cfg.gamma, lam=cfg.lambda_gae,
        )

        # --- Critic loss ---
        values, value_logits = self.critic(
            imag_features.detach().reshape(-1, imag_features.shape[-1])
        )
        values = values.reshape(imag_features.shape[:2])
        value_logits = value_logits.reshape(-1, 255)

        value_targets = self.critic.compute_target(
            returns.detach().reshape(-1)
        )

        critic_loss = F.cross_entropy(value_logits, value_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()

        # --- Actor loss ---
        advantages = (returns - values.detach()).reshape(imag_log_probs.shape)
        actor_loss = -(imag_log_probs * advantages).mean()
        actor_loss -= cfg.entropy_weight * (-imag_log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()

        # Update slow critic
        self._update_slow_critic()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "imag_reward_mean": imag_rewards.mean().item(),
            "value_mean": values.mean().item(),
        }

    # ========================================================================
    #  Online Crash Detector: binary classifier on RSSM features
    # ========================================================================

    def train_crash_detector(self, batch_size=64):
        """Train crash detector on collected (feature, label) pairs.

        Called after episodes where crash/out_of_road samples were added.
        Returns loss dict or None if buffer not ready.
        """
        if not self.crash_buffer.ready():
            return None

        features, labels = self.crash_buffer.sample_batch(batch_size, self.device)
        preds = self.crash_detector(features)
        loss = F.binary_cross_entropy(preds, labels)

        self.crash_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.crash_detector.parameters(), 10.0)
        self.crash_optimizer.step()

        self.crash_ready = True
        with torch.no_grad():
            acc = ((preds > 0.5).float() == labels).float().mean()
        return {"crash_det_loss": float(loss), "crash_det_acc": float(acc)}

    # ========================================================================
    #  DAPO: Decoupled Alignment from Policy Optimization
    #  Group-based advantage, no critic, no value inflation.
    # ========================================================================

    def _train_ac_dapo(self, start_state, K=8):
        """One step of DAPO policy gradient in imagination.

        Samples K action sequences per start state, rolls out in WM,
        computes group-relative advantage, and updates the actor via PG.

        Collision proxy: penalizes low continue probability during
        imagination — low continue at non-terminal steps suggests the
        WM predicts abnormal trajectory termination (≈ collision).

        Args:
            start_state: RSSM state dict from a real observation.
            K: number of action sequences per start state (group size).
        Returns:
            dict of metrics.
        """
        cfg = self.config
        B = cfg.batch_size
        H = cfg.imagination_horizon

        # Repeat start state K times -> (B*K, ...)
        deter = start_state["deter"].repeat_interleave(K, dim=0)
        stoch = start_state["stoch"].repeat_interleave(K, dim=0)
        state = {"deter": deter, "stoch": stoch}

        rewards_list = []
        log_probs_list = []
        collision_risk_list = []

        for t in range(H):
            feature = self.world_model.get_feature(state)

            # Actor samples action (differentiable)
            action, log_prob = self.actor(feature.detach())

            with torch.no_grad():
                if cfg.use_phase_head and self.world_model.phase_head is not None:
                    phase_logits = self.world_model.phase_head(feature)
                    phase_probs = F.softmax(phase_logits, dim=-1)
                    phase_values = torch.tensor([-0.3, 0.5, 1.0],
                                                device=feature.device, dtype=feature.dtype)
                    base_reward = (phase_probs * phase_values).sum(-1)
                    # Action-dependent phase bonus (scale=0.3)
                    throttle = action[:, 1]
                    steer = action[:, 0]
                    p0_bonus = 0.3 * throttle.clamp(min=0) - 0.1 * steer.abs()
                    p1_bonus = 0.2 * steer.abs() - 0.1 * throttle.clamp(max=0).abs()
                    p2_bonus = 0.1 * throttle.clamp(min=0) - 0.1 * (steer ** 2)
                    action_bonus = 0.3 * (phase_probs[:, 0] * p0_bonus +
                                          phase_probs[:, 1] * p1_bonus +
                                          phase_probs[:, 2] * p2_bonus)
                    reward = base_reward + action_bonus
                    cont_logit = self.world_model.continue_head(feature).squeeze()
                    cont = torch.sigmoid(cont_logit)
                    post_merge_prob = phase_probs[:, 2]
                    # Crash penalty: online detector when ready, else continue_head heuristic
                    if self.crash_ready:
                        crash_penalty = self.crash_detector(feature)
                    else:
                        crash_penalty = F.relu(0.3 - cont)
                    reward = reward - 10.0 * crash_penalty
                else:
                    reward = self.world_model.reward_head(feature).squeeze()
                    cont_logit = self.world_model.continue_head(feature).squeeze()
                    cont = torch.sigmoid(cont_logit)
                # Collision proxy: low continue ≈ unsafe termination
                collision_risk = F.relu(0.3 - cont)  # penalty when cont < 0.3
                states_imag = self.world_model.rssm.imagine(
                    action.unsqueeze(0), state
                )
                state = states_imag[0]

            # Reward = WM reward - collision proxy penalty
            combined_reward = reward - cfg.dapo_collision_weight * collision_risk
            rewards_list.append(combined_reward)
            log_probs_list.append(log_prob)
            collision_risk_list.append(collision_risk)

        # Stack: (H, B*K)
        rewards = torch.stack(rewards_list)
        log_probs = torch.stack(log_probs_list)
        collision_risks = torch.stack(collision_risk_list)

        # Discounted return per trajectory
        gamma_pow = torch.pow(
            cfg.gamma,
            torch.arange(H, device=self.device, dtype=torch.float32),
        )
        total_rewards = (rewards * gamma_pow.unsqueeze(1)).sum(0)   # (B*K,)
        total_log_probs = log_probs.sum(0)                          # (B*K,)

        # Reshape to (K, B) for group advantage
        total_rewards = total_rewards.reshape(K, B)
        total_log_probs = total_log_probs.reshape(K, B)

        # Group-relative advantage
        mean_r = total_rewards.mean(0, keepdim=True)
        std_r = total_rewards.std(0, keepdim=True).clamp(min=1e-6)

        # ── DAPO: Dynamic Sampling ──
        valid_mask = std_r.squeeze(0) > 1e-4
        if valid_mask.sum() == 0:
            return {
                "actor_loss": 0.0,
                "imag_reward_mean": rewards.mean().item(),
                "collision_risk_mean": collision_risks.mean().item(),
                "return_group_std": 0.0,
                "return_group_max": total_rewards.max(0).values.mean().item(),
                "return_group_min": total_rewards.min(0).values.mean().item(),
                "dynamic_skip": True,
            }
        if valid_mask.sum() < B:
            total_rewards = total_rewards[:, valid_mask]
            total_log_probs = total_log_probs[:, valid_mask]
            mean_r = total_rewards.mean(0, keepdim=True)
            std_r = total_rewards.std(0, keepdim=True).clamp(min=1e-6)

        # ── DAPO: Clip-Higher (asymmetric) ──
        advantage = (total_rewards - mean_r) / std_r
        advantage_high = advantage.clamp(max=3.0)
        advantage_low = advantage.clamp(min=-1.0)
        advantage = torch.where(advantage > 0, advantage_high, advantage_low)

        # Policy gradient
        actor_loss = -(total_log_probs * advantage.detach()).mean()
        actor_loss -= cfg.entropy_weight * (-total_log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "imag_reward_mean": rewards.mean().item(),
            "collision_risk_mean": collision_risks.mean().item(),
            "return_group_std": std_r.mean().item(),
            "return_group_max": total_rewards.max(0).values.mean().item(),
            "return_group_min": total_rewards.min(0).values.mean().item(),
        }

    def _train_phase3_dapo(self, K=8):
        """Phase 3 with DAPO: group-based policy gradient, no critic."""
        cfg = self.config
        from training.offline_buffer import OfflineDataset

        dataset = OfflineDataset(
            cfg.data_dir, bev_size=cfg.bev_size,
            seq_len=1, device=str(self.device),
            preload=cfg.preload_data,
            skip_resize=(getattr(cfg, 'bev_downsample', 'bilinear') in ('cnn', 'unshuffle')),
        )

        # Freeze world model
        for p in self.world_model.parameters():
            p.requires_grad = False
        self.world_model.eval()

        self.global_step = 0

        print(f"\n[Phase 3 DAPO] Group-based PG | "
              f"{cfg.total_imagine_steps} steps | K={K} | H={cfg.imagination_horizon}")
        print(f"[Phase 3 DAPO] No critic — no value inflation")

        t_start = time.time()

        while self.global_step < cfg.total_imagine_steps:
            start_obs, start_action = dataset.sample_start_obs(cfg.batch_size)
            start_obs = start_obs.to(self.device)
            start_action = start_action.to(self.device)

            with torch.no_grad():
                embed = self.world_model.encode(start_obs)
                prev_state = self.world_model.get_initial_state(
                    cfg.batch_size, self.device
                )
                states, _ = self.world_model.rssm.observe(
                    embed.unsqueeze(0), start_action.unsqueeze(0), prev_state
                )
                start_state = states[-1]

            ac_metrics = self._train_ac_dapo(start_state, K=K)
            self.global_step += 1

            if self.global_step % cfg.log_every == 0:
                elapsed = time.time() - t_start
                fps = self.global_step / max(elapsed, 1)
                print(f"[P3-DAPO] step={self.global_step}/{cfg.total_imagine_steps} "
                      f"| actor={ac_metrics['actor_loss']:.4f} "
                      f"rew_imag={ac_metrics['imag_reward_mean']:.4f} "
                      f"ret_max={ac_metrics['return_group_max']:.3f} "
                      f"| fps={fps:.1f}")

                self._write_log({
                    "phase": "phase3_dapo",
                    "global_step": self.global_step,
                    **{f"ac/{k}": v for k, v in ac_metrics.items()},
                })
                if self.writer is not None:
                    for k, v in ac_metrics.items():
                        self.writer.add_scalar(f"ac/{k}", v, self.global_step)
                    self.writer.add_scalar("ac/fps", fps, self.global_step)
                if self.wandb_run is not None:
                    wandb_log = {f"ac/{k}": v for k, v in ac_metrics.items()}
                    wandb_log["ac/fps"] = fps
                    self.wandb_run.log(wandb_log, step=self.global_step)

            if self.global_step % cfg.save_every == 0:
                self._save(f"step{self.global_step}")

        self._save("final")
        elapsed = time.time() - t_start
        print(f"[Phase 3 DAPO] Done in {elapsed:.0f}s.")

    # ========================================================================
    #  BC Pretraining: Behavior Cloning in frozen WM feature space
    # ========================================================================

    def _train_phase_bc(self):
        """Behavior Cloning in frozen WM feature space.

        Two modes:
          - Standard (bc_use_moe=False): feature → MLP → action (no phase info)
          - MoE (bc_use_moe=True): shared MLP → 3 expert heads + phase classifier
            Hard routing by GT phase (train), soft routing by classifier (infer)
        """
        cfg = self.config
        from training.offline_buffer import OfflineDataset

        use_moe = getattr(cfg, 'bc_use_moe', False)

        seq_len = cfg.batch_length
        dataset = OfflineDataset(
            cfg.data_dir, bev_size=cfg.bev_size,
            seq_len=seq_len, device=str(self.device),
            cache_size=cfg.offline_cache_size,
            preload=cfg.preload_data,
            skip_resize=(getattr(cfg, 'bev_downsample', 'bilinear') in ('cnn', 'unshuffle')),
            return_phase_labels=use_moe,  # Only need phase for MoE
            merge_zone_frames=getattr(cfg, 'merge_zone_frames', 20),
            merge_endpoints_path=getattr(cfg, 'merge_endpoints_path', ''),
            merge_zone_radius=getattr(cfg, 'merge_zone_radius', 25.0),
        )
        # Freeze world model
        for p in self.world_model.parameters():
            p.requires_grad = False
        self.world_model.eval()

        feat_dim = self.world_model.feature_dim()
        self.actor = Actor(
            feat_dim, cfg.action_dim,
            cfg.actor_hidden, cfg.actor_layers,
            moe_phase=use_moe,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr, eps=1e-5
        )

        if use_moe:
            cls_weight = getattr(cfg, 'moe_classifier_weight', 0.3)

        self.global_step = 0
        frames_per_step = cfg.batch_size * seq_len

        mode_str = "MoE-Phase" if use_moe else "Standard"
        print(f"\n[BC] {mode_str} | {cfg.total_imagine_steps} steps × {frames_per_step} frames/step")
        print(f"[BC] WM frozen, seq_len={seq_len}, actor lr={cfg.actor_lr}")
        if use_moe:
            print(f"[BC] MoE: 3 experts + classifier, cls_weight={cls_weight}")
        else:
            print(f"[BC] Standard: feature({feat_dim}) → MLP → action({cfg.action_dim})")

        t_start = time.time()
        best_loss = float("inf")

        while self.global_step < cfg.total_imagine_steps:
            if use_moe:
                obs, _, actions, _, _, phase_labels = dataset.sample(cfg.batch_size)
            else:
                obs, _, actions, _, _ = dataset.sample(cfg.batch_size)

            B, L = obs.shape[:2]
            with torch.no_grad():
                obs_flat = obs.reshape(-1, *obs.shape[2:]).to(self.device)
                embeds = self.world_model.encode(obs_flat)
                embeds = embeds.reshape(B, L, -1).permute(1, 0, 2)

                actions_seq = actions.permute(1, 0, 2).to(self.device)
                prev_state = self.world_model.get_initial_state(B, self.device)
                post_states, _ = self.world_model.rssm.observe(
                    embeds, actions_seq, prev_state
                )

                features = torch.stack(
                    [self.world_model.get_feature(s) for s in post_states]
                )  # (L, B, feat_dim)

            features_flat = features.reshape(-1, feat_dim)
            actions_flat = actions.reshape(-1, 2).to(self.device)

            if use_moe:
                phase_flat = phase_labels.reshape(-1).to(self.device).long()
                action_pred, _ = self.actor(
                    features_flat, phase_labels=phase_flat, deterministic=True
                )
                phase_logits = self.actor.last_phase_logits
                bc_loss = F.mse_loss(action_pred, actions_flat)
                cls_loss = F.cross_entropy(phase_logits, phase_flat)
                cls_acc = (phase_logits.argmax(-1) == phase_flat).float().mean()
                total_loss = bc_loss + cls_weight * cls_loss
                # Per-expert loss
                expert_losses = {}
                for i, name in enumerate(["pre", "merge", "post"]):
                    mask = (phase_flat == i)
                    if mask.sum() > 0:
                        expert_losses[name] = F.mse_loss(
                            action_pred[mask], actions_flat[mask]
                        ).item()
            else:
                action_pred, _ = self.actor(features_flat, deterministic=True)
                bc_loss = F.mse_loss(action_pred, actions_flat)
                total_loss = bc_loss

            self.actor_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
            self.actor_optimizer.step()

            self.global_step += 1

            # Logging
            if self.global_step % cfg.log_every == 0:
                elapsed = time.time() - t_start
                fps = frames_per_step * self.global_step / max(elapsed, 1)
                if use_moe:
                    exp_str = f" | pre={expert_losses.get('pre',0):.4f} merge={expert_losses.get('merge',0):.4f} post={expert_losses.get('post',0):.4f}"
                    extra = f" cls={cls_loss.item():.4f} acc={cls_acc.item():.2f}{exp_str}"
                else:
                    extra = ""
                print(f"[BC] step={self.global_step}/{cfg.total_imagine_steps} "
                      f"| loss={bc_loss.item():.4f}{extra}"
                      f"| {fps:.0f} f/s")

                log_entry = {"phase": "bc", "global_step": self.global_step,
                             "bc_loss": bc_loss.item(), "bc_mode": mode_str}
                if use_moe:
                    log_entry["phase_cls_loss"] = cls_loss.item()
                    log_entry["phase_cls_acc"] = cls_acc.item()
                    for k, v in expert_losses.items():
                        log_entry[f"bc_loss_{k}"] = v
                self._write_log(log_entry)
                if self.writer is not None:
                    self.writer.add_scalar("bc/loss", bc_loss.item(), self.global_step)
                    self.writer.add_scalar("bc/fps", fps, self.global_step)
                    if use_moe:
                        self.writer.add_scalar("bc/cls_acc", cls_acc.item(), self.global_step)
                        for k, v in expert_losses.items():
                            self.writer.add_scalar(f"bc/loss_{k}", v, self.global_step)
                if self.wandb_run is not None:
                    wl = {"bc/loss": bc_loss.item(), "bc/fps": fps}
                    if use_moe:
                        wl["bc/cls_acc"] = cls_acc.item()
                        for k, v in expert_losses.items():
                            wl[f"bc/loss_{k}"] = v
                    self.wandb_run.log(wl, step=self.global_step)

            # Save
            if self.global_step % cfg.save_every == 0:
                self._save(f"bc_step{self.global_step}")
                if bc_loss.item() < best_loss:
                    best_loss = bc_loss.item()
                    self._save("bc_best")

        self._save("bc_final")
        elapsed = time.time() - t_start
        total_frames = cfg.total_imagine_steps * frames_per_step
        print(f"[BC] Done in {elapsed:.0f}s. Best loss: {best_loss:.6f} "
              f"({total_frames/elapsed:.0f} f/s overall)")

    # ========================================================================
    #  Phase 4: Online finetuning in MetaDrive
    # ========================================================================

    def _train_phase4(self):
        """Online RL finetuning with MetaDrive environment."""
        cfg = self.config

        # Unfreeze world model (in case loaded from Phase 3 checkpoint)
        for p in self.world_model.parameters():
            p.requires_grad = True

        # Create environment
        env = self.env_factory() if self.env_factory else self._default_env()

        # Replay buffer for online data
        from training.replay_buffer import ReplayBuffer
        obs_shape = (cfg.input_channels, cfg.bev_size, cfg.bev_size)
        self.buffer = ReplayBuffer(cfg.replay_size, obs_shape, cfg.action_dim)

        # Warmup
        print(f"[Phase 4] Collecting warmup ({cfg.replay_warmup} steps)...")
        self._collect_episode(env, max_steps=cfg.replay_warmup)
        print(f"[Phase 4] Buffer: {len(self.buffer)}")

        print(f"[Phase 4] Online finetuning for {cfg.total_env_steps} steps...")
        episode = 0
        best_reward = -float("inf")
        wm_metrics = ac_metrics = {}

        while self.global_step < cfg.total_env_steps:
            ep_reward, ep_steps = self._collect_episode(env)
            episode += 1

            # Train multiple times per episode
            for _ in range(ep_steps // 2):
                if len(self.buffer) < cfg.batch_length + 1:
                    break
                wm_metrics = self._train_wm_online()
                ac_metrics = self._train_ac_online()
                self.global_step += 1
                if self.global_step >= cfg.total_env_steps:
                    break

            # Logging
            if episode % cfg.log_every == 0:
                reg_key = f"loss_{cfg.regularization}"
                reg_val = wm_metrics.get(reg_key, wm_metrics.get("loss_dyn", 0))
                if cfg.use_decoder:
                    rec_label, rec_val_print = "rec", wm_metrics.get('loss_rec', 0)
                else:
                    rec_label, rec_val_print = "barlow", wm_metrics.get('loss_barlow', wm_metrics.get('loss_rec', 0))
                print(f"[P4] step={self.global_step} ep={episode} "
                      f"| R={ep_reward:.1f} len={ep_steps} "
                      f"| {rec_label}={rec_val_print:.4f} "
                      f"dyn={reg_val:.4f} "
                      f"| actor={ac_metrics.get('actor_loss', 0):.4f} "
                      f"val={ac_metrics.get('value_mean', 0):.2f}")

                self._write_log({
                    "phase": "phase4",
                    "episode": episode,
                    "global_step": self.global_step,
                    "ep_reward": ep_reward,
                    "ep_steps": ep_steps,
                    **{f"wm/{k}": v for k, v in (wm_metrics or {}).items()},
                    **{f"ac/{k}": v for k, v in (ac_metrics or {}).items()},
                })
                if self.writer is not None:
                    self.writer.add_scalar("p4/ep_reward", ep_reward, self.global_step)
                    self.writer.add_scalar("p4/ep_steps", ep_steps, self.global_step)
                    for k, v in (wm_metrics or {}).items():
                        self.writer.add_scalar(f"wm/{k}", v, self.global_step)
                    for k, v in (ac_metrics or {}).items():
                        self.writer.add_scalar(f"ac/{k}", v, self.global_step)
                if self.wandb_run is not None:
                    wandb_log = {"p4/ep_reward": ep_reward, "p4/ep_steps": ep_steps}
                    for k, v in (wm_metrics or {}).items():
                        wandb_log[f"wm/{k}"] = v
                    for k, v in (ac_metrics or {}).items():
                        wandb_log[f"ac/{k}"] = v
                    self.wandb_run.log(wandb_log, step=self.global_step)

            # Save
            if episode % cfg.save_every == 0:
                self._save(f"ep{episode}")

            # Eval
            if episode % cfg.eval_every == 0:
                eval_reward = self._evaluate(env, cfg.eval_episodes)
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self._save("best")
                print(f"[P4] Eval ep={episode}: avgR={eval_reward:.2f} "
                      f"best={best_reward:.2f}")

        env.close()
        self._save("final")
        print(f"[Phase 4] Done. Best reward: {best_reward:.2f}")

    # ========================================================================
    #  Online data collection & training steps (Phase 4)
    # ========================================================================

    def _collect_episode(self, env, max_steps=None):
        """Collect one episode into replay buffer."""
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        horizon = max_steps or self.config.horizon

        state = None
        if self.global_step > 0:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                embed = self.world_model.encode(obs_t)
                state = self.world_model.get_initial_state(1, self.device)
                dummy_action = torch.zeros(1, self.config.action_dim, device=self.device)
                states, _ = self.world_model.rssm.observe(
                    embed.unsqueeze(0), dummy_action.unsqueeze(0), state
                )
                state = states[-1]

        for t in range(horizon):
            if state is None:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    feature = self.world_model.get_feature(state)
                    action = self.actor.act(feature).cpu().numpy()[0]

            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            self.buffer.add(next_obs, action, reward, done)

            if state is not None:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
                    embed = self.world_model.encode(obs_t)
                    action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    states, _ = self.world_model.rssm.observe(
                        embed.unsqueeze(0), action_t.unsqueeze(0), state
                    )
                    state = states[-1]

            obs = next_obs
            if done:
                break

        return total_reward, steps

    def _train_wm_online(self):
        """World model training step from replay buffer (Phase 4)."""
        cfg = self.config
        obs, actions, rewards, continues = self.buffer.sample(
            cfg.batch_size, cfg.batch_length
        )

        # Buffer stores uint8 [0,255]; normalize to [0,1] (encoder was trained on this range)
        obs = torch.FloatTensor(obs).to(self.device).permute(1, 0, 2, 3, 4) / 255.0
        actions = torch.FloatTensor(actions).to(self.device).permute(1, 0, 2)
        rewards = torch.FloatTensor(rewards).to(self.device).permute(1, 0)
        continues = torch.FloatTensor(continues).to(self.device).permute(1, 0)

        prev_state = self.world_model.get_initial_state(obs.shape[1], self.device)

        self.wm_optimizer.zero_grad()
        loss, metrics = self.world_model.compute_world_loss(
            obs, actions, rewards, continues, prev_state
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.wm_optimizer.step()

        return metrics

    def _train_wm_mixed(self, offline_dataset):
        """World model training step from mixed offline + online data.

        Samples half the batch from OfflineDataset (exiD npz, already [0,1])
        and half from ReplayBuffer (online, uint8 [0,255]).
        """
        cfg = self.config
        half = max(1, cfg.batch_size // 2)

        # Offline batch: torch tensors (B,L,...), already float [0,1]
        off_result = offline_dataset.sample(half)
        off_obs = off_result[0].permute(1, 0, 2, 3, 4).to(self.device)   # (T, B, C, H, W)
        off_actions = off_result[2].permute(1, 0, 2).to(self.device)       # (T, B, 2)
        off_rewards = off_result[3].permute(1, 0).to(self.device)          # (T, B)
        off_continues = off_result[4].permute(1, 0).to(self.device)        # (T, B)

        # Online batch: numpy uint8 [0,255] → normalize
        on_obs_np, on_actions_np, on_rewards_np, on_continues_np = self.buffer.sample(
            half, cfg.batch_length
        )
        on_obs = torch.FloatTensor(on_obs_np).to(self.device).permute(1, 0, 2, 3, 4) / 255.0
        on_actions = torch.FloatTensor(on_actions_np).to(self.device).permute(1, 0, 2)
        on_rewards = torch.FloatTensor(on_rewards_np).to(self.device).permute(1, 0)
        on_continues = torch.FloatTensor(on_continues_np).to(self.device).permute(1, 0)

        # Concatenate along batch dim
        obs = torch.cat([off_obs, on_obs], dim=1)
        actions = torch.cat([off_actions, on_actions], dim=1)
        rewards = torch.cat([off_rewards, on_rewards], dim=1)
        continues = torch.cat([off_continues, on_continues], dim=1)

        prev_state = self.world_model.get_initial_state(obs.shape[1], self.device)

        self.wm_optimizer.zero_grad()
        loss, metrics = self.world_model.compute_world_loss(
            obs, actions, rewards, continues, prev_state
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.wm_optimizer.step()

        return metrics

    def _train_ac_online(self):
        """Actor-critic training in imagination from buffer start states (Phase 4)."""
        cfg = self.config
        obs, _, _, _ = self.buffer.sample(cfg.batch_size, 1)
        obs = torch.FloatTensor(obs).to(self.device).squeeze(1)

        with torch.no_grad():
            embed = self.world_model.encode(obs)
            prev_state = self.world_model.get_initial_state(cfg.batch_size, self.device)
            dummy_action = torch.zeros(cfg.batch_size, cfg.action_dim, device=self.device)
            states, _ = self.world_model.rssm.observe(
                embed.unsqueeze(0), dummy_action.unsqueeze(0), prev_state
            )
            start_state = states[-1]

        return self._train_ac_imagination(start_state)

    # ========================================================================
    #  Helpers
    # ========================================================================

    def _default_env(self):
        from envs.metadrive_bev import MetaDriveBEVEnv
        return MetaDriveBEVEnv(dict(
            bev_size=self.config.bev_size,
            map_config=self.config.map_config,
            traffic_density=self.config.traffic_density,
            horizon=self.config.horizon,
        ))

    def _compute_lambda_returns(self, rewards, continues, values, gamma=0.997, lam=0.95):
        H, B = rewards.shape
        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1] + gamma * continues[-1] * values[-1]
        for t in reversed(range(H - 1)):
            next_val = rewards[t] + gamma * continues[t] * values[t + 1]
            returns[t] = next_val + lam * continues[t] * (returns[t + 1] - values[t + 1])
        return returns

    def _update_slow_critic(self):
        for p_slow, p_fast in zip(self.slow_critic.parameters(), self.critic.parameters()):
            p_slow.data.lerp_(p_fast.data, self.slow_tau)

    def _write_log(self, entry):
        path = os.path.join(self.logdir, "training_log.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()

    def _save(self, tag):
        path = os.path.join(self.logdir, f"checkpoint_{tag}.pt")
        ckpt = {
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config.to_dict(),
        }
        torch.save(ckpt, path)
        # Also save as latest for easy --resume latest
        latest_path = os.path.join(self.logdir, "checkpoint_latest.pt")
        torch.save(ckpt, latest_path)
        print(f"[Trainer] Saved: {path}")

    def _evaluate(self, env, num_episodes):
        """Run evaluation episodes with deterministic policy."""
        total_rewards = []
        for _ in range(num_episodes):
            obs, info = env.reset()
            state = self.world_model.get_initial_state(1, self.device)
            ep_reward = 0.0

            for t in range(self.config.horizon):
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    embed = self.world_model.encode(obs_t)
                    action_t = torch.zeros(1, self.config.action_dim, device=self.device)
                    states, _ = self.world_model.rssm.observe(
                        embed.unsqueeze(0), action_t.unsqueeze(0), state
                    )
                    state = states[-1]
                    feature = self.world_model.get_feature(state)
                    action = self.actor.act(feature, deterministic=True).cpu().numpy()[0]

                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                if done:
                    break

            total_rewards.append(ep_reward)

        return np.mean(total_rewards)
