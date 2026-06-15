#!/usr/bin/env python3
"""
SIGReg-Dreamer & DreamerV3(KL): Model-Based RL for Highway Merge-in Decision

Three-phase training (see docs/analysis/thesis_plan.md):
  Phase 2: Offline world model pretraining on exiD data
  Phase 3: Imagination policy training (frozen world model)
  Phase 4: Online finetuning in MetaDrive simulator

Usage:
  # Phase 2: SIGReg world model
  python main.py --phase phase2 --data-dir /path/to/exid_dreamer_data \
      --logdir logs/sigreg --seed 42

  # Phase 2: KL baseline (just change --reg)
  python main.py --phase phase2 --data-dir /path/to/exid_dreamer_data \
      --reg kl --logdir logs/kl --seed 42

  # Phase 3: Imagination training (load Phase 2 checkpoint)
  python main.py --phase phase3 --resume logs/sigreg/checkpoint_best.pt \
      --data-dir /path/to/exid_dreamer_data --logdir logs/sigreg_p3

  # Phase 4: Online finetuning
  python main.py --phase phase4 --resume logs/sigreg_p3/checkpoint_best.pt

  # Quick smoke test
  python main.py --mode test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import random
import numpy as np
import torch

from config import Config
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="SIGReg-Dreamer for Highway Merge")
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "test", "eval"])
    p.add_argument("--phase", type=str, default="phase2",
                   choices=["phase2", "phase3", "phase4", "bc"],
                   help="Training phase: phase2 (WM) | phase3 (imagination) | bc (behavior cloning) | phase4 (online)")
    p.add_argument("--reg", type=str, default="sigreg",
                   choices=["sigreg", "kl", "none"],
                   help="Regularization: sigreg (ours) | kl (DreamerV3)")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Path to exid_dreamer_data/loc{0,2,4,5,6}/")
    p.add_argument("--logdir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--bev-size", type=int, default=300,
                   help="BEV input resolution (center-cropped from 400x300)")
    p.add_argument("--sigreg-lambda", type=float, default=0.1)
    p.add_argument("--sigreg-target", type=str, default="deter",
                   choices=["stoch", "deter", "logits", "deter+logits"],
                   help="SIGReg target: continuous state to regularize")
    p.add_argument("--kl-beta", type=float, default=1.0)
    p.add_argument("--use-decoder", type=lambda x: x.lower() in ("true", "1", "yes"),
                   default=None, help="Use pixel decoder (True) or decoder-free (False)")
    p.add_argument("--barlow-lambda", type=float, default=0.005,
                   help="Barlow Twins loss weight (decoder-free only)")
    p.add_argument("--barlow-k", type=int, default=1,
                   help="Temporal offset for Barlow Twins alignment")
    p.add_argument("--log-every", type=int, default=None,
                   help="Log metrics every N steps (default: config value)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batch-length", type=int, default=None,
                   help="Sequence length (must be > jepa_k if JEPA enabled)")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint path, 'latest' to auto-detect in logdir")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="Custom WandB run name (default: {reg}_p2_s{seed})")
    p.add_argument("--preload", action="store_true",
                   help="Preload all npz files into RAM (eliminates I/O bottleneck)")
    p.add_argument("--bev-downsample", type=str, default="cnn",
                   choices=["bilinear", "cnn", "unshuffle"],
                   help="BEV downsampling: cnn (learnable strided conv) | unshuffle (lossless pixel unshuffle) | bilinear (not recommended)")
    p.add_argument("--cnn-factor", type=int, default=2,
                   choices=[2, 4],
                   help="CNN downsampling factor: 2 (300→150) or 4 (300→75)")
    p.add_argument("--e2e", action="store_true",
                   help="End-to-end: train WM+AC jointly in Phase 2 (skip Phase 3)")
    p.add_argument("--dapo", action="store_true",
                   help="DAPO: group-based PG in Phase 3 (no critic, no value inflation)")
    p.add_argument("--dapo-k", type=int, default=8,
                   help="DAPO group size (K action sequences per start state)")
    p.add_argument("--use-phase-head", action="store_true",
                   help="Add merge-phase prediction auxiliary head (Topology-Guided WM)")
    p.add_argument("--phase-head-weight", type=float, default=1.0,
                   help="Weight of phase classification loss")
    p.add_argument("--merge-zone-frames", type=int, default=20,
                   help="Frames after merge_idx classified as merge zone")
    p.add_argument("--bc-use-moe", action="store_true",
                   help="Use MoE-Phase actor in BC: 3 expert heads + classifier (hard routing train / soft infer)")
    p.add_argument("--use-speed-head", action="store_true",
                   help="Add ego speed prediction auxiliary head (from positions)")
    p.add_argument("--speed-head-weight", type=float, default=1.0,
                   help="Weight of speed prediction loss")
    p.add_argument("--use-spatial-head", action="store_true",
                   help="Add coarse BEV spatial layout prediction head")
    p.add_argument("--spatial-head-weight", type=float, default=0.5,
                   help="Weight of spatial prediction loss")
    p.add_argument("--spatial-head-resolution", type=int, default=32,
                   help="Coarse BEV resolution for spatial head (default: 32)")
    p.add_argument("--use-jepa", action="store_true",
                   help="Enable JEPA future-feature prediction (dense self-supervised signal)")
    p.add_argument("--jepa-weight", type=float, default=0.1,
                   help="Weight of JEPA loss (default: 0.1)")
    p.add_argument("--jepa-k", type=int, default=1,
                   help="JEPA prediction horizon: predict t+k from t (default: 1)")
    p.add_argument("--use-traj-head", action="store_true",
                   help="Add ego-centric trajectory prediction head (future Δx,Δy)")
    p.add_argument("--traj-head-weight", type=float, default=0.1,
                   help="Weight of trajectory prediction loss (default: 0.1)")
    p.add_argument("--traj-horizon", type=int, default=125,
                   help="Trajectory prediction horizon in frames (default: 125 = 5s)")
    p.add_argument("--use-curvature-head", action="store_true",
                   help="Add road curvature prediction head (computed from ego positions)")
    p.add_argument("--curvature-head-weight", type=float, default=0.1,
                   help="Weight of curvature prediction loss (default: 0.1)")
    p.add_argument("--use-vehicle-head", action="store_true",
                   help="Add surrounding vehicle prediction head (ego-centric dx,dy,vx,vy)")
    p.add_argument("--vehicle-head-weight", type=float, default=0.1,
                   help="Weight of vehicle prediction loss (default: 0.1)")
    p.add_argument("--n-surrounding-vehicles", type=int, default=5,
                   help="Number of nearest surrounding vehicles to predict (default: 5)")
    p.add_argument("--rssm-phase-conditional", action="store_true",
                   help="Plan A: Phase-conditional RSSM dynamics (3 heads gated by GT phase)")
    p.add_argument("--rssm-moe", action="store_true",
                   help="Plan B: MoE-style RSSM dynamics (router + 3 experts)")
    p.add_argument("--rssm-moe-experts", type=int, default=3,
                   help="Number of MoE experts (default: 3)")
    p.add_argument("--rssm-moe-lb-weight", type=float, default=0.01,
                   help="MoE load balancing loss weight (default: 0.01)")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def _find_latest_ckpt(logdir):
    """Find the latest checkpoint by step number in logdir."""
    import glob as _glob
    import re as _re
    ckpts = _glob.glob(os.path.join(logdir, "checkpoint_step*.pt"))
    if not ckpts:
        lat = os.path.join(logdir, "checkpoint_latest.pt")
        if os.path.exists(lat):
            return lat
        return None
    best = max(ckpts, key=lambda p: int(_re.search(r'step(\d+)', p).group(1)))
    return best


def main():
    args = parse_args()

    # Build config based on regularization type
    if args.reg == "kl":
        cfg = Config.kl_default(data_dir=args.data_dir)
        cfg.logdir = args.logdir or "./logs/dreamer_kl"
    else:
        cfg = Config.sigreg_default(data_dir=args.data_dir)
        cfg.logdir = args.logdir or "./logs/sigreg_dreamer"

    # Apply overrides
    cfg.regularization = args.reg  # Override reg method from CLI
    cfg.seed = args.seed
    cfg.bev_size = args.bev_size
    cfg.sigreg_lambda = args.sigreg_lambda
    cfg.sigreg_target = args.sigreg_target
    cfg.kl_beta = args.kl_beta
    cfg.batch_size = args.batch_size
    if args.batch_length is not None:
        cfg.batch_length = args.batch_length
    if args.log_every is not None:
        cfg.log_every = args.log_every
    cfg.train_phase = args.phase
    if args.use_decoder is not None:
        cfg.use_decoder = args.use_decoder
    cfg.barlow_lambda = args.barlow_lambda
    cfg.barlow_k = args.barlow_k
    cfg.preload_data = args.preload
    cfg.end_to_end = args.e2e
    cfg.use_dapo = args.dapo
    cfg.dapo_group_size = args.dapo_k
    cfg.use_phase_head = args.use_phase_head
    cfg.phase_head_weight = args.phase_head_weight
    cfg.merge_zone_frames = args.merge_zone_frames
    cfg.bc_use_moe = args.bc_use_moe
    cfg.use_speed_head = args.use_speed_head
    cfg.speed_head_weight = args.speed_head_weight
    cfg.use_spatial_head = args.use_spatial_head
    cfg.spatial_head_weight = args.spatial_head_weight
    cfg.spatial_head_resolution = args.spatial_head_resolution
    cfg.use_jepa = args.use_jepa
    cfg.jepa_weight = args.jepa_weight
    cfg.jepa_k = args.jepa_k
    cfg.use_traj_head = args.use_traj_head
    cfg.traj_head_weight = args.traj_head_weight
    cfg.traj_horizon = args.traj_horizon
    cfg.use_curvature_head = args.use_curvature_head
    cfg.curvature_head_weight = args.curvature_head_weight
    cfg.use_vehicle_head = args.use_vehicle_head
    cfg.vehicle_head_weight = args.vehicle_head_weight
    cfg.n_surrounding_vehicles = args.n_surrounding_vehicles
    cfg.rssm_phase_conditional = args.rssm_phase_conditional
    cfg.rssm_moe = args.rssm_moe
    cfg.rssm_moe_experts = args.rssm_moe_experts
    cfg.rssm_moe_load_balance_weight = args.rssm_moe_lb_weight
    cfg.bev_downsample = args.bev_downsample
    cfg.cnn_factor = args.cnn_factor
    if cfg.bev_downsample in ("cnn", "unshuffle"):
        if cfg.preload_data:
            print("[Main] WARNING: --preload disabled for CNN/unshuffle mode "
                  "(300x300 raw images too large for RAM, using mmap instead)")
        cfg.preload_data = False  # CNN mode: raw images from mmap (too large for RAM)
    if args.wandb_run_name is not None:
        cfg.wandb_run_name = args.wandb_run_name

    if args.total_steps:
        if args.phase == "phase2":
            cfg.total_wm_steps = args.total_steps
        elif args.phase in ("phase3", "bc"):
            cfg.total_imagine_steps = args.total_steps
        else:
            cfg.total_env_steps = args.total_steps

    # Set default merge_endpoints_path if not specified
    if not cfg.merge_endpoints_path:
        default_ep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_endpoints.json")
        if os.path.exists(default_ep):
            cfg.merge_endpoints_path = default_ep
            print(f"[Main] Using merge endpoints: {default_ep}")

    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"[Main] GPU {args.gpu}: {torch.cuda.get_device_name()}")
    else:
        print("[Main] CPU mode")

    # Print config
    print(f"\n{'='*60}")
    reg_name = {'sigreg': 'SIGReg', 'kl': 'DreamerV3(KL)', 'none': 'NoReg'}[args.reg]
    mode = "Decoder-Free" if not cfg.use_decoder else "Decoder"
    phase_label = f"{args.phase}" + ("+E2E" if args.e2e else "") + ("+DAPO" if args.dapo else "")
    print(f"{reg_name} | {mode} | {phase_label} | bev={cfg.bev_size} | seed={cfg.seed}")
    print(f"  data_dir={cfg.data_dir}")
    print(f"  reg={cfg.regularization} "
          f"(sigreg_λ={cfg.sigreg_lambda}, kl_β={cfg.kl_beta})")
    if not cfg.use_decoder:
        print(f"  barlow_λ={cfg.barlow_lambda} k={cfg.barlow_k}")
    if cfg.use_phase_head:
        print(f"  phase_head: merge_zone={cfg.merge_zone_frames}f, weight={cfg.phase_head_weight}")
    if cfg.use_speed_head:
        print(f"  speed_head: λ={cfg.speed_head_weight}")
    if cfg.use_spatial_head:
        print(f"  spatial_head: {cfg.spatial_head_resolution}x{cfg.spatial_head_resolution} λ={cfg.spatial_head_weight}")
    if cfg.use_jepa:
        print(f"  jepa: k={cfg.jepa_k}, λ={cfg.jepa_weight}, hidden={cfg.jepa_hidden}")
    if cfg.use_traj_head:
        print(f"  traj_head: H={cfg.traj_horizon} ({cfg.traj_horizon*0.04:.1f}s), λ={cfg.traj_head_weight}")
    if cfg.use_curvature_head:
        print(f"  curvature_head: λ={cfg.curvature_head_weight}")
    if cfg.use_vehicle_head:
        print(f"  vehicle_head: N={cfg.n_surrounding_vehicles}, λ={cfg.vehicle_head_weight}")
    if cfg.rssm_phase_conditional:
        print(f"  rssm_phase_conditional: 3 independent dynamics heads (Phase-Conditional RSSM)")
    if cfg.rssm_moe:
        print(f"  rssm_moe: {cfg.rssm_moe_experts} experts, lb_weight={cfg.rssm_moe_load_balance_weight}")
    print(f"  steps={cfg.total_steps}")
    print(f"{'='*60}\n")

    if args.mode == "test":
        _quick_test(cfg)
    elif args.mode == "train":
        trainer = Trainer(cfg)
        if args.resume:
            if args.resume == "latest":
                ckpt = _find_latest_ckpt(cfg.logdir)
                if ckpt:
                    print(f"[Main] Auto-resuming from {ckpt}")
                    _resume(trainer, ckpt)
                else:
                    print("[Main] No checkpoint found, starting fresh")
            else:
                _resume(trainer, args.resume)
        trainer.train()
    elif args.mode == "eval":
        trainer = Trainer(cfg)
        if args.resume:
            if args.resume == "latest":
                ckpt = _find_latest_ckpt(cfg.logdir)
                if ckpt:
                    _resume(trainer, ckpt)
            else:
                _resume(trainer, args.resume)
        env = trainer._default_env()
        eval_reward = trainer._evaluate(env, cfg.eval_episodes)
        env.close()
        print(f"[Eval] Avg reward: {eval_reward:.2f}")


def _quick_test(cfg):
    """Smoke test of all model components."""
    print("[Test] Running quick smoke test...")
    from models import WorldModel, Actor, Critic, SIGReg, Symlog
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SIGReg
    print("\n1. SIGReg...")
    sigreg = SIGReg(embed_dim=32 * 64, num_projections=1024).to(device)
    z = torch.randn(16, 32 * 64, device=device)
    loss = sigreg(z)
    print(f"   SIGReg loss: {loss.item():.6f}")

    # World Model
    print("\n2. World Model forward...")
    wm = WorldModel(cfg).to(device)
    obs = torch.randn(4, cfg.input_channels, cfg.bev_size, cfg.bev_size, device=device)
    embed = wm.encode(obs)
    print(f"   Encoder: {obs.shape} -> {embed.shape}")

    state = wm.get_initial_state(4, device)
    actions = torch.randn(8, 4, 2, device=device)
    embeds = embed.unsqueeze(0).expand(8, -1, -1)
    post_states, priors = wm.observe(embeds, actions, state)
    print(f"   RSSM observe: {len(post_states)} steps")

    dyn_metrics = wm.rssm.compute_loss(post_states, priors)
    print(f"   Dynamics loss ({dyn_metrics['reg_type']}): {dyn_metrics['total_dyn_loss']:.6f}")

    features = torch.stack([wm.get_feature(s) for s in post_states])
    if wm.use_decoder:
        recon = wm.decoder(features.reshape(-1, features.shape[-1]))
        print(f"   Decoder: {features.shape} -> {recon.shape}")
    else:
        barlow = wm.compute_barlow_loss(features)
        print(f"   Decoder-free (Barlow): loss={barlow.item():.6f}")

    imag_states = wm.imagine(actions, state)
    print(f"   Imagine: {len(imag_states)} steps")

    # Actor-Critic
    print("\n3. Actor-Critic...")
    feat_dim = wm.feature_dim()
    actor = Actor(feat_dim, action_dim=2).to(device)
    critic = Critic(feat_dim).to(device)

    feat = wm.get_feature(post_states[-1])
    action, log_prob = actor(feat)
    value, _ = critic(feat)
    print(f"   Actor: action={action[0].detach().cpu().numpy()} "
          f"log_prob={log_prob[0].item():.4f}")
    print(f"   Critic: value={value[0].item():.4f}")

    total = sum(p.numel() for p in
                list(wm.parameters()) + list(actor.parameters()) + list(critic.parameters()))
    print(f"\n4. Total params: {total/1e6:.2f}M")
    print("\n[Test] All components OK!")


def _resume(trainer, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=trainer.device)
    # strict=False: skip missing keys (e.g. phase_head not in old ckpts)
    trainer.world_model.load_state_dict(ckpt["world_model"], strict=False)
    if "actor" in ckpt:
        trainer.actor.load_state_dict(ckpt["actor"], strict=False)
    if "critic" in ckpt:
        trainer.critic.load_state_dict(ckpt["critic"], strict=False)
    trainer.global_step = ckpt.get("global_step", 0)
    print(f"[Main] Resumed from step {trainer.global_step}")


if __name__ == "__main__":
    main()
