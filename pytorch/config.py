"""
Configuration for SIGReg-Dreamer and DreamerV3(KL) experiments.

Two config presets share all hyperparameters except the regularization method:
  - sigreg: SIGReg (λ=0.1), our method
  - kl:     KL divergence (β=1.0, free_bits=1), DreamerV3 baseline
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # --- Observation ---
    bev_size: int = 300         # BEV input resolution (center-cropped from 400×300)
    input_channels: int = 3     # RGB channels

    # --- BEV downsampling: always use a strided frontend (no bilinear resize) ---
    bev_downsample: str = "cnn"  # "cnn" | "unshuffle" | "bilinear"
    cnn_factor: int = 2         # Downsampling factor: 2 (300→150) or 4 (300→75)

    # --- Action ---
    action_dim: int = 2         # [steering, throttle]

    # --- Encoder ---
    enc_depth: int = 64         # Base channel depth for CNN (64 → [128,192,256,256])

    # --- RSSM ---
    deter_dim: int = 1024       # GRU deterministic state
    stoch_dim: int = 32         # Number of stochastic variables
    stoch_classes: int = 64     # Classes per stochastic variable
    hidden_dim: int = 512       # MLP hidden size

    # --- Regularization (core experiment variable) ---
    regularization: str = "sigreg"  # "sigreg" | "kl" | "none"

    # SIGReg params
    sigreg_lambda: float = 0.1  # SIGReg weight (robust in [0.01, 0.2])
    sigreg_projections: int = 1024  # Random projection directions
    sigreg_target: str = "deter"  # "stoch" | "deter" | "logits" | "deter+logits"

    # KL params (DreamerV3 baseline)
    kl_beta: float = 1.0        # KL divergence weight
    kl_free_bits: float = 1.0   # Free bits to prevent posterior collapse

    # --- Decoder ---
    use_decoder: bool = True    # Set False for decoder-free mode
    barlow_lambda: float = 0.005  # Barlow Twins weight (decoder-free only)
    barlow_k: int = 1           # Temporal offset for Barlow alignment

    # --- Loss scales ---
    loss_scales: dict = field(default_factory=lambda: {
        "rec": 1.0, "dyn": 1.0, "rew": 1.0, "con": 1.0,
        "policy": 1.0, "value": 1.0,
    })

    # --- World model training ---
    embed_dim: int = 512        # Encoder output dim
    batch_size: int = 16
    batch_length: int = 50      # Sequence length (matches plan: seq_len=50)
    world_lr: float = 3e-4

    # --- Actor-Critic ---
    actor_hidden: int = 512
    actor_layers: int = 3
    critic_hidden: int = 512
    critic_layers: int = 3
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    imagination_horizon: int = 15
    gamma: float = 0.997
    lambda_gae: float = 0.5
    entropy_weight: float = 1e-4

    # --- Replay buffer ---
    replay_size: int = 500_000
    replay_warmup: int = 1000

    # --- Offline data ---
    data_dir: Optional[str] = None  # Path to exid_dreamer_data/loc{0,2,4,5,6}/
    offline_cache_size: int = 512   # Number of npz mmap handles cached (lightweight)
    preload_data: bool = False      # Preload ALL npz into RAM (~3GB, eliminates I/O bottleneck)

    # --- Environment (for Phase 4 online finetuning) ---
    env_name: str = "metadrive_on_ramp"
    map_config: str = "SSrSS"
    traffic_density: float = 0.3
    num_envs: int = 1
    horizon: int = 1000

    # --- DAPO (group-based policy gradient, no critic) ---
    use_dapo: bool = False      # Use DAPO instead of actor-critic for Phase 3
    dapo_group_size: int = 8    # K: number of action sequences per start state
    dapo_collision_weight: float = 5.0  # Collision proxy penalty weight in DAPO

    # --- Topology-Guided World Model (Phase-Aware auxiliary head) ---
    use_phase_head: bool = False  # Predict merge phase (pre / merge / post) from RSSM features
    phase_head_weight: float = 1.0  # Weight of phase classification loss
    merge_zone_frames: int = 20    # Frames after merge frame classified as "merge zone"
    merge_endpoints_path: str = ""  # Path to merge_endpoints.json (geometric endpoints per location)
    merge_zone_radius: float = 25.0  # Distance threshold (m) for merge zone detection

    # --- JEPA (Joint Embedding Predictive Architecture) ---
    # Predicts feature[t+k] from feature[t] + actions[t:t+k].
    # Dense 3072-dim self-supervised signal — replaces the content-guidance role
    # that pixel reconstruction played in original DreamerV3.
    use_jepa: bool = False        # Enable JEPA future-feature prediction
    jepa_weight: float = 0.1      # Weight of JEPA loss in total loss
    jepa_k: int = 1               # Predict t+k from t
    jepa_hidden: int = 1024       # Predictor MLP hidden size

    # --- RSSM Phase-Conditional / MoE dynamics (plan A & B) ---
    # Plan A: Ground-truth phase labels gate independent dynamics heads
    rssm_phase_conditional: bool = False  # Use 3 independent prior/posterior nets per phase
    # Plan B: Router + experts with softmax gating (self-discovered specialization)
    rssm_moe: bool = False               # Use Mixture-of-Experts dynamics
    rssm_moe_experts: int = 3            # Number of expert dynamics heads
    rssm_moe_load_balance_weight: float = 0.01  # Weight of load-balancing loss

    # --- Trajectory prediction head (ego-centric future displacements) ---
    # Predicts future (Δx, Δy) in ego frame from RSSM features.
    # Captures the ~10m lateral shift during merge — 100x stronger signal than steer.
    # H=125 (5s) forces WM to encode ramp curvature and merge trajectory.
    use_traj_head: bool = False     # Enable ego-centric trajectory prediction
    traj_head_weight: float = 0.1   # Weight of trajectory prediction loss
    traj_horizon: int = 125         # Number of future frames to predict (5s @ 25fps)

    # --- Curvature prediction head (road geometry from ego positions) ---
    # Predicts instantaneous road curvature (rad/m) from RSSM features.
    # Zero-cost label: computed from ego position heading changes.
    use_curvature_head: bool = False  # Enable road curvature prediction
    curvature_head_weight: float = 0.15  # Weight of curvature prediction loss

    # --- Surrounding vehicle prediction head (interaction awareness) ---
    # Predicts ego-centric (dx, dy, vx, vy) of N nearest surrounding vehicles.
    # Forces WM to encode "where are other cars and how fast are they moving?".
    # Labels from exiD CSV (add_surrounding_vehicles.py pre-processing).
    use_vehicle_head: bool = False     # Enable surrounding vehicle prediction
    vehicle_head_weight: float = 0.1   # Weight of vehicle prediction loss
    n_surrounding_vehicles: int = 5    # Number of nearest vehicles to predict

    # --- Auxiliary supervised heads (stronger signal than reward prediction) ---
    use_speed_head: bool = False    # Predict ego speed from RSSM features (regression)
    speed_head_weight: float = 1.0  # Weight of speed prediction loss
    use_spatial_head: bool = False  # Predict coarse BEV spatial layout from RSSM features
    spatial_head_weight: float = 0.5  # Weight of spatial prediction loss (MSE in symlog)
    spatial_head_resolution: int = 32  # Coarse BEV resolution for spatial head

    # --- Online Dreamer (exiD closed-loop) ---
    online_train_locs: tuple = (0, 2, 4, 5, 6)  # Training locations for online interaction
    online_eval_locs: tuple = (1, 3)             # Validation locations (zero-shot eval)
    online_offline_ratio: float = 0.5  # Fraction of WM batch from offline npz data
    online_collect_interval: int = 10  # Train WM+AC after every N episodes
    online_train_steps_per_collect: int = 50  # WM+AC training steps per collection round
    online_buffer_capacity: int = 100_000  # ReplayBuffer capacity for online data (~27GB for 300×300)

    # --- End-to-end (WM+AC joint training) ---
    end_to_end: bool = False    # Train WM+AC jointly in Phase 2 (skip Phase 3)
    joint_ac_every: int = 1     # Train AC every N WM steps

    # --- Training loop ---
    train_phase: str = "phase2"  # "phase2" | "phase3" | "phase4"
    total_wm_steps: int = 500_000          # Phase 2: world model training steps
    total_imagine_steps: int = 200_000     # Phase 3: imagination training steps
    total_env_steps: int = 1_000_000       # Phase 4: online finetuning steps
    log_every: int = 50
    save_every: int = 2000            # ~22h at 40s/step, fits 24h job
    eval_every: int = 2000
    eval_episodes: int = 10

    # --- Logging ---
    logdir: str = "./logs/sigreg_dreamer"
    seed: int = 42
    use_wandb: bool = True
    wandb_run_name: str = ""
    wandb_api_key: str = field(default_factory=lambda: os.environ.get("WANDB_API_KEY", ""))
    wandb_entity: str = field(default_factory=lambda: os.environ.get("WANDB_ENTITY", ""))

    # --- Slow critic (EMA target) ---
    slow_value_tau: float = 0.005

    @property
    def total_steps(self):
        """Compatibility: total training steps for current phase."""
        if self.train_phase == "phase2":
            return self.total_wm_steps
        elif self.train_phase == "phase3":
            return self.total_imagine_steps
        else:
            return self.total_env_steps

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def sigreg_default(cls, data_dir=None, **overrides):
        """SIGReg-Dreamer (our method)."""
        cfg = cls(regularization="sigreg", sigreg_lambda=0.1)
        if data_dir:
            cfg.data_dir = data_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def kl_default(cls, data_dir=None, **overrides):
        """DreamerV3(KL) baseline — identical except regularization."""
        cfg = cls(regularization="kl", kl_beta=1.0, kl_free_bits=1.0)
        if data_dir:
            cfg.data_dir = data_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg
