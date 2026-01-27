"""
VLA (Vision-Language-Action) World Model Architecture

This module implements the complete VLA + World Model architecture that combines:
1. SIGLIP 2 Vision Encoder - For rich visual representations
2. DreamerV3 World Model (RSSM) - For temporal dynamics and imagination
3. VLA-style Action Head - Leveraging both visual features and world state

Architecture Diagram:
                                                                      
    ┌──────────────────────────────────────────────────────────────────┐
    │                        VLA World Model                            │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                     Visual Observation (RGB)                      │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
    ┌─────────────────────┐               ┌─────────────────────┐
    │   SIGLIP 2 Vision   │               │  Proprioceptive     │
    │   Encoder (ViT)     │               │  Encoder (MLP)      │
    │   [Frozen/Finetune] │               │                     │
    └─────────┬───────────┘               └─────────┬───────────┘
              │                                     │
              └─────────────────┬───────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │   Projection Layer  │
                    │   (Adapter/Bridge)  │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   RSSM World Model  │
                    │   ┌─────────────┐   │
                    │   │ Deterministic│   │
                    │   │    GRU       │   │
                    │   └──────┬──────┘   │
                    │          │          │
                    │   ┌──────▼──────┐   │
                    │   │ Stochastic  │   │
                    │   │   Latent    │   │
                    │   └─────────────┘   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │   Decoder   │   │   Reward    │   │  Continue   │
    │  (Recons)   │   │    Head     │   │    Head     │
    └─────────────┘   └─────────────┘   └─────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  VLA Policy Head    │
                    │  ┌───────────────┐  │
                    │  │ World State   │──┼──→ Action
                    │  │ + Visual Feat │  │    (Steering, Throttle, Brake)
                    │  └───────────────┘  │
                    └─────────────────────┘

Future Extensions:
- Language conditioning for instruction following
- DAgger imitation learning integration  
- Multi-task policy heads
- Hierarchical action spaces
"""

import math
from typing import Dict, Optional, Tuple, Any, List

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import elements
import embodied.jax
import embodied.jax.nets as nn

f32 = jnp.float32
sg = jax.lax.stop_gradient


class VLAWorldModel(nj.Module):
    """
    Complete VLA + World Model architecture.
    
    This class orchestrates the full forward pass from visual observations
    to action predictions, integrating SIGLIP features with the RSSM world model.
    
    The key innovation is the fusion of:
    1. Rich semantic features from pretrained SIGLIP
    2. Temporal dynamics from the learned world model
    3. Future imagination for planning
    
    Args:
        obs_space: Observation space dictionary
        act_space: Action space dictionary  
        encoder_type: Type of visual encoder ('siglip', 'siglip_jax', 'simple')
        siglip_path: Path to pretrained SIGLIP model
        rssm_config: Configuration for RSSM world model
        policy_config: Configuration for policy head
    """
    
    # Architecture configuration
    fusion_type: str = 'concat'  # 'concat', 'cross_attention', 'film'
    use_visual_features_for_policy: bool = True
    policy_hidden: int = 1024
    policy_layers: int = 3
    norm: str = 'rms'
    act: str = 'silu'
    
    def __init__(
        self, 
        obs_space: Dict,
        act_space: Dict,
        encoder,
        rssm,
        decoder,
        reward_head,
        continue_head,
        policy_head,
        **kw
    ):
        """Initialize VLA World Model with components."""
        self.obs_space = obs_space
        self.act_space = act_space
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.reward_head = reward_head
        self.continue_head = continue_head
        self.policy_head = policy_head
        self.kw = kw
        
    def observe(
        self,
        carry: Tuple,
        obs: Dict,
        action: Dict,
        reset: jnp.ndarray,
        training: bool = True
    ) -> Tuple[Tuple, Dict, jnp.ndarray]:
        """
        Process observations through encoder and world model.
        
        Args:
            carry: (enc_carry, dyn_carry, dec_carry)
            obs: Observation dictionary
            action: Previous action dictionary
            reset: Episode reset flags
            training: Whether in training mode
            
        Returns:
            Tuple of (new_carry, entries, world_state_features)
        """
        enc_carry, dyn_carry, dec_carry = carry
        
        # Encode observations (SIGLIP or other encoder)
        enc_carry, enc_entries, tokens = self.encoder(
            enc_carry, obs, reset, training
        )
        
        # Process through RSSM world model
        dyn_carry, dyn_entries, feat = self.rssm.observe(
            dyn_carry, tokens, action, reset, training
        )
        
        carry = (enc_carry, dyn_carry, dec_carry)
        entries = {'enc': enc_entries, 'dyn': dyn_entries}
        
        return carry, entries, feat
    
    def imagine(
        self,
        start_state: Dict,
        policy_fn,
        horizon: int,
        training: bool = True
    ) -> Tuple[jnp.ndarray, Dict, Dict]:
        """
        Imagine future trajectories using the world model.
        
        Args:
            start_state: Initial world state
            policy_fn: Function mapping state to action
            horizon: Number of imagination steps
            training: Whether in training mode
            
        Returns:
            Tuple of (imagined_features, imagined_actions, imagined_states)
        """
        return self.rssm.imagine(start_state, policy_fn, horizon, training)
    
    def decode(
        self,
        carry: Dict,
        feat: jnp.ndarray,
        reset: jnp.ndarray,
        training: bool = True
    ) -> Tuple[Dict, Dict]:
        """
        Decode world state to reconstruct observations.
        
        Args:
            carry: Decoder carry state
            feat: World state features
            reset: Reset flags
            training: Whether in training mode
            
        Returns:
            Tuple of (carry, reconstructions)
        """
        dec_carry, dec_entries, recons = self.decoder(
            carry, feat, reset, training
        )
        return dec_carry, recons


class VLAPolicyHead(nj.Module):
    """
    VLA-style policy head that combines world state with visual features.
    
    This policy head is designed to leverage both:
    1. The world model's latent state (for temporal reasoning)
    2. Optional direct visual features (for fine-grained perception)
    
    For autonomous driving, outputs typically include:
    - Steering angle
    - Throttle/acceleration
    - Brake
    """
    
    hidden: int = 1024
    layers: int = 3
    norm: str = 'rms'
    act: str = 'silu'
    use_visual_residual: bool = False  # Add visual features as residual
    use_cross_attention: bool = False  # Use cross-attn from world state to visual
    attn_hidden: int = 512
    attn_heads: int = 8
    
    def __init__(self, act_space: Dict, **kw):
        self.act_space = act_space
        self.kw = kw
    
    def __call__(
        self,
        world_state: jnp.ndarray,
        visual_features: Optional[jnp.ndarray] = None,
        bdims: int = 2
    ):
        """
        Compute action distribution from world state.
        
        Args:
            world_state: Features from RSSM (deter + stoch)
            visual_features: Optional SIGLIP features for residual
            bdims: Number of batch dimensions
            
        Returns:
            Action distribution dictionary
        """
        x = world_state

        # Optional cross-attention to visual features
        if visual_features is not None and self.use_cross_attention:
            orig_shape = x.shape
            if bdims > 1:
                flat = int(math.prod(orig_shape[:bdims]))
                x_flat = x.reshape((flat, orig_shape[-1]))
                vis_shape = visual_features.shape
                if visual_features.ndim >= bdims + 1:
                    visual_features = visual_features.reshape(
                        (flat,) + vis_shape[bdims:]
                    )
            else:
                x_flat = x
            x_flat = self.sub(
                'cross_attn',
                CrossAttentionFusion,
                hidden=self.attn_hidden,
                heads=self.attn_heads,
                norm=self.norm,
                **self.kw
            )(x_flat, visual_features)
            if bdims > 1:
                x = x_flat.reshape(orig_shape)
            else:
                x = x_flat
        
        # Optionally fuse visual features
        if visual_features is not None and self.use_visual_residual:
            # Project visual features to same dimension
            vis_proj = self.sub('vis_proj', nn.Linear, x.shape[-1], **self.kw)(
                visual_features
            )
            x = x + vis_proj  # Residual connection
        
        # MLP policy network
        for i in range(self.layers):
            x = self.sub(f'policy{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'policy{i}norm', nn.Norm, self.norm)(x))
        
        # Output action distribution
        outputs = {}
        for key, space in self.act_space.items():
            if space.discrete:
                logits = self.sub(f'{key}_logits', nn.Linear, space.shape[0], **self.kw)(x)
                outputs[key] = embodied.jax.outs.OneHot(logits, unimix=0.01)
            else:
                # Continuous action (e.g., steering, throttle)
                mean = self.sub(f'{key}_mean', nn.Linear, space.shape[0], **self.kw)(x)
                # Use bounded normal for actions with known bounds
                std_raw = self.sub(f'{key}_std', nn.Linear, space.shape[0], **self.kw)(x)
                std = jax.nn.softplus(std_raw) + 0.1  # Minimum std
                outputs[key] = embodied.jax.outs.Normal(mean, std)
        
        return outputs


class CrossAttentionFusion(nj.Module):
    """
    Cross-attention fusion for combining visual features with world state.
    
    This allows the policy to attend to relevant visual features based on
    the current world model state, enabling more fine-grained control.
    """
    
    hidden: int = 512
    heads: int = 8
    norm: str = 'rms'
    
    def __init__(self, **kw):
        self.kw = kw
    
    def __call__(
        self,
        query: jnp.ndarray,  # World state (B, D)
        key_value: jnp.ndarray  # Visual features (B, N, D) or (B, D)
    ) -> jnp.ndarray:
        """
        Apply cross-attention from world state to visual features.
        
        Args:
            query: World model state features
            key_value: Visual encoder features
            
        Returns:
            Attended visual features fused with world state
        """
        # Ensure key_value has sequence dimension
        if key_value.ndim == 2:
            key_value = key_value[:, None, :]
        
        B, N, D = key_value.shape
        qdim = query.shape[-1]
        H = self.heads
        head_dim = self.hidden // H
        
        # Project query (world state)
        q = self.sub('q_proj', nn.Linear, self.hidden, **self.kw)(query)
        q = q[:, None, :]  # Add sequence dim: (B, 1, hidden)
        q = q.reshape(B, 1, H, head_dim).transpose(0, 2, 1, 3)
        
        # Project key and value (visual features)
        k = self.sub('k_proj', nn.Linear, self.hidden, **self.kw)(key_value)
        v = self.sub('v_proj', nn.Linear, self.hidden, **self.kw)(key_value)
        k = k.reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        
        # Apply attention
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, self.hidden)
        out = self.sub('out_proj', nn.Linear, qdim, **self.kw)(out)
        
        # Residual + norm
        out = self.sub('out_norm', nn.Norm, self.norm)(query + out)
        
        return out


class FiLMConditioning(nj.Module):
    """
    Feature-wise Linear Modulation (FiLM) for visual feature conditioning.
    
    FiLM allows the world model state to modulate visual features through
    learned scale and shift parameters, useful for attention-like behavior
    without the computational cost.
    """
    
    hidden: int = 512
    
    def __init__(self, **kw):
        self.kw = kw
    
    def __call__(
        self,
        features: jnp.ndarray,  # Visual features to modulate
        conditioning: jnp.ndarray  # World state for conditioning
    ) -> jnp.ndarray:
        """
        Apply FiLM conditioning.
        
        Args:
            features: Features to be modulated (B, D)
            conditioning: Conditioning signal (B, D_cond)
            
        Returns:
            Modulated features
        """
        D = features.shape[-1]
        
        # Generate scale (gamma) and shift (beta) from conditioning
        gamma = self.sub('gamma', nn.Linear, D, **self.kw)(conditioning)
        beta = self.sub('beta', nn.Linear, D, **self.kw)(conditioning)
        
        # Apply FiLM: y = gamma * x + beta
        return gamma * features + beta


class DAggerBuffer:
    """
    DAgger (Dataset Aggregation) buffer for imitation learning.
    
    Stores expert demonstrations and aggregated policy-expert pairs
    for DAgger-style imitation learning with the VLA architecture.
    
    Usage:
        1. Collect expert demonstrations
        2. Train policy on demonstrations
        3. Execute policy, query expert at some states
        4. Aggregate new data into buffer
        5. Repeat
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.observations = []
        self.actions = []
        self.is_expert = []  # Track which are expert vs policy samples
        self.position = 0
        self.full = False
    
    def add_expert_demo(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray]
    ):
        """Add expert demonstration to buffer."""
        batch_size = next(iter(observations.values())).shape[0]
        
        for i in range(batch_size):
            obs = {k: v[i] for k, v in observations.items()}
            act = {k: v[i] for k, v in actions.items()}
            
            if self.full:
                self.observations[self.position] = obs
                self.actions[self.position] = act
                self.is_expert[self.position] = True
            else:
                self.observations.append(obs)
                self.actions.append(act)
                self.is_expert.append(True)
            
            self.position = (self.position + 1) % self.capacity
            if self.position == 0:
                self.full = True
    
    def add_dagger_sample(
        self,
        observation: Dict[str, np.ndarray],
        expert_action: Dict[str, np.ndarray]
    ):
        """Add DAgger sample (policy observation + expert action)."""
        if self.full:
            self.observations[self.position] = observation
            self.actions[self.position] = expert_action
            self.is_expert[self.position] = False
        else:
            self.observations.append(observation)
            self.actions.append(expert_action)
            self.is_expert.append(False)
        
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True
    
    def sample(self, batch_size: int) -> Tuple[Dict, Dict]:
        """Sample a batch for training."""
        size = len(self.observations)
        indices = np.random.choice(size, min(batch_size, size), replace=False)
        
        obs_batch = {}
        act_batch = {}
        
        for key in self.observations[0].keys():
            obs_batch[key] = np.stack([self.observations[i][key] for i in indices])
        
        for key in self.actions[0].keys():
            act_batch[key] = np.stack([self.actions[i][key] for i in indices])
        
        return obs_batch, act_batch
    
    def __len__(self):
        return len(self.observations)


def create_vla_agent_components(obs_space, act_space, config):
    """
    Factory function to create VLA agent components.
    
    This creates all the necessary components for a VLA + World Model agent:
    - SIGLIP-based encoder
    - RSSM world model
    - VLA policy head
    - Reward and continue heads
    
    Args:
        obs_space: Observation space
        act_space: Action space  
        config: Agent configuration
        
    Returns:
        Dictionary of agent components
    """
    from . import rssm as rssm_module
    from . import siglip_encoder
    
    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    
    # Create encoder based on type
    enc_type = config.enc.typ
    if enc_type == 'siglip':
        encoder = siglip_encoder.SiglipVisionEncoder(
            enc_space, **dict(config.enc.siglip), name='enc'
        )
    elif enc_type == 'siglip_jax':
        encoder = siglip_encoder.SiglipEncoderJAX(
            enc_space, **dict(config.enc.siglip_jax), name='enc'
        )
    else:
        encoder = rssm_module.Encoder(
            enc_space, **dict(config.enc.simple), name='enc'
        )
    
    # Create RSSM world model
    rssm = rssm_module.RSSM(
        act_space, **dict(config.dyn.rssm), name='dyn'
    )
    
    # Create decoder
    decoder = rssm_module.Decoder(
        dec_space, **dict(config.dec.simple), name='dec'
    )
    
    return {
        'encoder': encoder,
        'rssm': rssm,
        'decoder': decoder,
        'enc_space': enc_space,
        'dec_space': dec_space,
    }
