"""
VLA (Vision-Language-Action) World Model Architecture v2

This module implements the complete VLA + World Model architecture that combines:
1. SIGLIP 2 Vision Encoder - For rich visual representations
2. Perceiver Resampler - For compressing visual tokens
3. Optional Language Conditioning - For instruction following
4. DreamerV3 World Model (RSSM) - For temporal dynamics and imagination
5. Flow Matching Action Head - For multi-modal action generation

Architecture Diagram (v2):
                                                                      
    ┌──────────────────────────────────────────────────────────────────┐
    │                    VLA World Model v2                             │
    └──────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
    ┌─────────────┐         ┌─────────────┐         ┌─────────────────┐
    │   Image     │         │  Language   │         │ Proprioceptive  │
    │   (RGB)     │         │ Instruction │         │    States       │
    └──────┬──────┘         └──────┬──────┘         └────────┬────────┘
           │                       │                         │
           ▼                       ▼                         │
    ┌─────────────┐         ┌─────────────┐                  │
    │ SIGLIP ViT  │         │  Language   │                  │
    │  Encoder    │         │   Encoder   │                  │
    └──────┬──────┘         └──────┬──────┘                  │
           │                       │                         │
           ▼                       │                         │
    ┌─────────────┐                │                         │
    │  Perceiver  │◄───────────────┘                         │
    │  Resampler  │                                          │
    └──────┬──────┘                                          │
           │                                                 │
           └─────────────────────┬───────────────────────────┘
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
                    │  Flow Matching      │
                    │  Action Head        │
                    │  ┌───────────────┐  │
                    │  │ Velocity Net  │  │
                    │  │ (DiT-style)   │  │
                    │  └───────────────┘  │
                    │         │           │
                    │         ▼           │
                    │  Action Chunk       │
                    │  [a_1, ..., a_H]    │
                    └─────────────────────┘

Key Improvements over v1:
- Flow Matching for multi-modal action distributions
- Action Chunking for temporal consistency
- Perceiver Resampler for efficient visual processing
- Language conditioning ready
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
        # Filter out VLA-specific kwargs that shouldn't be passed to nn.Linear
        vla_specific_keys = {
            'use_visual_residual', 'use_cross_attention',
            'attn_hidden', 'attn_heads', 'hidden', 'layers'
        }
        self.kw = {k: v for k, v in kw.items() if k not in vla_specific_keys}
    
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
            # Handle shape mismatch between x and visual_features
            # x: (B, T, D) or (B, D), visual_features might be (B*T, D) or (B, T, D)
            vf = visual_features
            if vf.ndim < x.ndim:
                # visual_features is flattened, reshape to match x
                if bdims > 1 and x.ndim == 3:
                    # x is (B, T, D), vf might be (B*T, D_vf)
                    B, T = x.shape[:2]
                    if vf.shape[0] == B * T:
                        vf = vf.reshape((B, T, vf.shape[-1]))
                    else:
                        # Cannot reshape, skip residual
                        vf = None
            elif vf.ndim > x.ndim:
                # visual_features has more dims, reduce
                while vf.ndim > x.ndim:
                    vf = vf.mean(axis=-2)  # Average pool over sequence dim
            
            if vf is not None:
                # Project visual features to same dimension as x
                vis_proj = self.sub('vis_proj', nn.Linear, x.shape[-1], **self.kw)(vf)
                x = x + vis_proj  # Residual connection
        
        # MLP policy network
        for i in range(self.layers):
            x = self.sub(f'policy{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'policy{i}norm', nn.Norm, self.norm)(x))
        
        # Output action distribution
        outputs = {}
        for key, space in self.act_space.items():
            actdim = int(np.prod(space.shape)) if space.shape else 1
            if space.discrete:
                logits = self.sub(f'{key}_logits', nn.Linear, actdim, **self.kw)(x)
                outputs[key] = embodied.jax.outs.OneHot(logits, unimix=0.01)
            else:
                # Continuous action (e.g., steering, throttle)
                mean = self.sub(f'{key}_mean', nn.Linear, actdim, **self.kw)(x)
                # Use bounded normal for actions with known bounds
                std_raw = self.sub(f'{key}_std', nn.Linear, actdim, **self.kw)(x)
                std = jax.nn.softplus(std_raw) + 0.1  # Minimum std
                # Squeeze last dim for scalar actions (shape=())
                if not space.shape:
                    mean = mean.squeeze(-1)
                    std = std.squeeze(-1)
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


# ============================================================================
# VLA v2 Components - Flow Matching + Perceiver Integration
# ============================================================================

class VLAWorldModelV2(nj.Module):
    """
    VLA + World Model Architecture v2 with Flow Matching.
    
    Key differences from v1:
    1. Uses Perceiver Resampler for visual token compression
    2. Supports language conditioning
    3. Uses Flow Matching for action generation
    4. Supports action chunking
    """
    
    # Architecture config
    perceiver_latents: int = 64
    perceiver_dim: int = 512
    use_language: bool = False
    flow_hidden: int = 512
    flow_layers: int = 4
    chunk_size: int = 1
    inference_steps: int = 10
    norm: str = 'rms'
    act: str = 'silu'
    
    def __init__(
        self,
        obs_space: Dict,
        act_space: Dict,
        encoder,
        rssm,
        decoder,
        **kw
    ):
        """Initialize VLA v2."""
        self.obs_space = obs_space
        self.act_space = act_space
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.kw = kw
        
        # Import flow matching components
        from . import flow_matching as fm
        
        # Perceiver for visual compression
        self._perceiver = None  # Lazy init
        
        # Flow matching action head
        self._flow_head = None  # Lazy init
        
        # Language encoder (optional)
        self._lang_encoder = None
    
    def get_perceiver(self):
        """Get or create Perceiver Resampler."""
        if self._perceiver is None:
            from . import flow_matching as fm
            self._perceiver = self.sub(
                'perceiver',
                fm.PerceiverResampler,
                num_latents=self.perceiver_latents,
                latent_dim=self.perceiver_dim,
                hidden=self.perceiver_dim,
                norm=self.norm,
                **self.kw
            )
        return self._perceiver
    
    def get_flow_head(self):
        """Get or create Flow Matching head."""
        if self._flow_head is None:
            from . import flow_matching as fm
            self._flow_head = self.sub(
                'flow_head',
                fm.FlowMatchingActionHead,
                self.act_space,
                hidden=self.flow_hidden,
                layers=self.flow_layers,
                chunk_size=self.chunk_size,
                inference_steps=self.inference_steps,
                norm=self.norm,
                act=self.act,
                **self.kw
            )
        return self._flow_head
    
    def observe(
        self,
        carry: Tuple,
        obs: Dict,
        action: Dict,
        reset: jnp.ndarray,
        language: Optional[jnp.ndarray] = None,
        training: bool = True
    ) -> Tuple[Tuple, Dict, jnp.ndarray, jnp.ndarray]:
        """
        Process observations with optional language conditioning.
        
        Returns:
            Tuple of (carry, entries, world_feat, visual_tokens)
        """
        enc_carry, dyn_carry, dec_carry = carry
        
        # Encode visual observations
        enc_carry, enc_entries, visual_tokens = self.encoder(
            enc_carry, obs, reset, training
        )
        
        # Apply Perceiver Resampler for compression
        # Note: This requires visual_tokens to be (B, T, N, D) or similar
        # For now, skip if tokens are already compressed
        compressed_tokens = visual_tokens
        
        # Process through RSSM
        dyn_carry, dyn_entries, feat = self.rssm.observe(
            dyn_carry, compressed_tokens, action, reset, training
        )
        
        carry = (enc_carry, dyn_carry, dec_carry)
        entries = {'enc': enc_entries, 'dyn': dyn_entries}
        
        return carry, entries, feat, visual_tokens
    
    def policy(
        self,
        world_feat: jnp.ndarray,
        visual_tokens: Optional[jnp.ndarray] = None,
        bdims: int = 2
    ) -> Dict:
        """
        Generate action using Flow Matching.
        
        Args:
            world_feat: RSSM features (deter + stoch)
            visual_tokens: Optional visual features for conditioning
            bdims: Number of batch dimensions
            
        Returns:
            Action distribution dictionary
        """
        flow_head = self.get_flow_head()
        return flow_head(world_feat, visual_tokens, bdims)
    
    def flow_loss(
        self,
        world_feat: jnp.ndarray,
        visual_tokens: Optional[jnp.ndarray],
        target_actions: jnp.ndarray,
        bdims: int = 2
    ) -> jnp.ndarray:
        """
        Compute Flow Matching loss for training.
        
        Args:
            world_feat: RSSM features
            visual_tokens: Visual features
            target_actions: Ground truth actions
            bdims: Number of batch dimensions
            
        Returns:
            Loss tensor of shape (B, T)
        """
        flow_head = self.get_flow_head()
        return flow_head.loss(world_feat, visual_tokens, target_actions, bdims)


class FlowMatchingPolicyHead(nj.Module):
    """
    Simplified Flow Matching Policy Head for direct integration.
    
    This version is designed to be a drop-in replacement for VLAPolicyHead
    while using flow matching internally.
    """
    
    hidden: int = 512
    layers: int = 4
    heads: int = 8
    norm: str = 'rms'
    act: str = 'silu'
    chunk_size: int = 1
    inference_steps: int = 10
    use_visual_residual: bool = False
    use_cross_attention: bool = False
    attn_hidden: int = 512
    attn_heads: int = 8
    visual_dim: int = 1280  # Default SIGLIP output dim, used for layer init
    
    def __init__(self, act_space: Dict, **kw):
        self.act_space = act_space
        # Filter out flow-specific kwargs that shouldn't be passed to nn.Linear
        flow_specific_keys = {
            'use_transformer', 'chunk_size', 'inference_steps', 
            'use_visual_residual', 'use_cross_attention',
            'attn_hidden', 'attn_heads', 'hidden', 'layers', 'heads',
            'visual_dim'
        }
        self.kw = {k: v for k, v in kw.items() if k not in flow_specific_keys}
        # Compute action dimension
        self.act_dim = sum(
            int(np.prod(s.shape)) if s.shape else 1 
            for s in act_space.values() 
            if not s.discrete
        )
        self.act_keys = [k for k, s in act_space.items() if not s.discrete]
        self.discrete_keys = [k for k, s in act_space.items() if s.discrete]
    
    def __call__(
        self,
        world_state: jnp.ndarray,
        visual_features: Optional[jnp.ndarray] = None,
        bdims: int = 2
    ) -> Dict:
        """
        Generate actions using flow matching.
        """
        # Prepare conditioning
        condition = self._encode_condition(world_state, visual_features, bdims)
        
        # Sample from flow - use same dtype as condition
        batch_shape = world_state.shape[:bdims]
        seed = nj.seed()
        x_0 = jax.random.normal(seed, batch_shape + (self.act_dim,), dtype=condition.dtype)
        
        # ODE integration
        x_1 = self._flow_ode(x_0, condition, bdims)
        
        # Convert to distributions
        return self._to_action_dists(x_1, batch_shape)
    
    def flow_loss(
        self,
        world_state: jnp.ndarray,
        visual_features: Optional[jnp.ndarray],
        target_actions: jnp.ndarray,
        bdims: int = 2
    ) -> jnp.ndarray:
        """Compute flow matching training loss."""
        batch_shape = world_state.shape[:bdims]
        condition = self._encode_condition(world_state, visual_features, bdims)
        
        # Flatten target actions if needed
        if isinstance(target_actions, dict):
            target_actions = jnp.concatenate(
                [target_actions[k] for k in self.act_keys], axis=-1
            )
        
        # Sample timestep - use same dtype as condition
        dtype = condition.dtype
        seed = nj.seed()
        t = jax.random.uniform(seed, batch_shape, dtype=dtype)
        
        # Sample noise
        seed2 = nj.seed()
        x_0 = jax.random.normal(seed2, target_actions.shape, dtype=dtype)
        
        # Interpolate
        t_exp = t[..., None]
        x_t = (1 - t_exp) * x_0 + t_exp * target_actions
        
        # Target velocity
        v_target = target_actions - x_0
        
        # Predict velocity
        v_pred = self._velocity_net(x_t, t, condition, bdims)
        
        # MSE loss
        loss = jnp.square(v_pred - v_target).mean(axis=-1)
        return loss
    
    def _encode_condition(
        self,
        world_state: jnp.ndarray,
        visual_features: Optional[jnp.ndarray],
        bdims: int
    ) -> jnp.ndarray:
        """Encode conditioning inputs."""
        x = self.sub('cond_proj', nn.Linear, self.hidden, **self.kw)(world_state)
        x = nn.act(self.act)(self.sub('cond_norm', nn.Norm, self.norm)(x))
        
        # Always create vf_proj layer if use_visual_residual is True
        # This ensures the layer exists even when visual_features is None during init
        if self.use_visual_residual:
            # Create dummy input to initialize layer if visual_features is None
            if visual_features is None:
                # Create zeros with expected visual_dim to initialize the layer
                dummy_shape = x.shape[:-1] + (self.visual_dim,)
                dummy_vf = jnp.zeros(dummy_shape, dtype=x.dtype)
                _ = self.sub('vf_proj', nn.Linear, self.hidden, **self.kw)(dummy_vf)
            else:
                vf = visual_features
                if vf.ndim < x.ndim:
                    if bdims > 1 and x.ndim == 3:
                        B, T = x.shape[:2]
                        if vf.shape[0] == B * T:
                            vf = vf.reshape((B, T, vf.shape[-1]))
                        else:
                            vf = None
                
                if vf is not None:
                    vf_proj = self.sub('vf_proj', nn.Linear, self.hidden, **self.kw)(vf)
                    x = x + vf_proj
        
        return x
    
    def _velocity_net(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        condition: jnp.ndarray,
        bdims: int
    ) -> jnp.ndarray:
        """Predict velocity field."""
        # Time embedding - use same dtype as condition
        dtype = condition.dtype
        t_emb = self._time_embed(t.astype(dtype), dtype)
        t_emb = self.sub('time_proj', nn.Linear, self.hidden, **self.kw)(t_emb)
        
        # Combine
        cond = condition + t_emb
        
        # Project noisy action
        x = self.sub('action_in', nn.Linear, self.hidden, **self.kw)(x_t)
        x = x + cond
        
        # MLP layers
        for i in range(self.layers):
            x = self.sub(f'vel_{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'vel_norm_{i}', nn.Norm, self.norm)(x))
        
        # Output
        x = self.sub('vel_out', nn.Linear, self.act_dim, **self.kw)(x)
        return x
    
    def _time_embed(self, t: jnp.ndarray, dtype=None) -> jnp.ndarray:
        """Sinusoidal time embedding."""
        if dtype is None:
            dtype = t.dtype
        dim = self.hidden
        half = dim // 2
        freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=dtype) / half)
        args = t[..., None] * freqs
        return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    
    def _flow_ode(
        self,
        x_0: jnp.ndarray,
        condition: jnp.ndarray,
        bdims: int
    ) -> jnp.ndarray:
        """Euler ODE integration."""
        dtype = condition.dtype
        dt = 1.0 / self.inference_steps
        x = x_0.astype(dtype)
        
        for i in range(self.inference_steps):
            t = jnp.full(condition.shape[:-1], i * dt, dtype=dtype)
            v = self._velocity_net(x, t, condition, bdims)
            x = x + v * dt
        
        return x
    
    def _to_action_dists(
        self,
        actions: jnp.ndarray,
        batch_shape: Tuple
    ) -> Dict:
        """Convert to action distributions."""
        outputs = {}
        offset = 0
        
        for key in self.act_keys:
            space = self.act_space[key]
            dim = int(np.prod(space.shape)) if space.shape else 1
            
            action = actions[..., offset:offset + dim]
            if not space.shape:
                action = action.squeeze(-1)
            
            # Clip to valid range
            if hasattr(space, 'low') and hasattr(space, 'high'):
                action = jnp.clip(action, space.low, space.high)
            
            # 使用可学习的std而不是固定的0.01
            # 使entropy可以正常计算和优化
            std = self.sub(f'{key}_std', nn.Linear, dim if dim > 0 else 1, **self.kw)(
                actions if actions.shape[-1] >= self.hidden // 2 else 
                self.sub(f'{key}_std_proj', nn.Linear, self.hidden, **self.kw)(actions)
            )
            std = nn.act('softplus')(std) + 0.1  # min_std=0.1, max_std~1.0
            if not space.shape:
                std = std.squeeze(-1)
            
            outputs[key] = embodied.jax.outs.Normal(action, std)
            offset += dim
        
        # Handle discrete actions with simple MLP
        for key in self.discrete_keys:
            space = self.act_space[key]
            x = self.sub(f'{key}_logits', nn.Linear, space.classes, **self.kw)(
                actions if actions.shape[-1] == self.hidden else 
                self.sub(f'{key}_proj', nn.Linear, self.hidden, **self.kw)(actions)
            )
            outputs[key] = embodied.jax.outs.OneHot(x, unimix=0.01)
        
        return outputs
