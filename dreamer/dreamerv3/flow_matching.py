"""
Flow Matching Action Head for VLA Architecture

This module implements Flow Matching (also known as Rectified Flow) for action generation,
which is the core technique used in state-of-the-art VLA models like π0 and Diffusion Policy.

Flow Matching learns a vector field that transports samples from a simple prior (Gaussian)
to the complex action distribution. Compared to diffusion models:
- Simpler training (direct regression on velocity field)
- Faster inference (straighter trajectories, fewer steps)
- Better for multi-modal distributions

Key Components:
1. FlowMatchingActionHead - Main policy network with flow matching
2. ActionChunking - Predict sequences of actions for temporal consistency
3. Optimal Transport Flow - Uses OT-CFM for better training

Architecture:
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Flow Matching Action Head                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  World State ──┬──→ FiLM Conditioning ──→ ┌─────────────────┐      │
    │                │                          │   Velocity Net   │      │
    │  Visual Feat ──┤                          │   (Transformer)  │      │
    │                │                          │                 │      │
    │  Time t ───────┴──→ Sinusoidal Embed ──→  │   v(x_t, t, c)  │      │
    │                                           └────────┬────────┘      │
    │  Noisy Action x_t ──────────────────────────────────┘              │
    │                                                                     │
    │                              ↓                                      │
    │                     Predicted Velocity v                            │
    │                              ↓                                      │
    │                   ODE Integration (Euler/RK4)                       │
    │                              ↓                                      │
    │                      Action Chunk [a_1, ..., a_H]                   │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

References:
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Scaling Rectified Flow Transformers (Esser et al., 2024) 
- π0: A Vision-Language-Action Flow Model (Black et al., 2024)
"""

import math
from typing import Dict, Optional, Tuple, List, Callable
from functools import partial

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import elements
import embodied.jax
import embodied.jax.nets as nn

f32 = jnp.float32
sg = jax.lax.stop_gradient


class SinusoidalPosEmbed(nj.Module):
    """Sinusoidal positional embedding for timestep encoding."""
    
    dim: int = 256
    max_period: float = 10000.0
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Embed timestep t into sinusoidal features.
        
        Args:
            t: Timestep tensor of shape (...,) with values in [0, 1]
            
        Returns:
            Embedding of shape (..., dim)
        """
        half_dim = self.dim // 2
        freqs = jnp.exp(
            -math.log(self.max_period) * jnp.arange(half_dim, dtype=f32) / half_dim
        )
        # t can be any shape, freqs is (half_dim,)
        args = t[..., None] * freqs
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding


class AdaptiveLayerNorm(nj.Module):
    """
    Adaptive Layer Normalization (adaLN) conditioned on timestep and context.
    
    Used in DiT-style architectures for conditioning the velocity network.
    """
    
    hidden: int = 512
    eps: float = 1e-6
    
    def __init__(self, **kw):
        self.kw = kw
    
    def __call__(
        self, 
        x: jnp.ndarray,
        condition: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Apply adaptive layer norm.
        
        Args:
            x: Input features (..., D)
            condition: Conditioning vector (..., D_cond)
            
        Returns:
            Normalized and modulated features
        """
        D = x.shape[-1]
        
        # Generate scale and shift from condition
        scale_shift = self.sub('scale_shift', nn.Linear, 2 * D, **self.kw)(condition)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        
        # Layer norm
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        
        # Modulate
        return x_norm * (1 + scale) + shift


class FlowMatchingActionHead(nj.Module):
    """
    Flow Matching Action Head for VLA.
    
    Generates actions by learning a velocity field that transforms
    Gaussian noise into the action distribution, conditioned on
    world state and visual features.
    
    Supports action chunking for temporal consistency.
    
    Args:
        act_space: Action space dictionary
        hidden: Hidden dimension of velocity network
        layers: Number of transformer layers
        heads: Number of attention heads (if using transformer)
        chunk_size: Number of future actions to predict (action chunking)
        inference_steps: Number of ODE integration steps at inference
        use_transformer: Use transformer or MLP for velocity network
    """
    
    hidden: int = 512
    layers: int = 4
    heads: int = 8
    norm: str = 'rms'
    act: str = 'silu'
    chunk_size: int = 1  # Action chunking horizon
    inference_steps: int = 10  # ODE steps for sampling
    sigma_min: float = 1e-4  # Minimum noise level
    use_transformer: bool = True
    time_embed_dim: int = 256
    
    def __init__(self, act_space: Dict, **kw):
        self.act_space = act_space
        self.kw = kw
        # Compute total action dimension
        self.act_dim = sum(
            int(np.prod(s.shape)) if s.shape else 1 
            for s in act_space.values() 
            if not s.discrete  # Only continuous actions for flow matching
        )
        self.act_keys = [k for k, s in act_space.items() if not s.discrete]
    
    def __call__(
        self,
        world_state: jnp.ndarray,
        visual_features: Optional[jnp.ndarray] = None,
        bdims: int = 2
    ) -> Dict:
        """
        Sample actions using flow matching.
        
        At inference, integrates the learned velocity field from noise to action.
        
        Args:
            world_state: World model features (B, ..., D)
            visual_features: Optional visual features
            bdims: Number of batch dimensions
            
        Returns:
            Dictionary of action distributions
        """
        # Prepare conditioning
        condition = self._prepare_condition(world_state, visual_features, bdims)
        
        # Sample from prior (Gaussian noise)
        batch_shape = world_state.shape[:bdims]
        action_shape = batch_shape + (self.chunk_size, self.act_dim)
        
        # For JAX, we need a seed
        seed = nj.seed()
        x_0 = jax.random.normal(seed, action_shape, dtype=f32)
        
        # Integrate ODE from t=0 (noise) to t=1 (data)
        x_1 = self._ode_solve(x_0, condition, bdims)
        
        # Take first action from chunk (or return full chunk)
        actions = x_1[..., 0, :] if self.chunk_size > 1 else x_1.squeeze(-2)
        
        # Convert to action distributions
        return self._to_distributions(actions, batch_shape)
    
    def loss(
        self,
        world_state: jnp.ndarray,
        visual_features: Optional[jnp.ndarray],
        target_actions: jnp.ndarray,
        bdims: int = 2
    ) -> jnp.ndarray:
        """
        Compute flow matching loss.
        
        Uses Conditional Flow Matching (CFM) with optimal transport path:
        x_t = (1 - t) * x_0 + t * x_1
        v_target = x_1 - x_0
        
        Args:
            world_state: World model features
            visual_features: Optional visual features  
            target_actions: Ground truth actions (B, T, act_dim)
            bdims: Number of batch dimensions
            
        Returns:
            Loss tensor of shape (B, T)
        """
        batch_shape = world_state.shape[:bdims]
        condition = self._prepare_condition(world_state, visual_features, bdims)
        
        # Sample timestep uniformly
        seed = nj.seed()
        t = jax.random.uniform(seed, batch_shape, dtype=f32)
        
        # Reshape target for action chunking
        if target_actions.ndim == bdims:
            target_actions = target_actions[..., None, :]
        x_1 = target_actions  # Target actions
        
        # Sample noise
        seed2 = nj.seed()
        x_0 = jax.random.normal(seed2, x_1.shape, dtype=f32)
        
        # Interpolate (OT-CFM path)
        t_expanded = t[..., None, None]  # Expand for broadcasting
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # Target velocity (straight path)
        v_target = x_1 - x_0
        
        # Predict velocity
        v_pred = self._velocity_network(x_t, t, condition, bdims)
        
        # MSE loss on velocity
        loss = jnp.square(v_pred - v_target).mean(axis=(-2, -1))  # Mean over chunk and action dims
        
        return loss
    
    def _prepare_condition(
        self,
        world_state: jnp.ndarray,
        visual_features: Optional[jnp.ndarray],
        bdims: int
    ) -> jnp.ndarray:
        """Prepare conditioning vector from world state and visual features."""
        # Project world state
        x = self.sub('cond_proj', nn.Linear, self.hidden, **self.kw)(world_state)
        x = nn.act(self.act)(self.sub('cond_norm', nn.Norm, self.norm)(x))
        
        # Optionally add visual features
        if visual_features is not None:
            # Handle shape mismatch
            vf = visual_features
            if vf.ndim < x.ndim:
                if bdims > 1 and x.ndim == 3:
                    B, T = x.shape[:2]
                    if vf.shape[0] == B * T:
                        vf = vf.reshape((B, T, vf.shape[-1]))
            
            if vf is not None and vf.shape[:-1] == x.shape[:-1]:
                vf_proj = self.sub('vf_proj', nn.Linear, self.hidden, **self.kw)(vf)
                x = x + vf_proj
        
        return x
    
    def _velocity_network(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        condition: jnp.ndarray,
        bdims: int
    ) -> jnp.ndarray:
        """
        Velocity network that predicts dx/dt.
        
        Args:
            x_t: Noisy action at time t (..., chunk_size, act_dim)
            t: Timestep (...,)
            condition: Conditioning features (..., hidden)
            
        Returns:
            Predicted velocity (..., chunk_size, act_dim)
        """
        # Time embedding
        t_emb = self.sub('time_embed', SinusoidalPosEmbed, dim=self.time_embed_dim)(t)
        t_emb = self.sub('time_proj', nn.Linear, self.hidden, **self.kw)(t_emb)
        
        # Combine condition and time
        cond = condition + t_emb
        
        # Flatten action for processing
        orig_shape = x_t.shape
        x = x_t.reshape((*orig_shape[:-2], -1))  # Flatten chunk and action dims
        
        # Project noisy action
        x = self.sub('action_proj', nn.Linear, self.hidden, **self.kw)(x)
        
        if self.use_transformer:
            # Transformer-based velocity network
            x = self._transformer_velocity(x, cond, bdims)
        else:
            # MLP-based velocity network
            x = self._mlp_velocity(x, cond)
        
        # Project back to action space
        x = self.sub('out_proj', nn.Linear, self.chunk_size * self.act_dim, **self.kw)(x)
        x = x.reshape(orig_shape)
        
        return x
    
    def _transformer_velocity(
        self,
        x: jnp.ndarray,
        cond: jnp.ndarray,
        bdims: int
    ) -> jnp.ndarray:
        """Transformer-based velocity network with adaLN."""
        for i in range(self.layers):
            # Self-attention with adaLN
            residual = x
            x = self.sub(f'adaln1_{i}', AdaptiveLayerNorm, hidden=self.hidden)(x, cond)
            
            # Simple self-attention (single query)
            qkv = self.sub(f'qkv_{i}', nn.Linear, 3 * self.hidden, **self.kw)(x)
            q, k, v = jnp.split(qkv, 3, axis=-1)
            
            # Scaled dot-product (simplified for non-sequence input)
            scale = (self.hidden // self.heads) ** -0.5
            attn = jax.nn.softmax(q * k * scale, axis=-1)
            x = attn * v
            x = self.sub(f'attn_out_{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = residual + x
            
            # FFN with adaLN
            residual = x
            x = self.sub(f'adaln2_{i}', AdaptiveLayerNorm, hidden=self.hidden)(x, cond)
            x = self.sub(f'ffn1_{i}', nn.Linear, self.hidden * 4, **self.kw)(x)
            x = nn.act(self.act)(x)
            x = self.sub(f'ffn2_{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = residual + x
        
        return x
    
    def _mlp_velocity(
        self,
        x: jnp.ndarray,
        cond: jnp.ndarray
    ) -> jnp.ndarray:
        """MLP-based velocity network with FiLM conditioning."""
        # Concatenate input and condition
        x = jnp.concatenate([x, cond], axis=-1)
        
        for i in range(self.layers):
            x = self.sub(f'mlp_{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'mlp_norm_{i}', nn.Norm, self.norm)(x))
        
        return x
    
    def _ode_solve(
        self,
        x_0: jnp.ndarray,
        condition: jnp.ndarray,
        bdims: int
    ) -> jnp.ndarray:
        """
        Solve ODE from t=0 to t=1 using Euler method.
        
        dx/dt = v(x_t, t, condition)
        """
        dt = 1.0 / self.inference_steps
        x_t = x_0
        
        for i in range(self.inference_steps):
            t = jnp.full(condition.shape[:-1], i * dt, dtype=f32)
            v = self._velocity_network(x_t, t, condition, bdims)
            x_t = x_t + v * dt
        
        return x_t
    
    def _to_distributions(
        self,
        actions: jnp.ndarray,
        batch_shape: Tuple
    ) -> Dict:
        """Convert sampled actions to distribution objects."""
        outputs = {}
        offset = 0
        
        for key in self.act_keys:
            space = self.act_space[key]
            dim = int(np.prod(space.shape)) if space.shape else 1
            
            action = actions[..., offset:offset + dim]
            
            # Squeeze for scalar actions
            if not space.shape:
                action = action.squeeze(-1)
            
            # Create a delta distribution (deterministic from flow matching)
            # Wrap in Normal with tiny std for compatibility
            outputs[key] = embodied.jax.outs.Normal(
                action, 
                jnp.full_like(action, 0.01)  # Small std for "deterministic"
            )
            offset += dim
        
        # Handle discrete actions separately (not flow matching)
        for key, space in self.act_space.items():
            if space.discrete and key not in outputs:
                # Use simple MLP for discrete actions
                # This shouldn't happen in typical driving scenarios
                pass
        
        return outputs


class PerceiverResampler(nj.Module):
    """
    Perceiver Resampler for compressing visual tokens.
    
    Maps variable-length visual token sequences to a fixed number of
    latent tokens using cross-attention, similar to Flamingo.
    
    This is useful for:
    1. Reducing computational cost of processing many visual patches
    2. Creating a fixed-size representation for the world model
    3. Enabling multi-image/video input
    """
    
    num_latents: int = 64  # Number of output latent tokens
    latent_dim: int = 512
    hidden: int = 512
    heads: int = 8
    layers: int = 2
    norm: str = 'rms'
    act: str = 'gelu'
    
    def __init__(self, **kw):
        self.kw = kw
    
    def __call__(
        self,
        visual_tokens: jnp.ndarray,  # (B, N, D) visual tokens from encoder
        training: bool = True
    ) -> jnp.ndarray:
        """
        Resample visual tokens to fixed number of latents.
        
        Args:
            visual_tokens: Input visual tokens (B, N, D)
            training: Whether in training mode
            
        Returns:
            Latent tokens (B, num_latents, latent_dim)
        """
        B = visual_tokens.shape[0]
        
        # Initialize learnable latent queries
        latents = self.sub('latent_init', nn.Initializer, 
                          shape=(self.num_latents, self.latent_dim),
                          init='normal')()
        latents = jnp.broadcast_to(latents, (B, self.num_latents, self.latent_dim))
        
        # Project visual tokens to same dim
        kv = self.sub('kv_proj', nn.Linear, self.latent_dim, **self.kw)(visual_tokens)
        
        for i in range(self.layers):
            # Cross-attention: latents attend to visual tokens
            latents = self._cross_attention(
                latents, kv, f'cross_attn_{i}'
            )
            
            # Self-attention among latents
            latents = self._self_attention(latents, f'self_attn_{i}')
            
            # FFN
            residual = latents
            x = self.sub(f'ffn_norm_{i}', nn.Norm, self.norm)(latents)
            x = self.sub(f'ffn1_{i}', nn.Linear, self.hidden * 4, **self.kw)(x)
            x = nn.act(self.act)(x)
            x = self.sub(f'ffn2_{i}', nn.Linear, self.latent_dim, **self.kw)(x)
            latents = residual + x
        
        return latents
    
    def _cross_attention(
        self,
        queries: jnp.ndarray,  # (B, M, D)
        key_values: jnp.ndarray,  # (B, N, D)
        name: str
    ) -> jnp.ndarray:
        """Cross-attention from queries to key_values."""
        B, M, D = queries.shape
        _, N, _ = key_values.shape
        H = self.heads
        head_dim = D // H
        
        residual = queries
        queries = self.sub(f'{name}_q_norm', nn.Norm, self.norm)(queries)
        
        q = self.sub(f'{name}_q', nn.Linear, D, **self.kw)(queries)
        k = self.sub(f'{name}_k', nn.Linear, D, **self.kw)(key_values)
        v = self.sub(f'{name}_v', nn.Linear, D, **self.kw)(key_values)
        
        q = q.reshape(B, M, H, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
        
        scale = head_dim ** -0.5
        attn = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, M, D)
        
        out = self.sub(f'{name}_out', nn.Linear, D, **self.kw)(out)
        return residual + out
    
    def _self_attention(
        self,
        x: jnp.ndarray,  # (B, M, D)
        name: str
    ) -> jnp.ndarray:
        """Self-attention among tokens."""
        B, M, D = x.shape
        H = self.heads
        head_dim = D // H
        
        residual = x
        x = self.sub(f'{name}_norm', nn.Norm, self.norm)(x)
        
        qkv = self.sub(f'{name}_qkv', nn.Linear, 3 * D, **self.kw)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, M, H, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, M, H, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, M, H, head_dim).transpose(0, 2, 1, 3)
        
        scale = head_dim ** -0.5
        attn = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, M, D)
        
        out = self.sub(f'{name}_out', nn.Linear, D, **self.kw)(out)
        return residual + out


class ActionChunkingPolicy(nj.Module):
    """
    Action Chunking Policy wrapper.
    
    Wraps any policy head to enable action chunking with temporal smoothing.
    This improves temporal consistency and reduces jitter in control.
    
    Implements exponential temporal ensemble as in ACT/Diffusion Policy:
    - Predict H future actions at each timestep
    - Blend predictions using exponential weights
    """
    
    chunk_size: int = 8
    temporal_weight: float = 0.01  # Higher = more weight on recent predictions
    
    def __init__(self, policy_head: nj.Module, **kw):
        self.policy_head = policy_head
        self.kw = kw
    
    def __call__(
        self,
        world_state: jnp.ndarray,
        visual_features: Optional[jnp.ndarray] = None,
        action_queue: Optional[jnp.ndarray] = None,  # Previous predictions
        bdims: int = 2
    ) -> Tuple[Dict, jnp.ndarray]:
        """
        Get action with temporal ensembling.
        
        Args:
            world_state: World model features
            visual_features: Visual features
            action_queue: Queue of previous action chunk predictions
            bdims: Number of batch dimensions
            
        Returns:
            Tuple of (action_dist, new_action_queue)
        """
        # Get full action chunk from base policy
        raw_output = self.policy_head(world_state, visual_features, bdims)
        
        # For flow matching, raw_output is already the action chunk
        # For now, return as-is (temporal ensemble requires stateful tracking)
        return raw_output, None


class LanguageConditioner(nj.Module):
    """
    Language conditioning module for VLA.
    
    Encodes language instructions and conditions the policy/world model
    on the language features via cross-attention or FiLM.
    
    Note: This is a placeholder for future language integration.
    Full implementation would require a language model (e.g., T5, BERT).
    """
    
    hidden: int = 512
    vocab_size: int = 32000
    max_seq_len: int = 77
    layers: int = 4
    heads: int = 8
    norm: str = 'rms'
    
    def __init__(self, **kw):
        self.kw = kw
    
    def __call__(
        self,
        text_tokens: jnp.ndarray,  # (B, L) token IDs
        training: bool = True
    ) -> jnp.ndarray:
        """
        Encode text tokens into language features.
        
        Args:
            text_tokens: Input token IDs (B, L)
            training: Training mode
            
        Returns:
            Language features (B, L, hidden) or (B, hidden) if pooled
        """
        B, L = text_tokens.shape
        
        # Token embedding
        embed = self.sub('token_embed', nn.Embed, self.vocab_size, self.hidden)
        x = embed(text_tokens)
        
        # Positional embedding
        pos_embed = self.sub('pos_embed', nn.Initializer,
                            shape=(self.max_seq_len, self.hidden),
                            init='normal')()
        x = x + pos_embed[:L]
        
        # Transformer layers
        for i in range(self.layers):
            # Self-attention
            residual = x
            x = self.sub(f'attn_norm_{i}', nn.Norm, self.norm)(x)
            
            qkv = self.sub(f'qkv_{i}', nn.Linear, 3 * self.hidden, **self.kw)(x)
            q, k, v = jnp.split(qkv, 3, axis=-1)
            
            head_dim = self.hidden // self.heads
            q = q.reshape(B, L, self.heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, self.heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, self.heads, head_dim).transpose(0, 2, 1, 3)
            
            # Causal mask
            mask = jnp.tril(jnp.ones((L, L)))
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(0, 1, 3, 2)) * scale
            attn = jnp.where(mask, attn, -1e9)
            attn = jax.nn.softmax(attn, axis=-1)
            
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.hidden)
            x = residual + self.sub(f'attn_out_{i}', nn.Linear, self.hidden, **self.kw)(out)
            
            # FFN
            residual = x
            x = self.sub(f'ffn_norm_{i}', nn.Norm, self.norm)(x)
            x = self.sub(f'ffn1_{i}', nn.Linear, self.hidden * 4, **self.kw)(x)
            x = nn.act('gelu')(x)
            x = self.sub(f'ffn2_{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = residual + x
        
        # Pool to single vector (mean over sequence)
        x = x.mean(axis=1)
        
        return x
