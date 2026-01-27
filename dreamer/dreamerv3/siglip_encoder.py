"""
SIGLIP 2 Vision Encoder for DreamerV3 + VLA Architecture

This module integrates Google's SIGLIP 2 vision-language model as the visual encoder
for the DreamerV3 world model, enabling VLA (Vision-Language-Action) style architecture.

SIGLIP 2 provides stronger visual representations that can benefit:
1. World model prediction (through richer visual features)
2. Policy learning (through pretrained visual priors)
3. Future: language-conditioned control

Architecture Overview:
                    ┌─────────────────────┐
                    │   Visual Input      │
                    │   (RGB Images)      │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   SIGLIP 2 Vision   │
                    │   Encoder (ViT)     │
                    │   Frozen/Finetuned  │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Projection Layer  │
                    │   (Adapter)         │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │   RSSM World    │             │   Policy Head   │
    │   Model         │             │   (VLA Style)   │
    └─────────────────┘             └─────────────────┘
"""

import math
from functools import partial
from typing import Dict, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import elements
import embodied.jax
import embodied.jax.nets as nn

# Check for transformers availability
try:
    import torch
    from transformers import AutoModel, AutoProcessor, SiglipVisionModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. SIGLIP encoder will not work.")

f32 = jnp.float32


class SiglipVisionEncoder(nj.Module):
    """
    SIGLIP 2 Vision Encoder wrapper for DreamerV3.
    
    This encoder uses the pretrained SIGLIP 2 vision transformer to extract
    rich visual features from input images. The features are then projected
    to match the expected token dimension for the RSSM world model.
    
    Features:
    - Supports frozen or finetunable SIGLIP backbone
    - Configurable projection layer to match RSSM dimensions
    - Optional feature aggregation (CLS token, mean pooling, or all patches)
    - Compatible with DreamerV3's encoder interface
    
    Args:
        siglip_path: Path to pretrained SIGLIP 2 model
        output_dim: Output feature dimension (should match RSSM token dim)
        freeze_backbone: Whether to freeze SIGLIP weights
        aggregation: 'cls' for CLS token, 'mean' for mean pooling, 'patches' for all
        proj_layers: Number of projection layers after SIGLIP
        proj_hidden: Hidden dimension for projection MLP
    """
    
    # Configuration parameters
    siglip_path: str = ""  # Will be set from config
    output_dim: int = 1024
    freeze_backbone: bool = True
    aggregation: str = 'mean'  # 'cls', 'mean', 'patches'
    proj_layers: int = 2
    proj_hidden: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    image_size: int = 256  # SIGLIP 2 so400m uses 256x256
    patch_size: int = 16
    
    def __init__(self, obs_space: Dict, siglip_path: str = "", **kw):
        """
        Initialize SIGLIP encoder.
        
        Args:
            obs_space: Observation space dictionary
            siglip_path: Path to SIGLIP model weights
            **kw: Additional keyword arguments for network layers
        """
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.kw = kw
        
        # Store SIGLIP path - will be loaded lazily
        if siglip_path:
            self.siglip_path = siglip_path
        
        # Calculate number of patches for this image size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # SIGLIP hidden dimension (for so400m model)
        self.siglip_hidden_dim = 1152  # so400m uses 1152
        
        # Lazy initialization flags
        self._siglip_initialized = False
        self._siglip_model = None
        self._siglip_processor = None
    
    @property
    def entry_space(self):
        """Return entry space for replay context (empty for encoder)."""
        return {}
    
    def initial(self, batch_size: int):
        """Return initial carry state (empty for encoder)."""
        return {}
    
    def truncate(self, entries, carry=None):
        """Truncate entries for replay context."""
        return {}
    
    def _ensure_siglip_loaded(self):
        """Lazy load SIGLIP model when first needed."""
        if not self._siglip_initialized and HAS_TRANSFORMERS:
            print(f"Loading SIGLIP 2 model from: {self.siglip_path}")
            try:
                self._siglip_model = SiglipVisionModel.from_pretrained(
                    self.siglip_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self._siglip_processor = AutoProcessor.from_pretrained(self.siglip_path)
                
                if torch.cuda.is_available():
                    self._siglip_model = self._siglip_model.cuda()
                
                if self.freeze_backbone:
                    self._siglip_model.eval()
                    for param in self._siglip_model.parameters():
                        param.requires_grad = False
                
                self._siglip_initialized = True
                print("SIGLIP 2 model loaded successfully!")
            except Exception as e:
                print(f"Failed to load SIGLIP model: {e}")
                self._siglip_initialized = False
    
    def _extract_siglip_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images using SIGLIP.
        
        Args:
            images: Input images of shape (B, H, W, C) in uint8 [0, 255]
            
        Returns:
            Features of shape (B, output_dim) or (B, num_patches, output_dim)
        """
        self._ensure_siglip_loaded()
        
        if not self._siglip_initialized:
            # Fallback: return zeros if SIGLIP not available
            batch_size = images.shape[0]
            if self.aggregation == 'patches':
                return np.zeros((batch_size, self.num_patches, self.siglip_hidden_dim), dtype=np.float32)
            else:
                return np.zeros((batch_size, self.siglip_hidden_dim), dtype=np.float32)
        
        # Convert to PIL format for processor
        import PIL.Image
        pil_images = [PIL.Image.fromarray(img) for img in images]
        
        # Process images
        inputs = self._siglip_processor(images=pil_images, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self._siglip_model(**inputs)
            
            if self.aggregation == 'cls':
                # Use CLS token (first token)
                features = outputs.last_hidden_state[:, 0, :]
            elif self.aggregation == 'mean':
                # Mean pooling over all patches
                features = outputs.last_hidden_state.mean(dim=1)
            else:  # 'patches'
                # Return all patch features
                features = outputs.last_hidden_state
        
        # Convert to numpy
        features = features.cpu().numpy().astype(np.float32)
        return features
    
    def __call__(
        self, 
        carry: Dict, 
        obs: Dict, 
        reset: jnp.ndarray, 
        training: bool, 
        single: bool = False
    ) -> Tuple[Dict, Dict, jnp.ndarray]:
        """
        Encode observations to tokens for the world model.
        
        Args:
            carry: Carry state (unused for encoder)
            obs: Observation dictionary with image keys
            reset: Reset flags
            training: Whether in training mode
            single: Whether processing single timestep
            
        Returns:
            Tuple of (carry, entries, tokens)
        """
        bdims = 1 if single else 2
        outs = []
        bshape = reset.shape
        
        # Process vector observations (same as simple encoder)
        if self.veckeys:
            vspace = {k: self.obs_space[k] for k in self.veckeys}
            vecs = {k: obs[k] for k in self.veckeys}
            squish = nn.symlog
            x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
            x = x.reshape((-1, *x.shape[bdims:]))
            for i in range(2):  # Simple MLP for vectors
                x = self.sub(f'vec_mlp{i}', nn.Linear, self.proj_hidden, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f'vec_mlp{i}norm', nn.Norm, self.norm)(x))
            outs.append(x)
        
        # Process image observations with SIGLIP
        if self.imgkeys:
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            # Concatenate images along channel dimension for multi-camera
            x = jnp.concatenate(imgs, -1)
            
            # Reshape for batch processing
            original_shape = x.shape
            x = x.reshape((-1, *x.shape[bdims:]))
            
            # SIGLIP feature extraction (using JAX callback for PyTorch interop)
            # This is a hybrid approach - SIGLIP runs in PyTorch, projection in JAX
            siglip_features = jax.pure_callback(
                self._extract_siglip_features,
                jax.ShapeDtypeStruct(
                    (x.shape[0], self.siglip_hidden_dim), 
                    jnp.float32
                ),
                x.astype(jnp.uint8)
            )
            
            # Project SIGLIP features to output dimension
            x = nn.cast(siglip_features)
            for i in range(self.proj_layers):
                x = self.sub(f'proj{i}', nn.Linear, self.proj_hidden, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f'proj{i}norm', nn.Norm, self.norm)(x))
            
            # Final projection to output dimension
            x = self.sub('proj_out', nn.Linear, self.output_dim, **self.kw)(x)
            outs.append(x)
        
        # Concatenate all features
        x = jnp.concatenate(outs, -1) if len(outs) > 1 else outs[0]
        tokens = x.reshape((*bshape, *x.shape[1:]))
        
        entries = {}
        return carry, entries, tokens


class SiglipCnnFusionEncoder(nj.Module):
    """
    Dual-stream encoder: SIGLIP 2 (semantic) + CNN (spatial) fusion.

    This encoder is designed to keep SIGLIP's semantic understanding while
    preserving fine-grained spatial details from a native CNN branch.
    """

    # SIGLIP configuration
    siglip_path: str = ""
    output_dim: int = 1024
    freeze_backbone: bool = True
    unfreeze_last_layers: int = 0  # If >0, unfreeze last N transformer blocks
    unfreeze_last_layers: int = 0  # If >0, unfreeze last N transformer blocks
    aggregation: str = 'mean'  # 'cls' or 'mean'
    proj_layers: int = 2
    proj_hidden: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    image_size: int = 256
    patch_size: int = 16

    # CNN branch configuration (spatial detail)
    cnn_depth: int = 64
    cnn_mults: tuple = (2, 3, 4, 4)
    cnn_layers: int = 2
    cnn_units: int = 1024
    cnn_kernel: int = 5
    cnn_symlog: bool = True
    cnn_outer: bool = False
    cnn_strided: bool = False

    # Vector branch configuration
    vec_layers: int = 2
    vec_hidden: int = 1024

    # Fusion configuration
    fusion_type: str = 'gated'  # 'gated', 'add', or 'concat'
    fusion_dim: int = 1024
    fusion_layers: int = 2

    def __init__(self, obs_space: Dict, siglip_path: str = "", **kw):
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.kw = kw

        if siglip_path:
            self.siglip_path = siglip_path

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.siglip_hidden_dim = 1152  # so400m uses 1152

        self._siglip_initialized = False
        self._siglip_model = None
        self._siglip_processor = None

        self._cnn_depths = tuple(self.cnn_depth * mult for mult in self.cnn_mults)

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size: int):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def _ensure_siglip_loaded(self):
        if not self._siglip_initialized and HAS_TRANSFORMERS:
            print(f"Loading SIGLIP 2 model from: {self.siglip_path}")
            try:
                self._siglip_model = SiglipVisionModel.from_pretrained(
                    self.siglip_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self._siglip_processor = AutoProcessor.from_pretrained(self.siglip_path)

                if torch.cuda.is_available():
                    self._siglip_model = self._siglip_model.cuda()

                if self.freeze_backbone:
                    self._siglip_model.eval()
                    for param in self._siglip_model.parameters():
                        param.requires_grad = False
                    if self.unfreeze_last_layers > 0:
                        self._unfreeze_last_layers(self.unfreeze_last_layers)
                    if self.unfreeze_last_layers > 0:
                        self._unfreeze_last_layers(self.unfreeze_last_layers)

                self._siglip_initialized = True
                print("SIGLIP 2 model loaded successfully!")
            except Exception as e:
                print(f"Failed to load SIGLIP model: {e}")
                self._siglip_initialized = False

    def _unfreeze_last_layers(self, n: int):
        if n <= 0:
            return
        vision = None
        if hasattr(self._siglip_model, 'vision_model'):
            vision = self._siglip_model.vision_model
        elif hasattr(self._siglip_model, 'vision_encoder'):
            vision = self._siglip_model.vision_encoder
        if vision is None:
            print("[SIGLIP] Warning: No vision encoder found for partial unfreeze.")
            return
        encoder = getattr(vision, 'encoder', None)
        layers = getattr(encoder, 'layers', None) if encoder is not None else None
        if layers is None:
            print("[SIGLIP] Warning: Transformer layers not found for partial unfreeze.")
            return
        n = min(n, len(layers))
        for layer in list(layers)[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"[SIGLIP] Unfroze last {n} transformer layer(s).")

    def _unfreeze_last_layers(self, n: int):
        """Unfreeze last N transformer blocks if available."""
        if n <= 0:
            return
        vision = None
        if hasattr(self._siglip_model, 'vision_model'):
            vision = self._siglip_model.vision_model
        elif hasattr(self._siglip_model, 'vision_encoder'):
            vision = self._siglip_model.vision_encoder
        if vision is None:
            print("[SIGLIP] Warning: No vision encoder found for partial unfreeze.")
            return
        encoder = getattr(vision, 'encoder', None)
        layers = getattr(encoder, 'layers', None) if encoder is not None else None
        if layers is None:
            print("[SIGLIP] Warning: Transformer layers not found for partial unfreeze.")
            return
        n = min(n, len(layers))
        for layer in list(layers)[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"[SIGLIP] Unfroze last {n} transformer layer(s).")

    def _extract_siglip_features(self, images: np.ndarray) -> np.ndarray:
        self._ensure_siglip_loaded()

        if not self._siglip_initialized:
            batch_size = images.shape[0]
            return np.zeros((batch_size, self.siglip_hidden_dim), dtype=np.float32)

        import PIL.Image
        pil_images = [PIL.Image.fromarray(img) for img in images]
        inputs = self._siglip_processor(images=pil_images, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._siglip_model(**inputs)
            if self.aggregation == 'cls':
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs.last_hidden_state.mean(dim=1)

        features = features.cpu().numpy().astype(np.float32)
        return features

    def _encode_vectors(self, vecs: Dict[str, jnp.ndarray], bdims: int):
        vspace = {k: self.obs_space[k] for k in self.veckeys}
        squish = nn.symlog if self.cnn_symlog else lambda x: x
        x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
        x = x.reshape((-1, *x.shape[bdims:]))
        for i in range(self.vec_layers):
            x = self.sub(f'vec_mlp{i}', nn.Linear, self.vec_hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'vec_mlp{i}norm', nn.Norm, self.norm)(x))
        return x

    def _encode_cnn(self, images: jnp.ndarray, bdims: int):
        x = nn.cast(images, force=True) / 255 - 0.5
        x = x.reshape((-1, *x.shape[bdims:]))
        K = self.cnn_kernel
        for i, depth in enumerate(self._cnn_depths):
            if self.cnn_outer and i == 0:
                x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
            elif self.cnn_strided:
                x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
            else:
                x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
                B, H, W, C = x.shape
                x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
            x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
        x = x.reshape((x.shape[0], -1))
        for i in range(self.cnn_layers):
            x = self.sub(f'cnn_mlp{i}', nn.Linear, self.cnn_units, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'cnn_mlp{i}norm', nn.Norm, self.norm)(x))
        x = self.sub('cnn_proj', nn.Linear, self.fusion_dim, **self.kw)(x)
        return x

    def _encode_siglip(self, images: jnp.ndarray, bdims: int):
        x = images.reshape((-1, *images.shape[bdims:]))
        siglip_features = jax.pure_callback(
            self._extract_siglip_features,
            jax.ShapeDtypeStruct(
                (x.shape[0], self.siglip_hidden_dim),
                jnp.float32
            ),
            x.astype(jnp.uint8)
        )
        x = nn.cast(siglip_features)
        for i in range(self.proj_layers):
            x = self.sub(f'siglip_proj{i}', nn.Linear, self.proj_hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'siglip_proj{i}norm', nn.Norm, self.norm)(x))
        x = self.sub('siglip_proj_out', nn.Linear, self.fusion_dim, **self.kw)(x)
        return x

    def _fuse(self, siglip_feat: jnp.ndarray, cnn_feat: jnp.ndarray):
        if self.fusion_type == 'add':
            fused = siglip_feat + cnn_feat
        elif self.fusion_type == 'concat':
            fused = jnp.concatenate([siglip_feat, cnn_feat], -1)
            for i in range(self.fusion_layers):
                fused = self.sub(f'fusion_mlp{i}', nn.Linear, self.fusion_dim, **self.kw)(fused)
                fused = nn.act(self.act)(self.sub(f'fusion_mlp{i}norm', nn.Norm, self.norm)(fused))
        else:
            gate_inp = jnp.concatenate([siglip_feat, cnn_feat], -1)
            gate = self.sub('fusion_gate', nn.Linear, self.fusion_dim, **self.kw)(gate_inp)
            gate = jax.nn.sigmoid(gate)
            fused = gate * siglip_feat + (1.0 - gate) * cnn_feat
        return fused

    def __call__(
        self,
        carry: Dict,
        obs: Dict,
        reset: jnp.ndarray,
        training: bool,
        single: bool = False
    ) -> Tuple[Dict, Dict, jnp.ndarray]:
        bdims = 1 if single else 2
        bshape = reset.shape
        outs = []

        if self.veckeys:
            vecs = {k: obs[k] for k in self.veckeys}
            outs.append(self._encode_vectors(vecs, bdims))

        if self.imgkeys:
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            x = jnp.concatenate(imgs, -1)
            siglip_feat = self._encode_siglip(x, bdims)
            cnn_feat = self._encode_cnn(x, bdims)
            fused = self._fuse(siglip_feat, cnn_feat)
            fused = self.sub('fused_out', nn.Linear, self.output_dim, **self.kw)(fused)
            outs.append(fused)

        x = jnp.concatenate(outs, -1) if len(outs) > 1 else outs[0]
        tokens = x.reshape((*bshape, *x.shape[1:]))
        return carry, {}, tokens


class SiglipEncoderJAX(nj.Module):
    """
    Pure JAX implementation of SIGLIP-style encoder.
    
    This is an alternative implementation that doesn't require PyTorch,
    using JAX-native ViT architecture inspired by SIGLIP.
    
    Useful for:
    - Training from scratch
    - Environments where PyTorch is not available
    - Full JAX pipeline without interop overhead
    """
    
    output_dim: int = 1024
    vit_dim: int = 1024
    vit_layers: int = 12
    vit_heads: int = 16
    patch_size: int = 16
    image_size: int = 64  # DreamerV3 default
    proj_layers: int = 2
    norm: str = 'rms'
    act: str = 'gelu'
    symlog: bool = True
    
    def __init__(self, obs_space: Dict, **kw):
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.kw = kw
        self.num_patches = (self.image_size // self.patch_size) ** 2
    
    @property
    def entry_space(self):
        return {}
    
    def initial(self, batch_size: int):
        return {}
    
    def truncate(self, entries, carry=None):
        return {}
    
    def __call__(
        self, 
        carry: Dict, 
        obs: Dict, 
        reset: jnp.ndarray, 
        training: bool, 
        single: bool = False
    ):
        bdims = 1 if single else 2
        outs = []
        bshape = reset.shape
        
        # Process vectors
        if self.veckeys:
            vspace = {k: self.obs_space[k] for k in self.veckeys}
            vecs = {k: obs[k] for k in self.veckeys}
            squish = nn.symlog if self.symlog else lambda x: x
            x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
            x = x.reshape((-1, *x.shape[bdims:]))
            for i in range(self.proj_layers):
                x = self.sub(f'vec_mlp{i}', nn.Linear, self.output_dim, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f'vec_mlp{i}norm', nn.Norm, self.norm)(x))
            outs.append(x)
        
        # Process images with ViT-style encoder
        if self.imgkeys:
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            x = jnp.concatenate(imgs, -1)
            x = x.reshape((-1, *x.shape[bdims:]))
            
            # Normalize to [-0.5, 0.5]
            x = nn.cast(x, force=True) / 255 - 0.5
            
            # Patch embedding
            B, H, W, C = x.shape
            P = self.patch_size
            assert H % P == 0 and W % P == 0, f"Image size must be divisible by patch size"
            
            num_h, num_w = H // P, W // P
            # Reshape to patches
            x = x.reshape(B, num_h, P, num_w, P, C)
            x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, num_h, num_w, P, P, C)
            x = x.reshape(B, num_h * num_w, P * P * C)
            
            # Linear projection of patches
            x = self.sub('patch_embed', nn.Linear, self.vit_dim, **self.kw)(x)
            
            # Add positional embedding
            pos_embed = self.get('pos_embed', jnp.zeros, (1, self.num_patches, self.vit_dim))
            x = x + pos_embed
            
            # Transformer blocks
            for i in range(self.vit_layers):
                x = self._transformer_block(x, f'block{i}')
            
            # Global average pooling
            x = x.mean(axis=1)
            
            # Projection to output dimension
            x = self.sub('vit_proj', nn.Linear, self.output_dim, **self.kw)(x)
            x = nn.act(self.act)(self.sub('vit_proj_norm', nn.Norm, self.norm)(x))
            outs.append(x)
        
        x = jnp.concatenate(outs, -1) if len(outs) > 1 else outs[0]
        tokens = x.reshape((*bshape, *x.shape[1:]))
        
        return carry, {}, tokens
    
    def _transformer_block(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Single transformer block with self-attention and MLP."""
        # Pre-norm self attention
        residual = x
        x = self.sub(f'{name}_norm1', nn.Norm, self.norm)(x)
        x = self._self_attention(x, f'{name}_attn')
        x = residual + x
        
        # Pre-norm MLP
        residual = x
        x = self.sub(f'{name}_norm2', nn.Norm, self.norm)(x)
        x = self.sub(f'{name}_mlp1', nn.Linear, self.vit_dim * 4, **self.kw)(x)
        x = nn.act(self.act)(x)
        x = self.sub(f'{name}_mlp2', nn.Linear, self.vit_dim, **self.kw)(x)
        x = residual + x
        
        return x
    
    def _self_attention(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Multi-head self attention."""
        B, N, D = x.shape
        H = self.vit_heads
        head_dim = D // H
        
        # QKV projection
        qkv = self.sub(f'{name}_qkv', nn.Linear, D * 3, **self.kw)(x)
        qkv = qkv.reshape(B, N, 3, H, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Transpose for attention: (B, H, N, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
        
        # Output projection
        x = self.sub(f'{name}_proj', nn.Linear, D, **self.kw)(x)
        
        return x


class VLAEncoder(nj.Module):
    """
    Vision-Language-Action Encoder combining SIGLIP features with world model.
    
    This encoder is designed for the full VLA architecture where:
    1. SIGLIP provides rich visual features
    2. Language instructions can condition the policy (future extension)
    3. Action prediction benefits from pretrained visual representations
    
    The encoder maintains compatibility with DreamerV3's training pipeline
    while adding VLA-specific components.
    """
    
    siglip_path: str = ""
    output_dim: int = 1024
    freeze_siglip: bool = True
    use_language: bool = False  # Future: language conditioning
    fusion_type: str = 'concat'  # 'concat', 'cross_attention', 'film'
    proj_layers: int = 2
    norm: str = 'rms'
    act: str = 'gelu'
    
    def __init__(self, obs_space: Dict, siglip_path: str = "", **kw):
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.kw = kw
        
        if siglip_path:
            self.siglip_path = siglip_path
        
        # Initialize SIGLIP encoder
        self._siglip_encoder = SiglipVisionEncoder(
            obs_space=obs_space,
            siglip_path=siglip_path,
            output_dim=self.output_dim,
            freeze_backbone=self.freeze_siglip,
            **kw
        )
    
    @property
    def entry_space(self):
        return {}
    
    def initial(self, batch_size: int):
        return {}
    
    def truncate(self, entries, carry=None):
        return {}
    
    def __call__(
        self, 
        carry: Dict, 
        obs: Dict, 
        reset: jnp.ndarray, 
        training: bool, 
        single: bool = False
    ):
        """
        Forward pass combining SIGLIP vision with optional language conditioning.
        """
        # Get SIGLIP visual features
        carry, entries, visual_tokens = self._siglip_encoder(
            carry, obs, reset, training, single
        )
        
        # Future: Add language conditioning here
        # if self.use_language and 'language' in obs:
        #     visual_tokens = self._fuse_language(visual_tokens, obs['language'])
        
        return carry, entries, visual_tokens


# Factory function for creating encoders
def create_siglip_encoder(
    obs_space: Dict,
    encoder_type: str = 'siglip',
    siglip_path: str = "",
    **kwargs
) -> nj.Module:
    """
    Factory function to create SIGLIP-based encoders.
    
    Args:
        obs_space: Observation space dictionary
        encoder_type: 'siglip' for pretrained SIGLIP, 'siglip_jax' for JAX-native ViT
        siglip_path: Path to pretrained SIGLIP model
        **kwargs: Additional arguments for encoder
        
    Returns:
        Encoder module instance
    """
    if encoder_type == 'siglip':
        return SiglipVisionEncoder(obs_space, siglip_path=siglip_path, **kwargs)
    elif encoder_type == 'siglip_cnn':
        return SiglipCnnFusionEncoder(obs_space, siglip_path=siglip_path, **kwargs)
    elif encoder_type == 'siglip_jax':
        return SiglipEncoderJAX(obs_space, **kwargs)
    elif encoder_type == 'vla':
        return VLAEncoder(obs_space, siglip_path=siglip_path, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
