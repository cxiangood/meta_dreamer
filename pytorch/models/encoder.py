"""
BEV Encoder: CNN that encodes 3-channel RGB BEV images into compact embeddings.

Architecture follows DreamerV3's SimpleEncoder adapted for BEV input:
- 4-layer strided CNN
- Symlog squash for stable training
- Output: deterministic embedding for RSSM posterior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Symlog(nn.Module):
    """Symlog squashing function from DreamerV3 for stable gradient flow."""
    def forward(self, x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def inverse(self, x):
        return torch.sign(x) * torch.expm1(torch.abs(x))


class BEVEncoder(nn.Module):
    """
    Encodes BEV image (3-ch or N-ch) into a compact feature vector.

    Architecture:
    - 4 conv layers with increasing depth [128, 192, 256, 256] (depth=64 base)
    - Each: Conv5x5, stride=2, BatchNorm, SiLU
    - AdaptiveAvgPool2d → fixed spatial dim (avoids Linear explosion on large inputs)
    - Flatten + Linear to embed_dim
    - Symlog squash before convolutions
    """

    def __init__(self, input_channels=3, input_size=256, embed_dim=512,
                 depth=64, act=nn.SiLU, norm=True, pool_size=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.symlog = Symlog()

        mults = [2, 3, 4, 4]  # DreamerV3 SimpleEncoder multipliers
        channels = [depth * m for m in mults]  # [128, 192, 256, 256]
        kernel_size = 5

        layers = []
        in_ch = input_channels
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2))
            if norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act())
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Compute flattened size after convs + pooling
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            dummy_out = self.pool(self.conv(dummy))
            flat_size = dummy_out.numel() // dummy_out.shape[0]

        self.head = nn.Linear(flat_size, embed_dim)

    def forward(self, bev_map):
        """
        Args:
            bev_map: (batch, 6, H, W) uint8 or float BEV semantic tensor
        Returns:
            embed: (batch, embed_dim)
        """
        x = bev_map.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = self.symlog(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.head(x)
        return x


class CNNFrontend(nn.Module):
    """Learnable CNN downsampling frontend.

    Replaces bilinear resize with strided convolutions that learn to preserve
    task-relevant information (vehicles, lane markings, merge gaps).

    2x mode: 300² → Conv(s=2) → 150² → Conv(s=1) → 150² → encoder
    4x mode: 300² → Conv(s=2) → 150² → Conv(s=2) → 75² → encoder

    The output feature map feeds directly into BEVEncoder, which sees it
    as a learned "pseudo-image" with out_channels channels.
    """

    def __init__(self, in_channels=3, out_channels=64, factor=2, act=nn.ReLU):
        super().__init__()
        self.factor = factor
        self.out_channels = out_channels

        if factor == 2:
            # 300 → 150 (one stride-2), refine at 150
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
                act(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                act(),
                nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            )
        elif factor == 4:
            # 300 → 150 → 75 (two stride-2), refine at 75
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
                act(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                act(),
                nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            )
        else:
            raise ValueError(f"factor must be 2 or 4, got {factor}")

        self._output_size = 300 // factor

    def forward(self, x):
        """x: (B, 3, H, W) raw BEV float [0,1], H=W=300 after center-crop"""
        return self.net(x)

    @property
    def output_size(self):
        return self._output_size


class PixelUnshuffleFrontend(nn.Module):
    """Lossless spatial downsampling via pixel unshuffle (space-to-depth).

    Rearranges spatial pixels into channel dimension — zero information loss,
    unlike strided convolution which discards 75% of pixel values.

    300²×3 → pixel_unshuffle(factor=2) → 150²×12 → conv refine → 150²×64

    Compares against CNNFrontend (strided conv) to measure the value of
    lossless downsampling for driving scene representation.
    """

    def __init__(self, in_channels=3, out_channels=64, factor=2, act=nn.ReLU):
        super().__init__()
        self.factor = factor
        self.out_channels = out_channels

        in_c = in_channels * (factor ** 2)  # 3→12 for factor=2

        # Light refinement after pixel unshuffle
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_channels, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            act(),
        )

        self._output_size = 300 // factor

    def forward(self, x):
        """x: (B, 3, H, W) BEV float [0,1], H=W=300"""
        x = F.pixel_unshuffle(x, downscale_factor=self.factor)
        return self.net(x)

    @property
    def output_size(self):
        return self._output_size


class BEVDecoder(nn.Module):
    """
    Decodes embeddings back to BEV semantic map (for reconstruction loss / visualization).
    """

    def __init__(self, output_channels=3, output_size=256, embed_dim=512,
                 depth=64, act=nn.SiLU, norm=True):
        super().__init__()
        self.output_channels = output_channels
        self.output_size = output_size
        self.symlog = Symlog()

        channels = [depth * m for m in [4, 4, 3, 2]]  # [256, 256, 192, 128]
        kernel_size = 5

        spatial = output_size // (2 ** 4)
        self.spatial = spatial

        self.head = nn.Linear(embed_dim, channels[0] * spatial * spatial)

        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(
                channels[i], channels[i + 1], kernel_size,
                stride=2, padding=kernel_size // 2, output_padding=1
            ))
            if norm:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(act())

        self.deconv = nn.Sequential(*layers)

        # Final conv to get output channels, no activation (use symlog inverse)
        self.final_conv = nn.Conv2d(channels[-1], output_channels, 1)

    def forward(self, embed):
        """
        Args:
            embed: (batch, embed_dim)
        Returns:
            recon: (batch, 6, 256, 256) reconstructed BEV map (in symlog space)
        """
        x = self.head(embed)
        x = x.reshape(x.shape[0], -1, self.spatial, self.spatial)
        x = self.deconv(x)

        # Crop or pad to exact output size
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)

        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    # Quick test
    enc = BEVEncoder(input_channels=6, embed_dim=512)
    dec = BEVDecoder(output_channels=3, embed_dim=512)
    x = torch.randn(4, 6, 256, 256)
    z = enc(x)
    print(f"Encoder: {x.shape} -> {z.shape}")
    recon = dec(z)
    print(f"Decoder: {z.shape} -> {recon.shape}")
    total_params = sum(p.numel() for p in list(enc.parameters()) + list(dec.parameters()))
    print(f"Total params: {total_params / 1e6:.2f}M")
