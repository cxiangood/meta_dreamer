"""
SIGReg: Sliced Isotropic Gaussian Regularization (from LeWM, Maes et al. 2026)

Encourages embeddings to match an isotropic Gaussian distribution N(0, I) via:
1. Random projection onto M unit directions
2. Epps-Pulley normality test statistic on each projection
3. By Cramer-Wold theorem, matching all 1D marginals ≡ matching joint distribution

Only hyperparameter: lambda (weight, default 0.1, robust in [0.01, 0.2])
"""

import torch
import torch.nn.functional as F
import math


class SIGReg(torch.nn.Module):
    def __init__(self, embed_dim, num_projections=1024, num_integration_nodes=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_projections = num_projections  # M
        self.num_nodes = num_integration_nodes  # Q (integration quadrature nodes)

        # Fixed random projection directions (not learned)
        projections = torch.randn(num_projections, embed_dim)
        projections = F.normalize(projections, dim=1)  # unit vectors on S^{d-1}
        self.register_buffer("projections", projections)

        # Integration nodes for Epps-Pulley: points t where we evaluate CF difference
        # Use Gauss-Hermite-like nodes centered at origin
        nodes = torch.linspace(-4.0, 4.0, num_integration_nodes)
        self.register_buffer("nodes", nodes)

    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch, embed_dim) or (history, batch, embed_dim)
        Returns:
            loss: scalar SIGReg loss
        """
        if embeddings.dim() == 3:
            # (history, batch, embed_dim) -> (history * batch, embed_dim)
            H, B, D = embeddings.shape
            embeddings = embeddings.reshape(H * B, D)
        else:
            B, D = embeddings.shape

        # Project onto random directions: (M, D) @ (D, B) -> (M, B)
        projected = embeddings @ self.projections.T  # (B, M)

        # Epps-Pulley test statistic for each projection direction
        loss = self._epps_pulley(projected)  # scalar

        return loss

    def _epps_pulley(self, projected):
        """
        Epps-Pulley test: measure divergence between empirical CF and standard normal CF.

        Args:
            projected: (B, M) - B samples projected onto M directions
        Returns:
            loss: scalar
        """
        B, M = projected.shape

        # Standardize each projection direction
        mean = projected.mean(dim=0, keepdim=True)  # (1, M)
        std = projected.std(dim=0, keepdim=True).clamp(min=1e-6)  # (1, M)
        standardized = (projected - mean) / std  # (B, M)

        # Integration nodes: (Q,)
        t = self.nodes.unsqueeze(0).unsqueeze(0)  # (1, 1, Q)

        # Empirical characteristic function: E[exp(itX)]
        # standardized: (B, M) -> (B, M, 1)
        x = standardized.unsqueeze(-1)  # (B, M, 1)
        # ECF: (1/B) * sum exp(i*t*x) for each (m, q)
        ecf_real = torch.cos(t * x).mean(dim=0)  # (M, Q)
        ecf_imag = torch.sin(t * x).mean(dim=0)  # (M, Q)

        # Standard normal CF: exp(-t^2/2)
        normal_cf = torch.exp(-t.squeeze(0).squeeze(0) ** 2 / 2)  # (Q,)

        # Weight function: w(t) = exp(-t^2 / (2*lambda^2))
        # Using lambda=1.0 (same as LeWM default)
        weight = torch.exp(-t.squeeze(0).squeeze(0) ** 2 / 2)  # (Q,)

        # |ECF(t) - Normal_CF(t)|^2 weighted integral
        diff_real = ecf_real - normal_cf.unsqueeze(0)  # (M, Q)
        diff_imag = ecf_imag  # standard normal has zero imaginary part

        # Weighted squared difference: w(t) * |diff|^2
        weighted_diff2 = weight.unsqueeze(0) * (diff_real ** 2 + diff_imag ** 2)  # (M, Q)

        # Sum over integration nodes, average over projection directions
        loss = weighted_diff2.sum(dim=1).mean()  # scalar

        return loss
