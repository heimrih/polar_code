"""Learnable symmetric β metric for DL-SCL."""

from __future__ import annotations

import torch
from torch import nn


class SymmetricBeta(nn.Module):
    """Symmetric correlation matrix with unit diagonal."""

    def __init__(self, dim: int, init_range: float = 0.2) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        super().__init__()
        self.dim = dim
        self.init_range = float(init_range)

        off_diag = torch.empty(dim, dim)
        nn.init.uniform_(off_diag, -self.init_range, self.init_range)
        off_diag.fill_diagonal_(0.0)
        self.off_diag = nn.Parameter(off_diag)

    def clamp_diagonal(self) -> None:
        """Force diagonal of the learnable matrix to zero (unit diag post-build)."""

        with torch.no_grad():
            self.off_diag.fill_diagonal_(0.0)

    def beta_matrix(self) -> torch.Tensor:
        """Return the symmetric β matrix with unit diagonal."""

        upper = torch.triu(self.off_diag, diagonal=1)
        symmetric = upper + upper.transpose(0, 1)
        diag = torch.ones(self.dim, device=symmetric.device, dtype=symmetric.dtype)
        return symmetric + torch.diag(diag)

    def forward(self, abs_l0: torch.Tensor) -> torch.Tensor:
        """Compute Q = |L0| @ β for 1D or 2D inputs."""

        beta = self.beta_matrix()
        if abs_l0.dim() == 1:
            return abs_l0 @ beta
        if abs_l0.dim() == 2:
            return abs_l0 @ beta
        raise ValueError("abs_l0 must be 1D or 2D tensor")


__all__ = ["SymmetricBeta"]
