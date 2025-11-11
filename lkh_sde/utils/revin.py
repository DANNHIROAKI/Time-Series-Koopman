"""Reversible Instance Normalization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class RevINState:
    """Container for storing normalization statistics.

    Attributes
    ----------
    mean: torch.Tensor
        Per-channel mean with shape ``(batch, 1, channels)`` or ``(batch, channels)``.
    std: torch.Tensor
        Per-channel standard deviation with same shape as ``mean``.
    """

    mean: torch.Tensor
    std: torch.Tensor


class RevIN(nn.Module):
    """Implements Reversible Instance Normalization.

    Parameters
    ----------
    num_channels:
        Number of channels to normalize.
    eps:
        Numerical stability constant.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = nn.Parameter(torch.ones(1, 1, num_channels))
        self.bias = nn.Parameter(torch.zeros(1, 1, num_channels))

    def _compute_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        return mean, std

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, RevINState]:
        mean, std = self._compute_stats(x)
        x_norm = (x - mean) / std
        x_norm = x_norm * self.affine + self.bias
        return x_norm, RevINState(mean=mean, std=std)

    def denormalize(self, x: torch.Tensor, state: RevINState) -> torch.Tensor:
        x = (x - self.bias) / (self.affine + self.eps)
        return x * state.std + state.mean

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "norm",
        state: Optional[RevINState] = None,
    ) -> Tuple[torch.Tensor, RevINState]:
        """Apply normalization or denormalization.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, length, channels)``.
        mode:
            Either ``"norm"`` or ``"denorm"``.
        state:
            Optional cached statistics for denormalization.
        """

        if mode == "norm":
            return self.normalize(x)
        if mode == "denorm":
            if state is None:
                raise ValueError("state must be provided when mode='denorm'")
            return self.denormalize(x), state
        raise ValueError(f"Unknown mode: {mode}")
