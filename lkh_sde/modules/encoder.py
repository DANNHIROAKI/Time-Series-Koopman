"""History encoder with RevIN normalization."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from lkh_sde.utils.revin import RevIN, RevINState


@dataclass
class EncoderOutput:
    context: torch.Tensor
    latent_init: torch.Tensor
    revin_state: RevINState
    normalized_history: torch.Tensor


class HistoryEncoder(nn.Module):
    """Encodes historical windows into context and latent states."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.revin = RevIN(num_channels=input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.context_proj = nn.Linear(hidden_dim, context_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        x_norm, state = self.revin(x, mode="norm")
        _, h_n = self.gru(x_norm)
        h_last = h_n[-1]
        context = self.context_proj(h_last)
        latent = self.latent_proj(h_last)
        return EncoderOutput(
            context=context,
            latent_init=latent,
            revin_state=state,
            normalized_history=x_norm,
        )

    def denormalize(self, x: torch.Tensor, state: RevINState) -> torch.Tensor:
        denorm, _ = self.revin(x, mode="denorm", state=state)
        return denorm
