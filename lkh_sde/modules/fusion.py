"""Fusion module combining CI-HiTS and Koopman branches."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    def __init__(
        self,
        context_dim: int,
        max_alpha: float,
        horizon: int,
        gamma: float,
        gate_bias_init: float = -5.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 1),
        )
        self.register_buffer("gate_bias", torch.tensor(gate_bias_init))
        self.max_alpha = max_alpha
        self.horizon = horizon
        self.gamma = gamma
        self.stage = 1
        self.alpha_multiplier = 0.0

    def set_stage(self, stage: int, progress: float = 0.0) -> None:
        self.stage = stage
        if stage == 1:
            self.alpha_multiplier = 0.0
        else:
            progress = float(max(0.0, min(1.0, progress)))
            self.alpha_multiplier = self.max_alpha * progress

    def compute_gate(self, context: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(context) + self.gate_bias
        gate = torch.sigmoid(logits)
        return gate * self.alpha_multiplier

    def fusion_weights(self, device: torch.device) -> torch.Tensor:
        if self.horizon <= 1:
            return torch.ones(1, self.horizon, 1, device=device)
        steps = torch.linspace(0, 1, steps=self.horizon, device=device)
        weights = steps.pow(self.gamma)
        weights = weights.view(1, self.horizon, 1)
        return weights

    def forward(
        self,
        context: torch.Tensor,
        ci_out: torch.Tensor,
        koopman_out: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = self.compute_gate(context)
        weights = self.fusion_weights(ci_out.device)
        fusion_scale = gate.unsqueeze(-1) * weights
        fused = ci_out + fusion_scale * koopman_out
        if return_details:
            return fused, gate, weights.squeeze(-1), fusion_scale
        return fused, gate, weights.squeeze(-1), fusion_scale
