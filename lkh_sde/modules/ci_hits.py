"""Implementation of the CI-HiTS multi-scale near-field head."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        padding = dilation
        self.dwconv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.glu = nn.GLU(dim=1)
        self.pwconv = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dwconv(x)
        out = self.glu(out)
        out = self.pwconv(out)
        out = self.activation(out)
        out = self.dropout(out)
        return residual + self.res_scale * out


class MultiScaleLevel(nn.Module):
    def __init__(
        self,
        channels: int,
        input_dim: int,
        dilation: int,
        horizon: int,
        reduction: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = DepthwiseDilatedBlock(channels, dilation, dropout)
        target_length = max(1, horizon // reduction)
        self.pool = nn.AdaptiveAvgPool1d(target_length)
        self.head = nn.Conv1d(channels, input_dim, kernel_size=1)
        self.reduction = reduction
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        pooled = self.pool(x)
        prediction = self.head(pooled)
        if prediction.size(-1) != self.horizon:
            prediction = F.interpolate(
                prediction,
                size=self.horizon,
                mode="linear",
                align_corners=False,
            )
        return x, prediction


class CIHiTS(nn.Module):
    """Channel-independent multi-scale temporal convolution head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        horizon: int,
        dilations: Sequence[int],
        dropout: float = 0.0,
        future_dim: Optional[int] = None,
        reductions: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        if reductions is None:
            reductions = [max(1, 2 ** i) for i in range(num_layers)]
        levels: List[MultiScaleLevel] = []
        for i in range(num_layers):
            dilation = dilations[i % len(dilations)]
            reduction = reductions[i % len(reductions)]
            levels.append(
                MultiScaleLevel(
                    hidden_dim,
                    input_dim,
                    dilation=dilation,
                    horizon=horizon,
                    reduction=reduction,
                    dropout=dropout,
                )
            )
        self.levels = nn.ModuleList(levels)
        self.level_weights = nn.Parameter(torch.ones(len(levels)))
        self.horizon = horizon
        self.input_dim = input_dim
        self.future_dim = future_dim or 0
        if self.future_dim > 0:
            self.future_proj = nn.Linear(self.future_dim, input_dim)
        else:
            self.future_proj = None

    def forward(
        self,
        x: torch.Tensor,
        future_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, channels, length)
        x = self.input_proj(x)
        predictions = []
        for weight, level in zip(self.level_weights, self.levels):
            x, pred_level = level(x)
            predictions.append(weight * pred_level)
        out = torch.stack(predictions, dim=0).sum(dim=0)
        out = out.transpose(1, 2)  # (batch, horizon, channels)
        if future_features is not None and future_features.numel() > 0:
            if self.future_proj is None or future_features.size(-1) != self.future_dim:
                self.future_dim = future_features.size(-1)
                self.future_proj = nn.Linear(self.future_dim, self.input_dim).to(
                    future_features.device
                )
            projected = self.future_proj(future_features)
            out = out + projected
        return out
