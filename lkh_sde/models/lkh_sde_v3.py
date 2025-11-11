"""LKH-SDE v3 composite model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from lkh_sde.modules.ci_hits import CIHiTS
from lkh_sde.modules.encoder import HistoryEncoder
from lkh_sde.modules.fusion import FusionModule
from lkh_sde.modules.koopman import KoopmanConfig, KoopmanReadout, TVKoopmanMoE


@dataclass
class CIConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    dilations: tuple = (1, 2, 4, 8)
    dropout: float = 0.05
    reductions: Optional[tuple] = None


@dataclass
class FusionConfig:
    max_alpha: float = 0.5
    gamma: float = 2.0
    gate_bias: float = -5.0


@dataclass
class ModelConfig:
    input_dim: int
    output_dim: int
    known_future_dim: int = 0
    context_dim: int = 64
    latent_dim: int = 16
    encoder_hidden: int = 128
    encoder_layers: int = 1
    ci: CIConfig = field(default_factory=CIConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    koopman: KoopmanConfig = field(
        default_factory=lambda: KoopmanConfig(
            latent_dim=16,
            context_dim=64,
            num_segments=4,
            steps_per_segment=[24, 24, 24, 24],
            dt=1.0,
            num_experts=2,
            low_rank=4,
            horizon=96,
        )
    )


class LKHSDEV3(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = HistoryEncoder(
            input_dim=config.input_dim,
            context_dim=config.context_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.encoder_hidden,
            num_layers=config.encoder_layers,
        )
        self.ci_head = CIHiTS(
            input_dim=config.input_dim,
            hidden_dim=config.ci.hidden_dim,
            num_layers=config.ci.num_layers,
            horizon=config.koopman.horizon,
            dilations=list(config.ci.dilations),
            dropout=config.ci.dropout,
            future_dim=config.known_future_dim,
            reductions=config.ci.reductions,
        )
        self.koopman = TVKoopmanMoE(config.koopman)
        self.readout = KoopmanReadout(config.latent_dim, config.output_dim)
        self.fusion = FusionModule(
            context_dim=config.context_dim,
            max_alpha=config.fusion.max_alpha,
            horizon=config.koopman.horizon,
            gamma=config.fusion.gamma,
            gate_bias_init=config.fusion.gate_bias,
        )
        self.stage = 1
        self.set_stage(1, 0.0)

    def set_stage(self, stage: int, progress: float = 0.0) -> None:
        self.stage = stage
        requires_grad = stage != 1
        for module in [self.koopman, self.readout, self.fusion]:
            for param in module.parameters():
                param.requires_grad = requires_grad
        self.fusion.set_stage(stage, progress)
        self.koopman.set_stage(stage, progress)

    def forward(
        self,
        history: torch.Tensor,
        future_features: Optional[torch.Tensor] = None,
        return_covariance: bool = False,
        return_diagnostics: bool = False,
        collect_regularizers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encoder(history)
        ci_input = enc.normalized_history
        ci_out = self.ci_head(ci_input, future_features)
        koopman_kwargs = {
            "return_covariance": return_covariance,
            "collect_diagnostics": return_diagnostics,
            "collect_regularizers": collect_regularizers,
        }
        koopman_out = self.koopman(enc.context, enc.latent_init, **koopman_kwargs)
        koopman_states = koopman_out["states"]
        koopman_mean = self.readout(koopman_states)
        fused, gate, weights, fusion_scale = self.fusion(
            enc.context, ci_out, koopman_mean, return_details=True
        )
        output: Dict[str, torch.Tensor] = {
            "mean": fused,
            "ci_mean": ci_out,
            "koopman_mean": koopman_mean,
            "koopman_states": koopman_states,
            "fusion_gate": gate,
            "fusion_weights": weights,
            "fusion_scale": fusion_scale,
            "revin_state": enc.revin_state,
        }
        if return_covariance and "covariances" in koopman_out:
            koop_cov = self.readout.covariance(koopman_out["covariances"])
            scaled_cov = fusion_scale.unsqueeze(-1) * fusion_scale.unsqueeze(-2) * koop_cov
            scaled_cov = 0.5 * (scaled_cov + scaled_cov.transpose(-1, -2))
            output["covariance"] = scaled_cov
        if return_diagnostics and "diagnostics" in koopman_out:
            diagnostics = dict(koopman_out["diagnostics"])
            diagnostics["alpha"] = gate.detach()
            diagnostics["rho_scale"] = weights.detach()
            output["diagnostics"] = diagnostics
        if collect_regularizers and "regularizers" in koopman_out:
            output["regularizers"] = koopman_out["regularizers"]
        return output
