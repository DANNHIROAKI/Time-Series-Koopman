"""Time-varying Koopman with stability-constrained mixture-of-experts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


@dataclass
class KoopmanConfig:
    """Configuration for the Koopman dynamics block."""

    latent_dim: int
    context_dim: int
    num_segments: int
    steps_per_segment: List[int]
    dt: float
    num_experts: int
    low_rank: int
    horizon: int
    decay_target: float = 0.7
    decay_scale: float = 1.0
    sigma_max: float = 1.0
    max_frobenius: float = 3.0
    gate_temperature_start: float = 2.0
    gate_temperature_end: float = 0.5
    semigroup_reg_scale: float = 1.0
    stitch_reg_scale: float = 1.0
    entropy_reg_scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.steps_per_segment) != self.num_segments:
            raise ValueError(
                "steps_per_segment length must equal num_segments"
            )
        if sum(self.steps_per_segment) != self.horizon:
            raise ValueError(
                "steps_per_segment must sum to the prediction horizon"
            )
        if self.decay_target <= 0 or self.decay_target >= 1:
            raise ValueError("decay_target must lie in (0, 1)")
        if self.low_rank <= 0:
            raise ValueError("low_rank must be positive")


@dataclass
class KoopmanDiagnostics:
    norm_A: torch.Tensor
    norm_sigma: torch.Tensor
    rho_phi: torch.Tensor
    gate_entropy: torch.Tensor


def make_skew_symmetric(raw: torch.Tensor) -> torch.Tensor:
    return raw - raw.transpose(-1, -2)


def clamp_frobenius(mat: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return mat
    frob = torch.linalg.norm(mat.flatten(start_dim=-2), dim=-1)
    scale = torch.ones_like(frob)
    mask = frob > max_norm
    scale[mask] = max_norm / (frob[mask] + EPS)
    shape = scale.shape + (1, 1)
    return mat * scale.view(*shape)


def compute_mu(
    dt: float, horizon: int, decay_target: float, decay_scale: float, device: torch.device
) -> torch.Tensor:
    ratio = torch.clamp(torch.tensor(decay_target, device=device), 1e-3, 0.999)
    mu = -torch.log(ratio.to(torch.float64)) / (dt * horizon)
    mu = mu * decay_scale
    return mu.to(torch.float32)


def spectral_radius(matrix: torch.Tensor) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(matrix.to(torch.float64))
    return singular_values[..., 0].to(torch.float32)


def van_loan_discretization(
    A: torch.Tensor, Sigma: torch.Tensor, dt: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute discrete transition and covariance via Van-Loan's method."""

    d = A.shape[-1]
    A64 = A.to(torch.float64)
    Sigma64 = Sigma.to(torch.float64)
    Q = Sigma64 @ Sigma64.transpose(-1, -2)
    zeros = torch.zeros_like(Q)

    top = torch.cat([A64, Q], dim=-1)
    bottom = torch.cat([zeros, -A64.transpose(-1, -2)], dim=-1)
    M = torch.cat([top, bottom], dim=-2)

    expM = torch.linalg.matrix_exp(M * dt)
    Phi = expM[..., :d, :d]
    Q_block = expM[..., :d, d:]
    Qd = Phi @ Q_block
    Qd = 0.5 * (Qd + Qd.transpose(-1, -2))

    return Phi.to(torch.float32), Qd.to(torch.float32)


class SegmentExpertGenerator(nn.Module):
    def __init__(self, config: KoopmanConfig) -> None:
        super().__init__()
        d = config.latent_dim
        c = config.context_dim
        k = config.num_experts
        r = config.low_rank
        self.S_proj = nn.Linear(c, k * d * d)
        self.L_proj = nn.Linear(c, k * d * r)
        self.gate = nn.Linear(c, k)
        self.register_buffer("identity", torch.eye(d))

    def forward(
        self, context: torch.Tensor, mu: torch.Tensor, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        d = self.identity.shape[0]
        k = self.gate.out_features
        raw_S = self.S_proj(context).view(batch, k, d, d)
        skew = make_skew_symmetric(raw_S)
        raw_L = self.L_proj(context).view(batch, k, d, -1)
        LLT = torch.matmul(raw_L, raw_L.transpose(-1, -2))
        identity = self.identity.view(1, 1, d, d)
        A_experts = skew - LLT - mu.view(1, 1, 1, 1) * identity
        gate_logits = self.gate(context)
        weights = F.softmax(gate_logits / max(temperature, EPS), dim=-1)
        entropy = -(weights * torch.log(weights + EPS)).sum(dim=-1)
        A = torch.sum(weights[..., None, None] * A_experts, dim=1)
        return A, weights, entropy


class SigmaGenerator(nn.Module):
    def __init__(self, config: KoopmanConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.context_dim, config.latent_dim)
        self.sigma_max = config.sigma_max

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        diag = F.softplus(self.linear(context))
        diag = torch.clamp(diag, max=self.sigma_max)
        return torch.diag_embed(diag)


class TVKoopmanMoE(nn.Module):
    def __init__(self, config: KoopmanConfig) -> None:
        super().__init__()
        self.config = config
        self.segments = nn.ModuleList(
            [SegmentExpertGenerator(config) for _ in range(config.num_segments)]
        )
        self.sigma_generators = nn.ModuleList(
            [SigmaGenerator(config) for _ in range(config.num_segments)]
        )
        self.temperature = config.gate_temperature_start

    def set_stage(self, stage: int, progress: float = 0.0) -> None:
        if stage == 1:
            self.temperature = self.config.gate_temperature_start
        else:
            start = self.config.gate_temperature_start
            end = self.config.gate_temperature_end
            self.temperature = max(end, start + (end - start) * progress)

    def forward(
        self,
        context: torch.Tensor,
        z0: torch.Tensor,
        return_covariance: bool = False,
        collect_diagnostics: bool = False,
        collect_regularizers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        device = context.device
        mu = compute_mu(cfg.dt, cfg.horizon, cfg.decay_target, cfg.decay_scale, device)
        z = z0
        states: List[torch.Tensor] = []
        covariances: List[torch.Tensor] = []
        if return_covariance:
            cov = torch.zeros(z.shape[0], cfg.latent_dim, cfg.latent_dim, device=device)
        diag_norm_A: List[torch.Tensor] = []
        diag_norm_sigma: List[torch.Tensor] = []
        diag_rho: List[torch.Tensor] = []
        diag_entropy: List[torch.Tensor] = []
        diag_gate_weights: List[torch.Tensor] = []
        A_mats: List[torch.Tensor] = []
        Phi_mats: List[torch.Tensor] = []

        for segment, sigma_gen, steps in zip(
            self.segments, self.sigma_generators, cfg.steps_per_segment
        ):
            A, weights, entropy = segment(context, mu, self.temperature)
            A = clamp_frobenius(A, cfg.max_frobenius)
            Sigma = sigma_gen(context)
            Phi, Qd = van_loan_discretization(A, Sigma, cfg.dt)

            if collect_diagnostics or collect_regularizers:
                A_mats.append(A)
                Phi_mats.append(Phi)
            if collect_diagnostics:
                norm_A = torch.linalg.norm(A.flatten(start_dim=-2), dim=-1)
                norm_sigma = torch.linalg.norm(
                    Sigma.flatten(start_dim=-2), dim=-1
                )
                rho = spectral_radius(Phi)
                diag_norm_A.append(norm_A)
                diag_norm_sigma.append(norm_sigma)
                diag_rho.append(rho)
                diag_gate_weights.append(weights)
            if collect_diagnostics or (
                collect_regularizers and cfg.entropy_reg_scale > 0
            ):
                diag_entropy.append(entropy)

            for _ in range(steps):
                z = torch.matmul(Phi, z.unsqueeze(-1)).squeeze(-1)
                states.append(z)
                if return_covariance:
                    cov = Phi @ cov @ Phi.transpose(-1, -2) + Qd
                    cov = 0.5 * (cov + cov.transpose(-1, -2))
                    covariances.append(cov)
        if states:
            states_tensor = torch.stack(states, dim=1)
        else:
            states_tensor = torch.empty(
                z.shape[0], 0, cfg.latent_dim, device=device
            )
        output: Dict[str, torch.Tensor] = {"states": states_tensor}
        if return_covariance and covariances:
            output["covariances"] = torch.stack(covariances, dim=1)

        if collect_diagnostics:
            diagnostics: Dict[str, torch.Tensor] = {}
            if diag_norm_A:
                diagnostics["norm_A"] = torch.stack(diag_norm_A, dim=1).detach()
            if diag_norm_sigma:
                diagnostics["norm_sigma"] = torch.stack(
                    diag_norm_sigma, dim=1
                ).detach()
            if diag_rho:
                diagnostics["rho_phi"] = torch.stack(diag_rho, dim=1).detach()
            if diag_entropy:
                diagnostics["gate_entropy"] = torch.stack(
                    diag_entropy, dim=1
                ).detach()
            if diag_gate_weights:
                diagnostics["gate_weights"] = torch.stack(
                    diag_gate_weights, dim=1
                ).detach()
            output["diagnostics"] = diagnostics

        if collect_regularizers:
            regularizers: Dict[str, torch.Tensor] = {}
            if cfg.semigroup_reg_scale > 0:
                semigroup_losses = []
                for A in A_mats:
                    Phi_dt = torch.linalg.matrix_exp((A * cfg.dt).to(torch.float64))
                    Phi_dt = Phi_dt.to(torch.float32)
                    Phi_2dt = torch.linalg.matrix_exp((A * (2 * cfg.dt)).to(torch.float64))
                    Phi_2dt = Phi_2dt.to(torch.float32)
                    diff = Phi_dt @ Phi_dt - Phi_2dt
                    semigroup_losses.append(diff.pow(2).mean(dim=(-2, -1)))
                if semigroup_losses:
                    regularizers["semigroup"] = (
                        torch.stack(semigroup_losses, dim=1).mean()
                        * cfg.semigroup_reg_scale
                    )
            if cfg.stitch_reg_scale > 0 and len(Phi_mats) > 1:
                stitch_losses = []
                for first, second in zip(Phi_mats[:-1], Phi_mats[1:]):
                    diff = second - first
                    stitch_losses.append(diff.pow(2).mean(dim=(-2, -1)))
                regularizers["stitch"] = (
                    torch.stack(stitch_losses, dim=1).mean() * cfg.stitch_reg_scale
                )
            if cfg.entropy_reg_scale > 0 and diag_entropy:
                ent = torch.stack(diag_entropy, dim=1).mean()
                regularizers["entropy"] = ent * cfg.entropy_reg_scale
            if regularizers:
                output["regularizers"] = regularizers

        return output


class KoopmanReadout(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    @property
    def kernel_size(self) -> int:
        return 1

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.linear(states)

    def covariance(self, covariances: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight  # (output_dim, latent_dim)
        cov = torch.einsum("...ij,aj->...ai", covariances, weight)
        cov = torch.einsum("...ai,bi->...ab", cov, weight)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        return cov
