"""Training utilities for LKH-SDE v3."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lkh_sde.models.lkh_sde_v3 import LKHSDEV3
from lkh_sde.utils.progress import progress


def _frequency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    topk: int,
    phase_weight: float,
) -> torch.Tensor:
    if topk <= 0:
        return prediction.new_tensor(0.0)
    freq_pred = torch.fft.rfft(prediction, dim=1)
    freq_target = torch.fft.rfft(target, dim=1)
    max_bins = freq_pred.size(1)
    k = min(topk, max_bins)
    freq_pred = freq_pred[:, :k]
    freq_target = freq_target[:, :k]
    amp_diff = (freq_pred.abs() - freq_target.abs()).pow(2)
    phase_diff = torch.angle(freq_pred) - torch.angle(freq_target)
    phase_term = 1.0 - torch.cos(phase_diff)
    return amp_diff.mean() + phase_weight * phase_term.mean()


def _gaussian_nll(
    prediction: torch.Tensor,
    covariance: torch.Tensor,
    target: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    batch, horizon, dims = prediction.shape
    eye = torch.eye(dims, device=prediction.device).view(1, 1, dims, dims)
    cov = covariance + jitter * eye
    chol = torch.linalg.cholesky(cov)
    diff = target - prediction
    diff_vec = diff.unsqueeze(-1)
    solve = torch.cholesky_solve(diff_vec, chol)
    mahal = torch.matmul(diff_vec.transpose(-1, -2), solve).squeeze(-1).squeeze(-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-12).sum(dim=-1)
    nll = 0.5 * (mahal + logdet + dims * math.log(2 * math.pi))
    return nll.mean()


def _gaussian_crps(
    prediction: torch.Tensor,
    covariance: torch.Tensor,
    target: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    var = torch.diagonal(covariance, dim1=-2, dim2=-1) + jitter
    sigma = torch.sqrt(torch.clamp(var, min=jitter))
    diff = target - prediction
    z = diff / sigma
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * z.pow(2))
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return crps.mean()


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    stage1_epochs: int = 10
    stage2_warmup_epochs: int = 5
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ema: float = 0.0
    log_interval: int = 100
    checkpoint_dir: Optional[str] = None
    freq_reg_weight: float = 0.0
    freq_reg_topk: int = 0
    freq_phase_weight: float = 0.0
    semigroup_weight: float = 0.0
    stitch_weight: float = 0.0
    entropy_weight: float = 0.0
    log_diagnostics: bool = False
    diagnostics_path: Optional[str] = None
    nll_weight: float = 0.0
    crps_weight: float = 0.0
    covariance_jitter: float = 1e-5


class Trainer:
    def __init__(
        self,
        model: LKHSDEV3,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        optim_config: OptimConfig,
    ) -> None:
        self.model = model.to(optim_config.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optim_config = optim_config
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=optim_config.lr,
            weight_decay=optim_config.weight_decay,
        )
        self.ema_decay = optim_config.ema
        self.ema_shadow: Optional[Dict[str, torch.Tensor]] = None
        self._reset_ema(reset=True)
        self.mse = nn.MSELoss()
        self.best_val = math.inf
        self.checkpoint_dir = (
            Path(optim_config.checkpoint_dir) if optim_config.checkpoint_dir else None
        )
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.diag_path = (
            Path(optim_config.diagnostics_path)
            if optim_config.diagnostics_path is not None
            else None
        )
        if self.diag_path:
            self.diag_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_covariance = (
            optim_config.nll_weight > 0.0 or optim_config.crps_weight > 0.0
        )
        self.collect_regularizers = any(
            weight > 0.0
            for weight in (
                optim_config.semigroup_weight,
                optim_config.stitch_weight,
                optim_config.entropy_weight,
            )
        )

    def _reset_ema(self, reset: bool = True) -> None:
        if self.ema_decay <= 0:
            self.ema_shadow = None
            return
        params = {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        if reset or self.ema_shadow is None:
            self.ema_shadow = {
                name: param.data.clone() for name, param in params.items()
            }
        else:
            updated = {}
            for name, param in params.items():
                if name in self.ema_shadow:
                    updated[name] = self.ema_shadow[name]
                else:
                    updated[name] = param.data.clone()
            self.ema_shadow = updated

    def _update_optimizer_parameters(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.optim_config.lr,
            weight_decay=self.optim_config.weight_decay,
        )

    def _apply_ema(self) -> None:
        if self.ema_decay <= 0 or self.ema_shadow is None:
            return
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.ema_shadow[name].mul_(self.ema_decay).add_(
                param.data * (1 - self.ema_decay)
            )

    def _swap_ema_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.ema_decay <= 0 or self.ema_shadow is None:
            return None
        backup: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            backup[name] = param.data.clone()
            param.data.copy_(self.ema_shadow[name])
        return backup

    def _log_epoch(
        self,
        epoch: int,
        stage: int,
        train_loss: float,
        val_loss: Optional[float],
        aux_losses: Dict[str, float],
        diagnostics: Dict[str, float],
    ) -> None:
        record = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": train_loss,
        }
        if val_loss is not None:
            record["val_loss"] = val_loss
        for key, value in aux_losses.items():
            record[f"loss/{key}"] = value
        for key, value in diagnostics.items():
            record[f"diag/{key}"] = value
        line = json.dumps(record)
        print(line)
        if self.diag_path:
            with self.diag_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        self.model.train()
        device = self.optim_config.device
        total_loss = 0.0
        total_steps = 0
        freq_total = 0.0
        nll_total = 0.0
        crps_total = 0.0
        reg_totals: Dict[str, float] = defaultdict(float)
        diag_sums: Dict[str, float] = defaultdict(float)
        diag_count = 0

        for step, batch in enumerate(progress(self.train_loader, description=f"Epoch {epoch}")):
            history, future, future_feats = batch
            history = history.to(device)
            future = future.to(device)
            future_feats = future_feats.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(
                history,
                future_feats,
                return_covariance=self.use_covariance,
                return_diagnostics=self.optim_config.log_diagnostics,
                collect_regularizers=self.collect_regularizers,
            )
            pred_norm = outputs["mean"]
            state = outputs["revin_state"]
            pred = self.model.encoder.denormalize(pred_norm, state)
            loss = self.mse(pred, future)

            if self.optim_config.freq_reg_weight > 0 and self.optim_config.freq_reg_topk > 0:
                freq_loss = _frequency_loss(
                    pred,
                    future,
                    self.optim_config.freq_reg_topk,
                    self.optim_config.freq_phase_weight,
                )
                loss = loss + self.optim_config.freq_reg_weight * freq_loss
                freq_total += freq_loss.item()
            if self.collect_regularizers:
                regularizers = outputs.get("regularizers", {})
                if "semigroup" in regularizers:
                    reg_val = regularizers["semigroup"]
                    loss = loss + self.optim_config.semigroup_weight * reg_val
                    reg_totals["semigroup"] += reg_val.item()
                if "stitch" in regularizers:
                    reg_val = regularizers["stitch"]
                    loss = loss + self.optim_config.stitch_weight * reg_val
                    reg_totals["stitch"] += reg_val.item()
                if "entropy" in regularizers:
                    reg_val = regularizers["entropy"]
                    loss = loss + self.optim_config.entropy_weight * reg_val
                    reg_totals["entropy"] += reg_val.item()
            if self.use_covariance and "covariance" in outputs:
                cov = outputs["covariance"]
                if self.optim_config.nll_weight > 0:
                    nll = _gaussian_nll(
                        pred,
                        cov,
                        future,
                        self.optim_config.covariance_jitter,
                    )
                    loss = loss + self.optim_config.nll_weight * nll
                    nll_total += nll.item()
                if self.optim_config.crps_weight > 0:
                    crps = _gaussian_crps(
                        pred,
                        cov,
                        future,
                        self.optim_config.covariance_jitter,
                    )
                    loss = loss + self.optim_config.crps_weight * crps
                    crps_total += crps.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.optim_config.gradient_clip,
            )
            self.optimizer.step()
            self._apply_ema()

            total_loss += loss.item()
            total_steps += 1

            if self.optim_config.log_diagnostics and "diagnostics" in outputs:
                diag = outputs["diagnostics"]
                for key, value in diag.items():
                    diag_sums[key] += value.mean().item()
                diag_count += 1

            if (step + 1) % self.optim_config.log_interval == 0:
                print(f"Epoch {epoch} Step {step+1}: loss={loss.item():.4f}")

        avg_loss = total_loss / max(total_steps, 1)
        aux_losses: Dict[str, float] = {}
        if self.optim_config.freq_reg_weight > 0 and total_steps > 0:
            aux_losses["freq"] = freq_total / total_steps
        if self.optim_config.nll_weight > 0 and total_steps > 0:
            aux_losses["nll"] = nll_total / total_steps
        if self.optim_config.crps_weight > 0 and total_steps > 0:
            aux_losses["crps"] = crps_total / total_steps
        for key, total in reg_totals.items():
            aux_losses[f"reg/{key}"] = total / max(total_steps, 1)
        diagnostics_avg = {
            key: value / max(diag_count, 1) for key, value in diag_sums.items()
        }
        return avg_loss, aux_losses, diagnostics_avg

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        device = self.optim_config.device
        total = 0.0
        steps = 0
        for history, future, future_feats in loader:
            history = history.to(device)
            future = future.to(device)
            future_feats = future_feats.to(device)
            outputs = self.model(
                history,
                future_feats,
                return_covariance=False,
                return_diagnostics=False,
                collect_regularizers=False,
            )
            pred_norm = outputs["mean"]
            pred = self.model.encoder.denormalize(pred_norm, outputs["revin_state"])
            loss = self.mse(pred, future)
            total += loss.item()
            steps += 1
        return total / max(steps, 1)

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        if self.checkpoint_dir is None:
            return
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(ckpt, self.checkpoint_dir / f"epoch_{epoch}.pt")

    def fit(self) -> None:
        previous_stage: Optional[int] = None
        for epoch in range(1, self.optim_config.epochs + 1):
            if epoch <= self.optim_config.stage1_epochs:
                stage = 1
                progress = 0.0
            else:
                stage = 2
                warmup = max(self.optim_config.stage2_warmup_epochs, 1)
                progress = min(
                    1.0,
                    (epoch - self.optim_config.stage1_epochs) / warmup,
                )
            self.model.set_stage(stage, progress)
            if stage != previous_stage:
                self._update_optimizer_parameters()
                self._reset_ema(reset=previous_stage is None)
            elif self.ema_decay > 0:
                self._reset_ema(reset=False)
            train_loss, aux_losses, diagnostics = self.train_epoch(epoch)
            if self.valid_loader is not None:
                backup = self._swap_ema_weights()
                val_loss = self.evaluate(self.valid_loader)
                if backup is not None:
                    for name, param in self.model.named_parameters():
                        if not param.requires_grad:
                            continue
                        param.data.copy_(backup[name])
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self.save_checkpoint(epoch, val_loss)
            else:
                val_loss = None
            self._log_epoch(epoch, stage, train_loss, val_loss, aux_losses, diagnostics)
            previous_stage = stage
