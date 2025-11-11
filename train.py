"""Command-line training entry point for LKH-SDE v3."""
from __future__ import annotations

import argparse
from torch.utils.data import DataLoader

from lkh_sde.data.dataset import DatasetConfig, WindowedTimeSeriesDataset
from lkh_sde.models.lkh_sde_v3 import CIConfig, FusionConfig, LKHSDEV3, ModelConfig
from lkh_sde.modules.koopman import KoopmanConfig
from lkh_sde.trainer import OptimConfig, Trainer
from lkh_sde.utils.simple_yaml import parse_simple_yaml
from lkh_sde.utils.time_features import TimeFeatureConfig


def build_model_config(cfg: dict, dataset_cfg: DatasetConfig) -> ModelConfig:
    cfg = dict(cfg)
    future_dim = dataset_cfg.future_feature_dim()
    cfg["known_future_dim"] = future_dim
    koopman_cfg = KoopmanConfig(**cfg["koopman"])
    ci_cfg = CIConfig(**cfg.get("ci", {}))
    fusion_cfg = FusionConfig(**cfg.get("fusion", {}))
    model_kwargs = {
        k: v
        for k, v in cfg.items()
        if k
        not in {
            "koopman",
            "ci",
            "fusion",
        }
    }
    return ModelConfig(
        koopman=koopman_cfg,
        ci=ci_cfg,
        fusion=fusion_cfg,
        **model_kwargs,
    )


def build_dataset_config(cfg: dict) -> DatasetConfig:
    cfg = dict(cfg)
    tf_cfg = cfg.get("time_feature_config")
    if isinstance(tf_cfg, dict):
        cfg["time_feature_config"] = TimeFeatureConfig(**tf_cfg)
    return DatasetConfig(**cfg)


def build_optim_config(cfg: dict) -> OptimConfig:
    return OptimConfig(**cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LKH-SDE v3")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = parse_simple_yaml(f.read())

    dataset_cfg = build_dataset_config(config["dataset"])
    model_cfg = build_model_config(config["model"], dataset_cfg)
    optim_cfg = build_optim_config(config["optim"])

    train_dataset = WindowedTimeSeriesDataset(dataset_cfg, split="train")
    valid_dataset = WindowedTimeSeriesDataset(dataset_cfg, split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        drop_last=False,
    )

    model = LKHSDEV3(model_cfg)
    trainer = Trainer(model, train_loader, valid_loader, optim_cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
