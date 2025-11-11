"""Windowed time-series dataset with optional known future features."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from lkh_sde.utils.time_features import TimeFeatureConfig, build_time_features


@dataclass
class DatasetConfig:
    data_path: str
    target_columns: Sequence[str]
    known_future_columns: Optional[Sequence[str]] = None
    lookback: int = 336
    horizon: int = 96
    stride: int = 1
    time_feature_config: TimeFeatureConfig = field(default_factory=TimeFeatureConfig)


class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, config: DatasetConfig, split: str = "train") -> None:
        super().__init__()
        self.config = config
        timestamps: List[str] = []
        targets: List[List[float]] = []
        known_future: List[List[float]] = []
        with open(config.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(row[reader.fieldnames[0]])
                targets.append([float(row[col]) for col in config.target_columns])
                if config.known_future_columns:
                    known_future.append([float(row[col]) for col in config.known_future_columns])
        self.timestamps = timestamps
        features = np.asarray(targets, dtype=np.float32)
        if config.known_future_columns:
            known_future_arr = np.asarray(known_future, dtype=np.float32)
        else:
            known_future_arr = None
        time_feats = build_time_features(timestamps, config.time_feature_config)
        split_slice = self._compute_split_indices(len(features), split, config.lookback)
        self.features = features[split_slice]
        self.time_features = time_feats[split_slice]
        self.known_future = (
            known_future_arr[split_slice] if known_future_arr is not None else None
        )
        self.lookback = config.lookback
        self.horizon = config.horizon
        self.stride = config.stride

    def _compute_split_indices(self, length: int, split: str, lookback: int) -> slice:
        train_end = int(0.6 * length)
        valid_end = int(0.8 * length)
        if split == "train":
            return slice(0, train_end)
        if split == "valid":
            return slice(max(train_end - lookback, 0), valid_end)
        if split == "test":
            return slice(max(valid_end - lookback, 0), length)
        raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        available = self.features.shape[0] - (self.lookback + self.horizon)
        if available < 0:
            return 0
        return available // self.stride + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.lookback
        future_end = end + self.horizon
        history = self.features[start:end]
        future = self.features[end:future_end]
        time_future = self.time_features[end:future_end]
        if self.known_future is not None:
            known_future = self.known_future[end:future_end]
            future_features = np.concatenate([time_future, known_future], axis=-1)
        else:
            future_features = time_future
        return (
            torch.from_numpy(history.copy()),
            torch.from_numpy(future.copy()),
            torch.from_numpy(future_features.astype(np.float32).copy()),
        )
