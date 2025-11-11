"""Utilities for constructing time and seasonal features."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

import math
import numpy as np


@dataclass
class TimeFeatureConfig:
    use_hour: bool = True
    use_day_of_week: bool = True
    use_day_of_year: bool = True
    use_month: bool = True


def _sin_cos(values: Sequence[float], period: float) -> np.ndarray:
    angles = [2 * math.pi * v / period for v in values]
    return np.stack([np.sin(angles), np.cos(angles)], axis=-1)


def _ensure_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def build_time_features(
    timestamps: Iterable[object],
    config: Optional[TimeFeatureConfig] = None,
) -> np.ndarray:
    if config is None:
        config = TimeFeatureConfig()

    timestamps = [_ensure_datetime(ts) for ts in timestamps]
    features: List[np.ndarray] = []

    if config.use_hour:
        hours = [ts.hour for ts in timestamps]
        features.append(_sin_cos(hours, 24.0))
    if config.use_day_of_week:
        dows = [ts.weekday() for ts in timestamps]
        features.append(_sin_cos(dows, 7.0))
    if config.use_day_of_year:
        doys = [ts.timetuple().tm_yday for ts in timestamps]
        features.append(_sin_cos(doys, 366.0))
    if config.use_month:
        months = [ts.month for ts in timestamps]
        features.append(_sin_cos(months, 12.0))

    if not features:
        return np.empty((len(timestamps), 0), dtype=np.float32)

    combined = np.concatenate(features, axis=-1)
    return combined.astype(np.float32)
