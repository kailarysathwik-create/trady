"""TRADY Feature Engine.

Transforms validated historical OHLCV data into reproducible, strictly causal
numerical features for quantitative research and machine-learning models.

Guarantees:
- Every feature at timestamp T depends exclusively on observations at or before T.
- No future leakage.
- No random shuffling.
- Deterministic calculations.
- Clean feature registry and point-in-time safety metadata.
"""

from typing import Any, Protocol, runtime_checkable

from trady.features.config import FeatureConfig
from trady.features.momentum import (
    compute_price_to_sma_ratio,
    compute_rate_of_change,
    compute_rolling_momentum,
    compute_sma,
    compute_sma_ratio,
)
from trady.features.pipeline import (
    FeatureDatasetMetadata,
    FeaturePipeline,
    FeatureValidationReport,
)
from trady.features.price import (
    compute_log_return,
    compute_multi_period_returns,
    compute_simple_return,
)
from trady.features.registry import (
    FeatureMetadata,
    FeatureRegistry,
    registry,
)
from trady.features.structure import (
    compute_distance_from_rolling_high,
    compute_distance_from_rolling_low,
    compute_normalized_range_position,
)
from trady.features.volatility import (
    compute_average_true_range,
    compute_normalized_atr,
    compute_realized_volatility,
    compute_rolling_std,
    compute_true_range,
)
from trady.features.volume import (
    compute_volume_change,
    compute_volume_to_sma_ratio,
    compute_volume_zscore,
)


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for causal feature extraction."""

    @property
    def metadata(self) -> FeatureMetadata:
        """Return the feature metadata."""
        ...

    def extract(self, data: Any) -> Any:
        """Extract features causally from input data without future knowledge."""
        ...


__all__ = [
    "FeatureConfig",
    "FeatureDatasetMetadata",
    "FeatureExtractor",
    "FeatureMetadata",
    "FeaturePipeline",
    "FeatureRegistry",
    "FeatureValidationReport",
    "compute_average_true_range",
    "compute_distance_from_rolling_high",
    "compute_distance_from_rolling_low",
    "compute_log_return",
    "compute_multi_period_returns",
    "compute_normalized_atr",
    "compute_normalized_range_position",
    "compute_price_to_sma_ratio",
    "compute_rate_of_change",
    "compute_realized_volatility",
    "compute_rolling_momentum",
    "compute_rolling_std",
    "compute_simple_return",
    "compute_sma",
    "compute_sma_ratio",
    "compute_true_range",
    "compute_volume_change",
    "compute_volume_to_sma_ratio",
    "compute_volume_zscore",
    "registry",
]
