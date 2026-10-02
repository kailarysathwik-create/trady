"""Configuration management for TRADY Feature Engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

VALID_FEATURE_GROUPS = (
    "PRICE",
    "MOMENTUM",
    "VOLATILITY",
    "VOLUME",
    "PRICE_STRUCTURE",
)

NaNPolicy = Literal["keep", "drop", "error"]


@dataclass(frozen=True)
class FeatureConfig:
    """Declarative configuration for feature calculation pipelines.

    Attributes:
        enabled_groups: List of feature groups to compute.
        selected_features: Optional explicit list of feature names. If specified,
            takes precedence over enabled_groups.
        nan_policy: Handling strategy for initial lookback warmup NaNs:
            - 'keep': Retain rows with NaNs to maintain exact bar alignment (default).
            - 'drop': Drop rows containing NaNs across computed features.
            - 'error': Raise exception if any NaNs exist.
        output_dir: Directory where processed feature Parquet files are stored.
    """

    enabled_groups: tuple[str, ...] = VALID_FEATURE_GROUPS
    selected_features: tuple[str, ...] | None = None
    nan_policy: NaNPolicy = "keep"
    output_dir: str = "data/features"
    custom_params: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.nan_policy not in ("keep", "drop", "error"):
            msg = (
                f"Invalid nan_policy '{self.nan_policy}'. Must be keep, drop, or error"
            )
            raise ValueError(msg)

        for group in self.enabled_groups:
            if group.upper() not in VALID_FEATURE_GROUPS:
                msg = f"Unknown feature group '{group}'. Valid: {VALID_FEATURE_GROUPS}"
                raise ValueError(msg)

    def resolved_output_path(self, base_dir: Path | None = None) -> Path:
        """Resolve output directory relative to project root or provided base."""
        out = Path(self.output_dir)
        if out.is_absolute() or base_dir is None:
            return out
        return base_dir / out
