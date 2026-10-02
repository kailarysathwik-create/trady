"""Feature registry and metadata management for TRADY.

Every feature exposed in TRADY must register metadata declaring its lookback
requirements, input dependencies, and point-in-time causality safety.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata describing a causal feature transform.

    Attributes:
        name: Canonical unique name of the feature (e.g. 'return_1d', 'vol_std_20').
        description: Mathematical and conceptual description of the feature.
        feature_group: Category (e.g. 'PRICE', 'MOMENTUM', 'VOLATILITY', 'VOLUME',
            'PRICE_STRUCTURE').
        required_columns: Input column names required (e.g. ('close',)).
        lookback_period: Number of past periods required for the window calculation.
        min_observations: Minimum valid observations before outputting non-null values.
        is_point_in_time: Whether the calculation is strictly causal.
        lookback_window: Backward-compatible alias for lookback_period.
    """

    name: str
    description: str
    feature_group: str = "PRICE"
    required_columns: tuple[str, ...] = ("close",)
    lookback_period: int = 1
    min_observations: int = 1
    is_point_in_time: bool = True
    lookback_window: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Feature name cannot be empty")

        # Reconcile lookback_window alias if supplied
        effective_lookback = (
            self.lookback_window
            if self.lookback_window is not None
            else self.lookback_period
        )
        if effective_lookback < 1:
            raise ValueError(
                "lookback_window must be at least 1 bar to prevent lookahead"
            )

        object.__setattr__(self, "lookback_period", effective_lookback)
        if self.lookback_window is None:
            object.__setattr__(self, "lookback_window", effective_lookback)

        if self.min_observations < 1:
            raise ValueError("min_observations must be at least 1 bar")

        if not self.required_columns:
            raise ValueError("required_columns cannot be empty")


class FeatureRegistry:
    """Central registry for TRADY quantitative features."""

    def __init__(self) -> None:
        self._features: dict[str, Callable[..., Any]] = {}
        self._metadata: dict[str, FeatureMetadata] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        metadata: FeatureMetadata,
    ) -> None:
        """Register a feature calculation function and its metadata."""
        if not name or not name.strip():
            raise ValueError("Feature name cannot be empty")
        if name in self._features:
            raise ValueError(f"Feature '{name}' is already registered")

        self._features[name] = func
        self._metadata[name] = metadata

    def register_decorator(
        self,
        metadata: FeatureMetadata,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a feature calculation function."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(metadata.name, func, metadata)
            return func

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        """Retrieve the calculation function for a registered feature."""
        if name not in self._features:
            avail = list(self._features.keys())
            raise KeyError(f"Feature '{name}' not found. Available features: {avail}")
        return self._features[name]

    def get_metadata(self, name: str) -> FeatureMetadata:
        """Retrieve metadata for a registered feature."""
        if name not in self._metadata:
            avail = list(self._metadata.keys())
            raise KeyError(f"Feature '{name}' metadata not found. Available: {avail}")
        return self._metadata[name]

    def list_features(self) -> list[FeatureMetadata]:
        """List all registered feature metadata sorted by name."""
        return sorted(self._metadata.values(), key=lambda m: m.name)

    def list_by_group(self, group: str) -> list[FeatureMetadata]:
        """List registered features belonging to a specific group."""
        group_upper = group.upper()
        return [
            m for m in self.list_features() if m.feature_group.upper() == group_upper
        ]

    def contains(self, name: str) -> bool:
        """Check if a feature is registered."""
        return name in self._features

    def clear(self) -> None:
        """Clear the registry (used primarily in test teardown)."""
        self._features.clearseeded = False  # type: ignore[attr-defined]
        self._features.clear()
        self._metadata.clear()


# Global default registry instance
registry = FeatureRegistry()
