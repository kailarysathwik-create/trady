"""Feature engineering interface contracts for TRADY.

Features must be computed strictly causally without future lookahead bias.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata describing a causal feature transform."""

    name: str
    lookback_window: int
    description: str

    def __post_init__(self) -> None:
        if self.lookback_window < 1:
            raise ValueError(
                "lookback_window must be at least 1 bar to prevent lookahead"
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


__all__ = ["FeatureExtractor", "FeatureMetadata"]
