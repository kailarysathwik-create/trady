"""Data ingestion and storage interface contracts for TRADY.

NOTE: Concrete market data providers are intentionally deferred in this foundation.
This module defines the architectural contracts and protocols for future data pipelines.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MarketDataBatch:
    """Immutable representation of a validated historical market data slice.

    Adheres to the principle of never silently altering historical data.
    """

    symbol: str
    start_time: datetime
    end_time: datetime
    record_count: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.start_time > self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) cannot exceed "
                f"end_time ({self.end_time})"
            )
        if self.record_count < 0:
            raise ValueError("record_count must be non-negative")


@runtime_checkable
class MarketDataSource(Protocol):
    """Protocol defining the public interface for historical data sources."""

    def load_slice(
        self, symbol: str, start: datetime, end: datetime
    ) -> MarketDataBatch:
        """Load a validated slice of historical market data."""
        ...

    def validate_integrity(self, batch: MarketDataBatch) -> bool:
        """Verify that data contains no lookahead leakage or timestamp anomalies."""
        ...


__all__ = ["MarketDataBatch", "MarketDataSource"]
