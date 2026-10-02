"""Canonical OHLCV data model, typing, and invariants for TRADY."""

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import pyarrow as pa

CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


@dataclass(frozen=True, slots=True)
class OHLCVRecord:
    """Canonical representation of a single OHLCV market bar.

    All timestamps must be timezone-aware (strictly UTC).
    All prices must be positive floats.
    Volume must be non-negative.
    Standard high/low invariants must hold:
      high >= open, high >= close, high >= low
      low <= open, low <= close, low <= high
    """

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                f"Timestamp {self.timestamp} must be timezone-aware (UTC)."
            )
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError(f"Symbol must be a non-empty string, got {self.symbol!r}")

        # Price positivity
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError(
                f"All prices must be strictly positive: open={self.open}, "
                f"high={self.high}, low={self.low}, close={self.close}"
            )

        # Volume non-negativity
        if self.volume < 0:
            raise ValueError(f"Volume must be non-negative, got {self.volume}")

        # High bounds
        if self.high < self.open or self.high < self.close or self.high < self.low:
            raise ValueError(
                f"High ({self.high}) must be >= open ({self.open}), "
                f"close ({self.close}), and low ({self.low})"
            )

        # Low bounds
        if self.low > self.open or self.low > self.close or self.low > self.high:
            raise ValueError(
                f"Low ({self.low}) must be <= open ({self.open}), "
                f"close ({self.close}), and high ({self.high})"
            )


def get_canonical_arrow_schema() -> pa.Schema:
    """Return the canonical PyArrow schema for TRADY processed Parquet datasets."""
    return pa.schema(
        [
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("symbol", pa.string()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
        ]
    )
