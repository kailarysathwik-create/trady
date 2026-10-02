"""Data layer for TRADY: ingestion, validation, normalization, and Parquet/DuckDB.

Establishes the foundational data engine for quantitative research without
live execution.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from trady.data.metadata import DatasetMetadata
from trady.data.normalizer import (
    NormalizationConfig,
    arrow_table_to_records,
    normalize_records,
    records_to_arrow_table,
)
from trady.data.provider import MarketDataProvider, SyntheticDataProvider
from trady.data.schema import CANONICAL_COLUMNS, OHLCVRecord, get_canonical_arrow_schema
from trady.data.storage import (
    calculate_file_sha256,
    inspect_dataset,
    load_dataset,
    load_records,
    query_dataset,
    save_dataset,
)
from trady.data.validator import (
    ValidationError,
    ValidationResult,
    ValidationWarning,
    validate_arrow_table,
    validate_records,
)


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


__all__ = [
    "CANONICAL_COLUMNS",
    "DatasetMetadata",
    "MarketDataBatch",
    "MarketDataProvider",
    "MarketDataSource",
    "NormalizationConfig",
    "OHLCVRecord",
    "SyntheticDataProvider",
    "ValidationError",
    "ValidationResult",
    "ValidationWarning",
    "arrow_table_to_records",
    "calculate_file_sha256",
    "get_canonical_arrow_schema",
    "inspect_dataset",
    "load_dataset",
    "load_records",
    "normalize_records",
    "query_dataset",
    "records_to_arrow_table",
    "save_dataset",
    "validate_arrow_table",
    "validate_records",
]
