"""Historical market data normalization pipeline for TRADY."""

import zoneinfo
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from trady.data.schema import OHLCVRecord, get_canonical_arrow_schema
from trady.utils.logging import get_logger

logger = get_logger("data.normalizer")

# Default column name synonyms mapped to TRADY canonical column names
DEFAULT_COLUMN_SYNONYMS: dict[str, str] = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "timestamp": "timestamp",
    "ts": "timestamp",
    "ticker": "symbol",
    "symbol": "symbol",
    "sym": "symbol",
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "adj_close": "close",
    "adjusted_close": "close",
    "volume": "volume",
    "vol": "volume",
    "v": "volume",
}


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration governing the normalization pipeline."""

    deduplicate: bool = False
    column_mapping: dict[str, str] = field(default_factory=dict)
    target_timezone: str = "UTC"
    default_symbol: str | None = None


def _parse_to_utc_datetime(val: Any, target_tz_name: str = "UTC") -> datetime:
    """Parse heterogeneous timestamp inputs into a timezone-aware UTC datetime."""
    if isinstance(val, datetime):
        if val.tzinfo is None:
            # Assume UTC if naive, but log notice
            return val.replace(tzinfo=UTC)
        return val.astimezone(UTC)

    if isinstance(val, (int, float)):
        # Treat numeric as UNIX timestamp (seconds if < 1e11, milliseconds if >= 1e11)
        if val > 1e11:
            val = val / 1000.0
        return datetime.fromtimestamp(val, tz=UTC)

    if isinstance(val, str):
        val_str = val.strip()
        # Handle ISO strings with Z or offsets
        if val_str.endswith("Z"):
            val_str = val_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(val_str)
        if dt.tzinfo is None:
            tz = zoneinfo.ZoneInfo(target_tz_name)
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(UTC)

    raise ValueError(f"Unsupported timestamp format: {type(val)} ({val})")


def normalize_records(
    raw_records: Sequence[dict[str, Any]],
    config: NormalizationConfig | None = None,
) -> tuple[list[OHLCVRecord], list[str]]:
    """Normalize raw dictionary records into canonical OHLCVRecord objects.

    Returns:
        tuple of (normalized_records, audit_log_messages).
    """
    if config is None:
        config = NormalizationConfig()

    audit_logs: list[str] = []
    if not raw_records:
        return [], ["Input record set is empty."]

    # Build active column mapping (synonyms + custom overrides)
    active_mapping = DEFAULT_COLUMN_SYNONYMS.copy()
    active_mapping.update(config.column_mapping)

    normalized_raw: list[dict[str, Any]] = []

    for idx, raw_row in enumerate(raw_records):
        new_row: dict[str, Any] = {}
        # 1. Column renaming
        for col_name, val in raw_row.items():
            cleaned_col = str(col_name).strip().lower()
            mapped_col = active_mapping.get(cleaned_col, cleaned_col)
            new_row[mapped_col] = val

        # 2. Inject default symbol if missing
        if "symbol" not in new_row or new_row["symbol"] is None:
            if config.default_symbol:
                new_row["symbol"] = config.default_symbol
            else:
                raise ValueError(
                    f"Row {idx} missing 'symbol' and no default_symbol configured."
                )

        # 3. Timestamp conversion
        if "timestamp" not in new_row or new_row["timestamp"] is None:
            raise ValueError(f"Row {idx} is missing a timestamp.")
        new_row["timestamp"] = _parse_to_utc_datetime(
            new_row["timestamp"], config.target_timezone
        )

        # 4. Enforce uppercase symbol
        new_row["symbol"] = str(new_row["symbol"]).strip().upper()

        # 5. Numerical type casting
        try:
            new_row["open"] = float(new_row["open"])
            new_row["high"] = float(new_row["high"])
            new_row["low"] = float(new_row["low"])
            new_row["close"] = float(new_row["close"])
            new_row["volume"] = float(new_row["volume"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Row {idx} failed numerical casting: {exc}") from exc

        normalized_raw.append(new_row)

    audit_logs.append(
        f"Normalized {len(normalized_raw)} records: "
        "standardized column names and types."
    )

    # 6. Deduplication (only when explicitly configured)
    final_raw: list[dict[str, Any]]
    if config.deduplicate:
        seen_keys: set[tuple[str, datetime]] = set()
        deduped: list[dict[str, Any]] = []
        for r in normalized_raw:
            k = (r["symbol"], r["timestamp"])
            if k not in seen_keys:
                seen_keys.add(k)
                deduped.append(r)
        dropped_count = len(normalized_raw) - len(deduped)
        if dropped_count > 0:
            msg = (
                f"Explicit deduplication enabled: dropped {dropped_count} "
                "duplicate record(s)."
            )
            audit_logs.append(msg)
            logger.info(msg)
        final_raw = deduped
    else:
        final_raw = normalized_raw

    # 7. Chronological sorting
    final_raw.sort(key=lambda r: (r["symbol"], r["timestamp"]))
    audit_logs.append(
        "Chronologically sorted records by symbol and timestamp ascending."
    )

    # 8. Instantiate OHLCVRecord objects
    records = [
        OHLCVRecord(
            timestamp=r["timestamp"],
            symbol=r["symbol"],
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=r["volume"],
        )
        for r in final_raw
    ]

    return records, audit_logs


def records_to_arrow_table(
    records: Sequence[OHLCVRecord | dict[str, Any]],
) -> pa.Table:
    """Convert OHLCV records or dicts into PyArrow Table with canonical schema."""
    schema = get_canonical_arrow_schema()
    if not records:
        return schema.empty_table()

    timestamps: list[datetime] = []
    symbols: list[str] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []

    for r in records:
        if isinstance(r, OHLCVRecord):
            timestamps.append(r.timestamp)
            symbols.append(r.symbol)
            opens.append(r.open)
            highs.append(r.high)
            lows.append(r.low)
            closes.append(r.close)
            volumes.append(r.volume)
        else:
            raw_ts = r["timestamp"]
            ts = (
                raw_ts
                if isinstance(raw_ts, datetime)
                else _parse_to_utc_datetime(raw_ts)
            )
            timestamps.append(ts)
            symbols.append(str(r["symbol"]).upper())
            opens.append(float(r["open"]))
            highs.append(float(r["high"]))
            lows.append(float(r["low"]))
            closes.append(float(r["close"]))
            volumes.append(float(r["volume"]))

    return pa.Table.from_arrays(
        [
            pa.array(timestamps, type=schema.field("timestamp").type),
            pa.array(symbols, type=pa.string()),
            pa.array(opens, type=pa.float64()),
            pa.array(highs, type=pa.float64()),
            pa.array(lows, type=pa.float64()),
            pa.array(closes, type=pa.float64()),
            pa.array(volumes, type=pa.float64()),
        ],
        schema=schema,
    )


def arrow_table_to_records(table: pa.Table) -> list[OHLCVRecord]:
    """Convert a PyArrow Table into a list of OHLCVRecord objects."""
    pylist = table.to_pylist()
    return [
        OHLCVRecord(
            timestamp=r["timestamp"],
            symbol=r["symbol"],
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r["volume"]),
        )
        for r in pylist
    ]
