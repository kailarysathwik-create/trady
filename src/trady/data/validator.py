"""Structured historical market data validation engine for TRADY."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pyarrow as pa

from trady.data.schema import CANONICAL_COLUMNS, OHLCVRecord


@dataclass(frozen=True)
class ValidationError:
    """Structured representation of a data integrity violation."""

    code: str
    message: str
    row_index: int | None = None
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationWarning:
    """Structured representation of a suspicious pattern or non-critical anomaly."""

    code: str
    message: str
    row_index: int | None = None
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    """Complete structured outcome of a dataset validation run."""

    is_valid: bool
    total_records: int
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def summary(self) -> dict[str, int]:
        """Aggregate occurrences by error/warning code."""
        counts: dict[str, int] = {}
        for err in self.errors:
            counts[err.code] = counts.get(err.code, 0) + 1
        for warn in self.warnings:
            counts[warn.code] = counts.get(warn.code, 0) + 1
        return counts

    def format_report(self) -> str:
        """Render a clean textual audit summary for CLI and logs."""
        status_label = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"Validation Status: {status_label}",
            f"Total Records Inspected: {self.total_records}",
            f"Errors: {self.error_count} | Warnings: {self.warning_count}",
        ]
        if self.errors:
            lines.append("\nErrors Detail (first 10):")
            for err in self.errors[:10]:
                loc = f"row {err.row_index}" if err.row_index is not None else "dataset"
                col = f" [{err.column}]" if err.column else ""
                lines.append(f"  - [{err.code}] at {loc}{col}: {err.message}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} additional error(s)")

        if self.warnings:
            lines.append("\nWarnings Detail (first 10):")
            for warn in self.warnings[:10]:
                loc = (
                    f"row {warn.row_index}" if warn.row_index is not None else "dataset"
                )
                col = f" [{warn.column}]" if warn.column else ""
                lines.append(f"  - [{warn.code}] at {loc}{col}: {warn.message}")
            if len(self.warnings) > 10:
                lines.append(
                    f"  ... and {len(self.warnings) - 10} additional warning(s)"
                )

        return "\n".join(lines)


def validate_records(
    records: Sequence[OHLCVRecord | dict[str, Any]],
    expected_symbol: str | None = None,
) -> ValidationResult:
    """Validate a sequence of OHLCV records or dictionaries against TRADY invariants."""
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []
    seen_keys: set[tuple[str, datetime]] = set()

    prev_time: datetime | None = None

    for idx, rec in enumerate(records):
        # Extract fields based on type
        if isinstance(rec, OHLCVRecord):
            timestamp = rec.timestamp
            symbol = rec.symbol
            open_p = rec.open
            high_p = rec.high
            low_p = rec.low
            close_p = rec.close
            vol = rec.volume
        elif isinstance(rec, dict):
            # Check for unexpected columns
            for col in rec:
                if col not in CANONICAL_COLUMNS:
                    warnings.append(
                        ValidationWarning(
                            code="WARN_UNEXPECTED_COLUMN",
                            message=f"Encountered non-canonical column '{col}'",
                            row_index=idx,
                            column=col,
                        )
                    )

            # Check for missing values / None / null
            missing_cols = [
                c for c in CANONICAL_COLUMNS if c not in rec or rec[c] is None
            ]
            if missing_cols:
                errors.append(
                    ValidationError(
                        code="ERR_MISSING_VALUE",
                        message=f"Missing or null required field(s): {missing_cols}",
                        row_index=idx,
                    )
                )
                continue

            # Validate types and values
            raw_ts = rec["timestamp"]
            if not isinstance(raw_ts, datetime):
                try:
                    timestamp = datetime.fromisoformat(str(raw_ts))
                except Exception as exc:
                    errors.append(
                        ValidationError(
                            code="ERR_MALFORMED_TIMESTAMP",
                            message=f"Cannot parse timestamp '{raw_ts}': {exc}",
                            row_index=idx,
                            column="timestamp",
                        )
                    )
                    continue
            else:
                timestamp = raw_ts

            symbol = str(rec["symbol"])
            try:
                open_p = float(rec["open"])
                high_p = float(rec["high"])
                low_p = float(rec["low"])
                close_p = float(rec["close"])
                vol = float(rec["volume"])
            except (ValueError, TypeError) as exc:
                errors.append(
                    ValidationError(
                        code="ERR_TYPE_MISMATCH",
                        message=f"Numerical parsing failed: {exc}",
                        row_index=idx,
                    )
                )
                continue
        else:
            errors.append(
                ValidationError(
                    code="ERR_INVALID_RECORD_TYPE",
                    message=f"Unknown record type {type(rec)}",
                    row_index=idx,
                )
            )
            continue

        # 1. Timezone check
        if timestamp.tzinfo is None:
            errors.append(
                ValidationError(
                    code="ERR_NAIVE_TIMESTAMP",
                    message=f"Timestamp '{timestamp}' must be timezone-aware (UTC)",
                    row_index=idx,
                    column="timestamp",
                )
            )

        # 2. Chronological sorting check
        if prev_time is not None:
            if timestamp < prev_time:
                errors.append(
                    ValidationError(
                        code="ERR_UNSORTED_TIMESTAMPS",
                        message=(
                            f"Timestamp {timestamp} is earlier than previous "
                            f"record timestamp {prev_time}"
                        ),
                        row_index=idx,
                        column="timestamp",
                    )
                )

        # 3. Duplicate timestamp / symbol-timestamp check
        key = (symbol, timestamp)
        if key in seen_keys:
            errors.append(
                ValidationError(
                    code="ERR_DUPLICATE_TIMESTAMP",
                    message=(
                        f"Duplicate timestamp detected for symbol '{symbol}': "
                        f"{timestamp}"
                    ),
                    row_index=idx,
                    column="timestamp",
                )
            )
        else:
            seen_keys.add(key)

        prev_time = timestamp

        # 4. Symbol matching check
        if expected_symbol and symbol != expected_symbol:
            errors.append(
                ValidationError(
                    code="ERR_UNEXPECTED_SYMBOL",
                    message=(f"Expected symbol '{expected_symbol}', got '{symbol}'"),
                    row_index=idx,
                    column="symbol",
                )
            )

        # 5. Price positivity
        for price_name, price_val in [
            ("open", open_p),
            ("high", high_p),
            ("low", low_p),
            ("close", close_p),
        ]:
            if price_val <= 0:
                errors.append(
                    ValidationError(
                        code="ERR_NON_POSITIVE_PRICE",
                        message=(
                            f"{price_name.capitalize()} price must be > 0, "
                            f"got {price_val}"
                        ),
                        row_index=idx,
                        column=price_name,
                    )
                )

        # 6. Volume non-negativity
        if vol < 0:
            errors.append(
                ValidationError(
                    code="ERR_NEGATIVE_VOLUME",
                    message=f"Volume cannot be negative, got {vol}",
                    row_index=idx,
                    column="volume",
                )
            )
        elif vol == 0:
            warnings.append(
                ValidationWarning(
                    code="WARN_ZERO_VOLUME",
                    message="Volume is zero for this trading period",
                    row_index=idx,
                    column="volume",
                )
            )

        # 7. OHLC mathematical relationships
        if high_p < open_p:
            errors.append(
                ValidationError(
                    code="ERR_INVALID_OHLC_HIGH_OPEN",
                    message=f"High ({high_p}) is lower than Open ({open_p})",
                    row_index=idx,
                    column="high",
                )
            )
        if high_p < close_p:
            errors.append(
                ValidationError(
                    code="ERR_INVALID_OHLC_HIGH_CLOSE",
                    message=f"High ({high_p}) is lower than Close ({close_p})",
                    row_index=idx,
                    column="high",
                )
            )
        if high_p < low_p:
            errors.append(
                ValidationError(
                    code="ERR_INVALID_OHLC_HIGH_LOW",
                    message=f"High ({high_p}) is lower than Low ({low_p})",
                    row_index=idx,
                    column="high",
                )
            )
        if low_p > open_p:
            errors.append(
                ValidationError(
                    code="ERR_INVALID_OHLC_LOW_OPEN",
                    message=f"Low ({low_p}) is higher than Open ({open_p})",
                    row_index=idx,
                    column="low",
                )
            )
        if low_p > close_p:
            errors.append(
                ValidationError(
                    code="ERR_INVALID_OHLC_LOW_CLOSE",
                    message=f"Low ({low_p}) is higher than Close ({close_p})",
                    row_index=idx,
                    column="low",
                )
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        total_records=len(records),
        errors=errors,
        warnings=warnings,
    )


def validate_arrow_table(
    table: pa.Table, expected_symbol: str | None = None
) -> ValidationResult:
    """Validate a PyArrow Table containing market data against TRADY canonical rules."""
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    # Check required columns
    table_cols = set(table.column_names)
    for col in CANONICAL_COLUMNS:
        if col not in table_cols:
            errors.append(
                ValidationError(
                    code="ERR_MISSING_COLUMN",
                    message=f"Required canonical column '{col}' is missing",
                    column=col,
                )
            )

    # Check for unexpected columns
    for col in table_cols:
        if col not in CANONICAL_COLUMNS:
            warnings.append(
                ValidationWarning(
                    code="WARN_UNEXPECTED_COLUMN",
                    message=f"Encountered non-canonical column '{col}' in table",
                    column=col,
                )
            )

    if errors:
        return ValidationResult(
            is_valid=False,
            total_records=table.num_rows,
            errors=errors,
            warnings=warnings,
        )

    # Convert table to row-level dictionaries for detailed invariant checking
    pydict_rows = table.to_pylist()
    return validate_records(pydict_rows, expected_symbol=expected_symbol)
