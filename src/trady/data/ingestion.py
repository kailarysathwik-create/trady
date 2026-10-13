"""Ingestion, provenance tracking, and storage pipeline for real market data."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from trady.data.metadata import DatasetMetadata
from trady.data.normalizer import NormalizationConfig, normalize_records
from trady.data.provider import MarketDataProvider
from trady.data.storage import save_dataset
from trady.data.validator import ValidationResult, validate_records
from trady.data.yahoo import YahooFinanceProvider
from trady.utils.logging import get_logger

logger = get_logger("data.ingestion")


def _sanitize_symbol_slug(symbol: str) -> str:
    """Sanitize symbol string for filesystem paths (e.g. '^NSEI' -> 'nsei')."""
    cleaned = symbol.strip().lstrip("^").replace("/", "_").replace("\\", "_")
    return cleaned.lower()


def download_and_ingest_market_data(
    symbol: str,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    timeframe: str = "1d",
    output_dir: Path | str | None = None,
    raw_dir: Path | str | None = None,
    custom_output: Path | str | None = None,
    custom_raw_output: Path | str | None = None,
    overwrite: bool = False,
    auto_adjust: bool = False,
    preview_only: bool = False,
    provider: MarketDataProvider | None = None,
) -> dict[str, Any]:
    """Download historical market data, record provenance, validate, and store.

    Args:
        symbol: Ticker symbol (e.g. '^NSEI', 'SPY', 'NVDA').
        start_date: Start date for historical bars (YYYY-MM-DD).
        end_date: End date for historical bars (YYYY-MM-DD).
        timeframe: Bar interval (defaults to '1d').
        output_dir: Directory for normalized Parquet files (default data/processed).
        raw_dir: Directory for raw JSON responses (default data/raw).
        custom_output: Explicit destination path for processed Parquet file.
        custom_raw_output: Explicit destination path for raw JSON file.
        overwrite: If False, raises error if files already exist.
        auto_adjust: Whether to adjust OHLC prices for splits/dividends.
        preview_only: If True, fetches & validates in-memory without saving.
        provider: Provider instance (defaults to YahooFinanceProvider).

    Returns:
        Structured ingestion result dictionary.

    Raises:
        FileExistsError: If destination files exist and overwrite=False.
        ValueError: If validation fails or input dates are invalid.
    """
    prov = provider or YahooFinanceProvider()
    slug = _sanitize_symbol_slug(symbol)

    # 1. Resolve storage paths
    root = Path.cwd()
    p_dir = Path(output_dir) if output_dir else root / "data" / "processed"
    r_dir = Path(raw_dir) if raw_dir else root / "data" / "raw"

    start_str = YahooFinanceProvider._format_date(start_date) or "start"
    end_str = YahooFinanceProvider._format_date(end_date) or "end"

    raw_path = (
        Path(custom_raw_output)
        if custom_raw_output
        else r_dir / f"{slug}_{timeframe}_{start_str}_{end_str}_raw.json"
    )

    processed_path = (
        Path(custom_output) if custom_output else p_dir / f"{slug}_{timeframe}.parquet"
    )

    # 2. Overwrite Protection Check
    if not preview_only and not overwrite:
        if raw_path.exists():
            raise FileExistsError(
                f"Raw dataset already exists at '{raw_path}'. "
                "Specify --overwrite to replace existing data."
            )
        if processed_path.exists():
            raise FileExistsError(
                f"Normalized dataset already exists at '{processed_path}'. "
                "Specify --overwrite to replace existing data."
            )

    # 3. Retrieve raw records from provider
    raw_records, provenance = prov.fetch_raw(
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_date,
        end_time=end_date,
        auto_adjust=auto_adjust,
    )

    if not raw_records:
        raise ValueError(
            f"Provider '{prov.provider_name}' returned zero records for {symbol}."
        )

    # 4. Save raw downloaded response separately if not preview
    if not preview_only:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_payload = {
            "provenance": provenance,
            "raw_records": raw_records,
        }
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_payload, f, indent=2)
        logger.info("Preserved raw download payload at %s", raw_path)

    # 5. Normalize into TRADY canonical OHLCV records
    norm_cfg = NormalizationConfig(default_symbol=symbol)
    try:
        canonical_records, audit_logs = normalize_records(raw_records, config=norm_cfg)
    except ValueError as val_err:
        provenance["validation_outcome"] = "FAILED"
        provenance["validation_errors"] = [str(val_err)]
        logger.error(
            "Dataset normalization/invariant check failed for %s: %s",
            symbol,
            val_err,
        )
        raise ValueError(
            f"Dataset validation failed for '{symbol}'. Invalid dataset quarantined "
            f"and prevented from entering research pipeline.\n{val_err}"
        ) from val_err

    # 6. Validate canonical records against strict invariants
    exp_start_dt = datetime.fromisoformat(start_str) if start_str != "start" else None
    exp_end_dt = datetime.fromisoformat(end_str) if end_str != "end" else None

    validation: ValidationResult = validate_records(
        canonical_records,
        expected_symbol=symbol,
        expected_start=exp_start_dt,
        expected_end=exp_end_dt,
    )

    if not validation.is_valid:
        provenance["validation_outcome"] = "FAILED"
        provenance["validation_errors"] = [e.message for e in validation.errors]
        report = validation.format_report()
        logger.error("Dataset validation failed for %s:\n%s", symbol, report)
        raise ValueError(
            f"Dataset validation failed for '{symbol}'. Invalid dataset quarantined "
            f"and prevented from entering research pipeline.\n{report}"
        )

    provenance["validation_outcome"] = "PASSED"
    provenance["validation_warnings"] = [w.message for w in validation.warnings]

    # 7. Store normalized dataset in Parquet and DuckDB workflow
    if not preview_only:
        meta = DatasetMetadata(
            provider=prov.provider_name,
            symbol=symbol,
            timeframe=timeframe,
            start_time=canonical_records[0].timestamp,
            end_time=canonical_records[-1].timestamp,
            timezone="UTC",
            record_count=len(canonical_records),
            extra=provenance,
        )
        save_dataset(canonical_records, destination_path=processed_path, metadata=meta)
        logger.info(
            "Normalized & validated dataset stored at %s (%d rows)",
            processed_path,
            len(canonical_records),
        )

    return {
        "status": "success",
        "symbol": symbol,
        "record_count": len(canonical_records),
        "start_time": canonical_records[0].timestamp.isoformat(),
        "end_time": canonical_records[-1].timestamp.isoformat(),
        "raw_file": str(raw_path) if not preview_only else None,
        "processed_file": str(processed_path) if not preview_only else None,
        "meta_file": (
            str(processed_path.with_suffix(".meta.json")) if not preview_only else None
        ),
        "validation": validation,
        "preview_records": [
            {
                "timestamp": r.timestamp.isoformat(),
                "symbol": r.symbol,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in canonical_records[:5]
        ],
        "is_preview": preview_only,
    }
