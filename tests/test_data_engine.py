"""Comprehensive unit test suite for the TRADY Data Engine."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trady.data.metadata import DatasetMetadata
from trady.data.normalizer import NormalizationConfig, normalize_records
from trady.data.provider import SyntheticDataProvider
from trady.data.schema import OHLCVRecord
from trady.data.storage import (
    calculate_file_sha256,
    inspect_dataset,
    load_records,
    query_dataset,
    save_dataset,
)
from trady.data.validator import validate_records


@pytest.fixture
def provider() -> SyntheticDataProvider:
    """Deterministic synthetic data generator fixture."""
    return SyntheticDataProvider(seed=42)


def test_valid_dataset_passes_validation(provider: SyntheticDataProvider) -> None:
    """Verify that a pristine synthetic dataset passes validation with zero errors."""
    records, meta = provider.generate_valid_dataset(symbol="SPY", num_bars=25)
    res = validate_records(records, expected_symbol="SPY")

    assert res.is_valid is True
    assert res.error_count == 0
    assert res.total_records == 25
    assert "VALID" in res.format_report()


def test_invalid_ohlc_high_fails_validation(provider: SyntheticDataProvider) -> None:
    """Verify that high < open or high < close triggers validation error."""
    fixtures = provider.generate_anomalous_fixtures()
    res = validate_records(fixtures["invalid_ohlc_high"])

    assert res.is_valid is False
    assert res.error_count >= 1
    codes = [e.code for e in res.errors]
    assert any("ERR_INVALID_OHLC_HIGH" in c for c in codes)


def test_invalid_ohlc_low_fails_validation(provider: SyntheticDataProvider) -> None:
    """Verify that low > open or low > close triggers validation error."""
    fixtures = provider.generate_anomalous_fixtures()
    res = validate_records(fixtures["invalid_ohlc_low"])

    assert res.is_valid is False
    assert res.error_count >= 1
    codes = [e.code for e in res.errors]
    assert any("ERR_INVALID_OHLC_LOW" in c for c in codes)


def test_negative_volume_fails_validation(provider: SyntheticDataProvider) -> None:
    """Verify that volume < 0 triggers ERR_NEGATIVE_VOLUME."""
    fixtures = provider.generate_anomalous_fixtures()
    res = validate_records(fixtures["negative_volume"])

    assert res.is_valid is False
    codes = [e.code for e in res.errors]
    assert "ERR_NEGATIVE_VOLUME" in codes


def test_duplicate_timestamp_detection(provider: SyntheticDataProvider) -> None:
    """Verify that duplicate timestamps for the same symbol are detected."""
    fixtures = provider.generate_anomalous_fixtures()
    res = validate_records(fixtures["duplicate_timestamps"])

    assert res.is_valid is False
    codes = [e.code for e in res.errors]
    assert "ERR_DUPLICATE_TIMESTAMP" in codes


def test_missing_value_detection(provider: SyntheticDataProvider) -> None:
    """Verify that null/missing values trigger ERR_MISSING_VALUE."""
    fixtures = provider.generate_anomalous_fixtures()
    res = validate_records(fixtures["missing_values"])

    assert res.is_valid is False
    codes = [e.code for e in res.errors]
    assert "ERR_MISSING_VALUE" in codes


def test_timestamp_ordering_detection(provider: SyntheticDataProvider) -> None:
    """Verify that unsorted timestamps trigger ERR_UNSORTED_TIMESTAMPS."""
    fixtures = provider.generate_anomalous_fixtures()
    res = validate_records(fixtures["unsorted_timestamps"])

    assert res.is_valid is False
    codes = [e.code for e in res.errors]
    assert "ERR_UNSORTED_TIMESTAMPS" in codes


def test_non_positive_price_detection(provider: SyntheticDataProvider) -> None:
    """Verify that prices <= 0 trigger ERR_NON_POSITIVE_PRICE."""
    fixtures = provider.generate_anomalous_fixtures()
    res = validate_records(fixtures["non_positive_price"])

    assert res.is_valid is False
    codes = [e.code for e in res.errors]
    assert "ERR_NON_POSITIVE_PRICE" in codes


def test_ohlcv_record_invariants_raise_on_construction() -> None:
    """Verify that OHLCVRecord enforces invariants on direct instantiation."""
    now = datetime.now(UTC)

    # Valid construction
    rec = OHLCVRecord(
        timestamp=now,
        symbol="AAPL",
        open=150.0,
        high=155.0,
        low=149.0,
        close=152.0,
        volume=10000.0,
    )
    assert rec.symbol == "AAPL"

    # Naive timestamp should fail
    with pytest.raises(ValueError, match="must be timezone-aware"):
        OHLCVRecord(
            timestamp=datetime(2024, 1, 1),
            symbol="AAPL",
            open=10.0,
            high=12.0,
            low=9.0,
            close=11.0,
            volume=100.0,
        )

    # High < Low should fail
    with pytest.raises(ValueError, match="High"):
        OHLCVRecord(
            timestamp=now,
            symbol="AAPL",
            open=10.0,
            high=8.0,
            low=9.0,
            close=10.0,
            volume=100.0,
        )


def test_normalization_column_synonyms_and_timezones() -> None:
    """Verify that normalizer standardizes column synonyms and parses UTC time."""
    raw_data = [
        {
            "Date": "2024-01-02 09:30:00+00:00",
            "Ticker": "qqq",
            "Open": "380.5",
            "High": "382.0",
            "Low": "379.0",
            "Adj_Close": "381.2",
            "Vol": "2500000",
        },
        {
            "Date": "2024-01-01 09:30:00+00:00",
            "Ticker": "qqq",
            "Open": "375.0",
            "High": "378.0",
            "Low": "374.0",
            "Adj_Close": "377.5",
            "Vol": "2000000",
        },
    ]

    records, logs = normalize_records(raw_data)
    assert len(records) == 2
    # Verify chronological sorting
    assert records[0].timestamp < records[1].timestamp
    assert records[0].symbol == "QQQ"
    assert records[0].open == 375.0
    assert records[0].close == 377.5
    assert records[0].volume == 2000000.0
    assert any("Chronologically sorted" in lg for lg in logs)


def test_normalization_explicit_deduplication() -> None:
    """Verify deduplication behavior: preserved by default, removed when explicit."""
    t = datetime(2024, 1, 1, 10, 0, tzinfo=UTC).isoformat()
    row = {
        "timestamp": t,
        "symbol": "SPY",
        "open": 400.0,
        "high": 402.0,
        "low": 399.0,
        "close": 401.0,
        "volume": 1000.0,
    }
    duped_raw = [row, dict(row)]

    # Default: do NOT silently drop duplicates
    kept_records, _ = normalize_records(
        duped_raw, NormalizationConfig(deduplicate=False)
    )
    assert len(kept_records) == 2

    # Explicit: deduplicate=True drops duplicates and logs audit message
    deduped_records, logs = normalize_records(
        duped_raw, NormalizationConfig(deduplicate=True)
    )
    assert len(deduped_records) == 1
    assert any("dropped 1 duplicate" in lg for lg in logs)


def test_parquet_round_trip_and_storage(
    provider: SyntheticDataProvider, tmp_path: Path
) -> None:
    """Verify saving to Parquet and deserializing back to OHLCVRecord matches."""
    records, meta = provider.generate_valid_dataset(num_bars=15)
    dest_path = tmp_path / "processed" / "SPY_1d.parquet"

    saved = save_dataset(records, dest_path, metadata=meta)
    assert saved.is_file()
    assert dest_path.with_suffix(".meta.json").is_file()

    # Load back
    loaded_records = load_records(saved)
    assert len(loaded_records) == 15
    for orig, loaded in zip(records, loaded_records, strict=True):
        assert orig["symbol"] == loaded.symbol
        assert orig["open"] == loaded.open
        assert orig["high"] == loaded.high
        assert orig["low"] == loaded.low
        assert orig["close"] == loaded.close
        assert orig["volume"] == loaded.volume


def test_duckdb_analytical_query(
    provider: SyntheticDataProvider, tmp_path: Path
) -> None:
    """Verify DuckDB analytical queries execute directly against Parquet files."""
    records, meta = provider.generate_valid_dataset(num_bars=20)
    parquet_path = tmp_path / "test_data.parquet"
    save_dataset(records, parquet_path, metadata=meta)

    # Execute analytical SQL aggregation
    parquet_str = str(parquet_path).replace("\\", "/")
    sql = (
        f"SELECT symbol, COUNT(*) as bar_count, AVG(close) as avg_close, "
        f"MAX(high) as max_high, MIN(low) as min_low "
        f"FROM '{parquet_str}' GROUP BY symbol"
    )
    result = query_dataset(sql)

    assert len(result) == 1
    row = result[0]
    assert row["symbol"] == "SPY"
    assert row["bar_count"] == 20
    assert row["max_high"] >= row["min_low"]


def test_metadata_persistence_and_checksum(tmp_path: Path) -> None:
    """Verify DatasetMetadata serialization and SHA-256 calculation."""
    now = datetime.now(UTC)
    meta = DatasetMetadata(
        provider="test_provider",
        symbol="NVDA",
        timeframe="1h",
        start_time=now,
        end_time=now + timedelta(hours=5),
        record_count=5,
    )
    meta_file = tmp_path / "test_meta.json"
    meta.save(meta_file)

    loaded = DatasetMetadata.load(meta_file)
    assert loaded.provider == "test_provider"
    assert loaded.symbol == "NVDA"
    assert loaded.record_count == 5

    # Checksum calculation on dummy file
    dummy = tmp_path / "dummy.bin"
    dummy.write_bytes(b"deterministic_content")
    sha = calculate_file_sha256(dummy)
    assert isinstance(sha, str)
    assert len(sha) == 64


def test_dataset_inspection(provider: SyntheticDataProvider, tmp_path: Path) -> None:
    """Verify inspect_dataset returns accurate file metrics and column types."""
    records, meta = provider.generate_valid_dataset(num_bars=10)
    p_file = tmp_path / "inspect_test.parquet"
    save_dataset(records, p_file, metadata=meta)

    info = inspect_dataset(p_file)
    assert info["record_count"] == 10
    assert info["symbols"] == ["SPY"]
    assert "open" in info["columns"]
    assert "close" in info["columns"]
    assert info["file_size_bytes"] > 0
    assert info["metadata"]["provider"] == "synthetic_fixture"
