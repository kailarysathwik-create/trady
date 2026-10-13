"""Unit and integration tests for TRADY Phase 3.1: Yahoo Finance Ingestion."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trady.data.ingestion import download_and_ingest_market_data
from trady.data.storage import load_dataset
from trady.data.validator import validate_records
from trady.data.yahoo import YahooFinanceProvider

# -----------------------------------------------------------------------------
# Fixtures & Helpers
# -----------------------------------------------------------------------------


def _create_mock_yfinance_df() -> pd.DataFrame:
    """Create a pristine 5-day daily OHLCV DataFrame mimicking yfinance."""
    dates = pd.date_range("2024-01-08", periods=5, freq="B", tz="UTC")
    data = {
        "Open": [21500.0, 21550.0, 21600.0, 21580.0, 21620.0],
        "High": [21600.0, 21650.0, 21700.0, 21650.0, 21720.0],
        "Low": [21450.0, 21500.0, 21550.0, 21520.0, 21580.0],
        "Close": [21580.0, 21620.0, 21680.0, 21600.0, 21710.0],
        "Volume": [1500000.0, 1600000.0, 1450000.0, 1700000.0, 1800000.0],
    }
    return pd.DataFrame(data, index=dates)


# -----------------------------------------------------------------------------
# 1. Provider Unit Tests (Mocked / Offline)
# -----------------------------------------------------------------------------


def test_yahoo_provider_fetch_raw_success() -> None:
    """Verify YahooFinanceProvider accurately parses yfinance response."""
    provider = YahooFinanceProvider()
    mock_df = _create_mock_yfinance_df()

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_instance

        records, provenance = provider.fetch_raw(
            symbol="^NSEI",
            timeframe="1d",
            start_time="2024-01-08",
            end_time="2024-01-13",
        )

        assert len(records) == 5
        assert records[0]["symbol"] == "^NSEI"
        assert records[0]["open"] == 21500.0
        assert records[0]["high"] == 21600.0
        assert records[0]["low"] == 21450.0
        assert records[0]["close"] == 21580.0
        assert records[0]["volume"] == 1500000.0
        assert "timestamp" in records[0]

        # Verify provenance tracking
        assert provenance["provider"] == "yahoo_finance"
        assert provenance["requested_symbol"] == "^NSEI"
        assert provenance["row_count"] == 5
        assert provenance["bar_interval"] == "1d"
        assert provenance["requested_start_date"] == "2024-01-08"
        assert provenance["requested_end_date"] == "2024-01-13"


def test_yahoo_provider_empty_response_raises() -> None:
    """Verify empty provider response raises descriptive ValueError."""
    provider = YahooFinanceProvider()

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_instance

        with pytest.raises(ValueError, match="No market data returned"):
            provider.fetch_raw(symbol="INVALID_TICKER", start_time="2024-01-01")


def test_yahoo_provider_rate_limit_error() -> None:
    """Verify rate limit exception is captured and re-raised cleanly."""
    provider = YahooFinanceProvider()

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.side_effect = Exception("Too Many Requests (Rate limit)")
        mock_ticker_cls.return_value = mock_instance

        with pytest.raises(RuntimeError, match="rate limit"):
            provider.fetch_raw(symbol="^NSEI")


def test_yahoo_provider_timeout_error() -> None:
    """Verify timeout exception is captured and re-raised cleanly."""
    provider = YahooFinanceProvider()

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.side_effect = Exception("Connection timeout error")
        mock_ticker_cls.return_value = mock_instance

        with pytest.raises(RuntimeError, match="Timeout"):
            provider.fetch_raw(symbol="^NSEI")


# -----------------------------------------------------------------------------
# 2. Ingestion Pipeline & Provenance Tests
# -----------------------------------------------------------------------------


def test_ingestion_pipeline_end_to_end(tmp_path: Path) -> None:
    """Verify end-to-end download, raw archiving, normalization, and Parquet saving."""
    mock_df = _create_mock_yfinance_df()
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_instance

        result = download_and_ingest_market_data(
            symbol="^NSEI",
            start_date="2024-01-08",
            end_date="2024-01-13",
            timeframe="1d",
            raw_dir=raw_dir,
            output_dir=processed_dir,
        )

        assert result["status"] == "success"
        assert result["record_count"] == 5

        # Check raw archive exists
        raw_file = Path(result["raw_file"])
        assert raw_file.exists()
        with open(raw_file, encoding="utf-8") as f:
            raw_content = json.load(f)
            assert "provenance" in raw_content
            assert len(raw_content["raw_records"]) == 5

        # Check processed Parquet exists
        processed_file = Path(result["processed_file"])
        assert processed_file.exists()
        table = load_dataset(processed_file)
        assert table.num_rows == 5
        assert set(table.column_names) == {
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        # Check companion .meta.json exists
        meta_file = Path(result["meta_file"])
        assert meta_file.exists()
        with open(meta_file, encoding="utf-8") as f:
            meta_json = json.load(f)
            assert meta_json["provider"] == "yahoo_finance"
            assert meta_json["symbol"] == "^NSEI"
            assert meta_json["checksum_sha256"] is not None


def test_ingestion_overwrite_guard(tmp_path: Path) -> None:
    """Verify existing files are protected from silent overwrite."""
    mock_df = _create_mock_yfinance_df()
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_instance

        # 1. First run creates files
        download_and_ingest_market_data(
            symbol="SPY",
            start_date="2024-01-08",
            end_date="2024-01-13",
            raw_dir=raw_dir,
            output_dir=processed_dir,
        )

        # 2. Second run without overwrite must raise FileExistsError
        with pytest.raises(FileExistsError, match="already exists"):
            download_and_ingest_market_data(
                symbol="SPY",
                start_date="2024-01-08",
                end_date="2024-01-13",
                raw_dir=raw_dir,
                output_dir=processed_dir,
                overwrite=False,
            )

        # 3. Third run with overwrite=True succeeds
        res = download_and_ingest_market_data(
            symbol="SPY",
            start_date="2024-01-08",
            end_date="2024-01-13",
            raw_dir=raw_dir,
            output_dir=processed_dir,
            overwrite=True,
        )
        assert res["status"] == "success"


def test_ingestion_preview_dry_run(tmp_path: Path) -> None:
    """Verify dry-run preview inspects data without writing files."""
    mock_df = _create_mock_yfinance_df()
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_instance

        res = download_and_ingest_market_data(
            symbol="AAPL",
            start_date="2024-01-08",
            end_date="2024-01-13",
            raw_dir=raw_dir,
            output_dir=processed_dir,
            preview_only=True,
        )

        assert res["status"] == "success"
        assert res["is_preview"] is True
        assert len(res["preview_records"]) == 5
        assert not list(raw_dir.glob("*"))
        assert not list(processed_dir.glob("*"))


def test_ingestion_quarantines_invalid_data(tmp_path: Path) -> None:
    """Verify invalid OHLC data from provider is quarantined and not stored."""
    # Create invalid DataFrame where High < Low
    dates = pd.date_range("2024-01-08", periods=3, freq="B", tz="UTC")
    corrupt_data = {
        "Open": [100.0, 101.0, 102.0],
        "High": [90.0, 95.0, 98.0],  # High lower than Open/Low
        "Low": [110.0, 112.0, 115.0],
        "Close": [105.0, 106.0, 107.0],
        "Volume": [1000.0, 1000.0, 1000.0],
    }
    corrupt_df = pd.DataFrame(corrupt_data, index=dates)

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = corrupt_df
        mock_ticker_cls.return_value = mock_instance

        with pytest.raises(ValueError, match="quarantined"):
            download_and_ingest_market_data(
                symbol="BAD_STOCK",
                raw_dir=tmp_path / "raw",
                output_dir=tmp_path / "processed",
            )

        # Confirm nothing was saved in processed
        assert not (tmp_path / "processed" / "bad_stock_1d.parquet").exists()


# -----------------------------------------------------------------------------
# 3. Calendar Gap & Date Range Verification Tests
# -----------------------------------------------------------------------------


def test_trading_session_gap_detection() -> None:
    """Verify calendar gap heuristic distinguishes normal weekends from dropouts."""
    # 1. Normal trading week + weekend (Monday to Monday): no gap warning
    normal_dates = [
        "2024-01-08T00:00:00+00:00",  # Monday
        "2024-01-09T00:00:00+00:00",  # Tuesday
        "2024-01-10T00:00:00+00:00",  # Wednesday
        "2024-01-11T00:00:00+00:00",  # Thursday
        "2024-01-12T00:00:00+00:00",  # Friday
        "2024-01-15T00:00:00+00:00",  # Monday (3-day weekend)
    ]
    normal_recs = [
        {
            "timestamp": d,
            "symbol": "SPY",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
        }
        for d in normal_dates
    ]
    res_normal = validate_records(normal_recs)
    gap_warns = [w for w in res_normal.warnings if w.code == "WARN_TRADING_SESSION_GAP"]
    assert len(gap_warns) == 0

    # 2. Suspicious 14-day gap (Jan 15 to Jan 29: 10 business days missed)
    gap_recs = list(normal_recs)
    gap_recs.append(
        {
            "timestamp": "2024-01-29T00:00:00+00:00",
            "symbol": "SPY",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
        }
    )
    res_gap = validate_records(gap_recs)
    gap_warns_detected = [
        w for w in res_gap.warnings if w.code == "WARN_TRADING_SESSION_GAP"
    ]
    assert len(gap_warns_detected) == 1
    assert "Suspicious gap" in gap_warns_detected[0].message


def test_date_range_mismatch_warning() -> None:
    """Verify mismatch warning when returned data diverges from requested start."""
    recs = [
        {
            "timestamp": "2024-06-01T00:00:00+00:00",
            "symbol": "SPY",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
        }
    ]
    # Requested start date was Jan 1, but dataset starts in June
    req_start = datetime(2024, 1, 1, tzinfo=UTC)
    res = validate_records(recs, expected_start=req_start)
    mismatch_warns = [w for w in res.warnings if w.code == "WARN_DATE_RANGE_MISMATCH"]
    assert len(mismatch_warns) == 1


# -----------------------------------------------------------------------------
# 4. Optional Live Network Integration Test
# -----------------------------------------------------------------------------


@pytest.mark.network
def test_live_yahoo_finance_download_small_sample(tmp_path: Path) -> None:
    """Live integration test querying a small sample from Yahoo Finance (if online)."""
    import urllib.request

    # Check if network is reachable
    try:
        urllib.request.urlopen("https://finance.yahoo.com", timeout=3.0)
    except Exception:
        pytest.skip("External internet access to Yahoo Finance not available.")

    res = download_and_ingest_market_data(
        symbol="SPY",
        start_date="2024-01-02",
        end_date="2024-01-10",
        raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "processed",
    )

    assert res["status"] == "success"
    assert res["record_count"] >= 4
    assert Path(res["processed_file"]).exists()
