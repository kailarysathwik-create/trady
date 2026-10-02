"""Unit tests for domain interfaces, protocols, and data contracts."""

from datetime import UTC, datetime

import pytest

from trady.data import MarketDataBatch
from trady.features import FeatureMetadata


def test_market_data_batch_validation() -> None:
    """Ensure MarketDataBatch enforces chronological validity."""
    now = datetime.now(UTC)

    # Valid batch
    batch = MarketDataBatch(
        symbol="SPY",
        start_time=now,
        end_time=now,
        record_count=100,
        metadata={"source": "test"},
    )
    assert batch.symbol == "SPY"
    assert batch.record_count == 100

    # Invalid: start_time > end_time
    past = datetime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="cannot exceed end_time"):
        MarketDataBatch(
            symbol="SPY",
            start_time=now,
            end_time=past,
            record_count=50,
            metadata={},
        )

    # Invalid: negative record_count
    with pytest.raises(ValueError, match="non-negative"):
        MarketDataBatch(
            symbol="SPY",
            start_time=now,
            end_time=now,
            record_count=-1,
            metadata={},
        )


def test_feature_metadata_validation() -> None:
    """Ensure FeatureMetadata lookback window is at least 1 bar to prevent lookahead."""
    meta = FeatureMetadata(
        name="SMA_20",
        lookback_window=20,
        description="Simple Moving Average 20 bars",
    )
    assert meta.lookback_window == 20

    with pytest.raises(ValueError, match="at least 1 bar"):
        FeatureMetadata(
            name="INVALID",
            lookback_window=0,
            description="Zero lookback invalid",
        )
