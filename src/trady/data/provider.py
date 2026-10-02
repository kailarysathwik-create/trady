"""Provider abstractions and deterministic synthetic market data generators."""

import random
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from trady.data.metadata import DatasetMetadata


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol defining the interface for historical data ingestion providers."""

    @property
    def provider_name(self) -> str:
        """Name of the data provider source."""
        ...

    def fetch_raw(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch raw market data records as dictionaries."""
        ...


class SyntheticDataProvider:
    """Deterministic generator of synthetic market data for offline testing.

    Requires zero external network access and zero API keys.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.provider_name = "synthetic_fixture"

    def generate_valid_dataset(
        self,
        symbol: str = "SPY",
        timeframe: str = "1d",
        num_bars: int = 30,
        start_time: datetime | None = None,
        base_price: float = 100.0,
    ) -> tuple[list[dict[str, Any]], DatasetMetadata]:
        """Generate a pristine, strictly valid synthetic OHLCV dataset."""
        rng = random.Random(self.seed)
        if start_time is None:
            start_time = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)

        records: list[dict[str, Any]] = []
        current_price = base_price
        current_time = start_time

        delta = timedelta(days=1) if timeframe == "1d" else timedelta(hours=1)

        for _ in range(num_bars):
            # Deterministic price oscillation
            pct_change = rng.uniform(-0.02, 0.02)
            open_p = round(current_price, 2)
            close_p = round(open_p * (1.0 + pct_change), 2)
            high_wiggle = rng.uniform(0.001, 0.015)
            low_wiggle = rng.uniform(0.001, 0.015)

            high_p = round(max(open_p, close_p) * (1.0 + high_wiggle), 2)
            low_p = round(min(open_p, close_p) * (1.0 - low_wiggle), 2)

            # Ensure strict high/low bounds
            high_p = max(high_p, open_p, close_p)
            low_p = min(low_p, open_p, close_p)
            vol = round(rng.uniform(100000, 5000000), 0)

            records.append(
                {
                    "timestamp": current_time.isoformat(),
                    "symbol": symbol,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": vol,
                }
            )

            current_price = close_p
            current_time += delta

        end_time = current_time - delta
        metadata = DatasetMetadata(
            provider=self.provider_name,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            timezone="UTC",
            record_count=len(records),
            extra={"seed": self.seed, "synthetic": True},
        )

        return records, metadata

    def generate_anomalous_fixtures(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
        """Generate test suites with isolated anomalies for testing Validator."""
        base_records, _ = self.generate_valid_dataset(num_bars=10)

        # 1. Duplicate timestamp
        dup_records = [dict(r) for r in base_records]
        dup_records.append(dict(base_records[2]))  # duplicate of row 2

        # 2. Missing values
        missing_val_records = [dict(r) for r in base_records]
        missing_val_records[3]["close"] = None

        # 3. Invalid OHLC: high < open
        bad_high_records = [dict(r) for r in base_records]
        bad_high_records[1]["high"] = bad_high_records[1]["open"] - 10.0

        # 4. Invalid OHLC: low > close
        bad_low_records = [dict(r) for r in base_records]
        bad_low_records[4]["low"] = bad_low_records[4]["close"] + 15.0

        # 5. Negative volume
        neg_vol_records = [dict(r) for r in base_records]
        neg_vol_records[2]["volume"] = -500.0

        # 6. Unsorted timestamps
        unsorted_records = [dict(r) for r in base_records]
        unsorted_records[5], unsorted_records[6] = (
            unsorted_records[6],
            unsorted_records[5],
        )

        # 7. Non-positive price
        zero_price_records = [dict(r) for r in base_records]
        zero_price_records[3]["open"] = 0.0

        # 8. Malformed timestamp
        malformed_ts_records = [dict(r) for r in base_records]
        malformed_ts_records[0]["timestamp"] = "invalid-date-string"

        return {
            "valid": base_records,
            "duplicate_timestamps": dup_records,
            "missing_values": missing_val_records,
            "invalid_ohlc_high": bad_high_records,
            "invalid_ohlc_low": bad_low_records,
            "negative_volume": neg_vol_records,
            "unsorted_timestamps": unsorted_records,
            "non_positive_price": zero_price_records,
            "malformed_timestamp": malformed_ts_records,
        }
