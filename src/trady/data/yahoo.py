"""Yahoo Finance market data provider adapter for TRADY."""

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from trady.data.provider import MarketDataProvider
from trady.utils.logging import get_logger

logger = get_logger("data.yahoo")


class YahooFinanceProvider(MarketDataProvider):
    """Retrieves historical market data from Yahoo Finance via yfinance.

    NOTICE: Yahoo Finance is an educational research and exploratory data source,
    not an official exchange direct-feed. Data may contain revisions, adjustments,
    or provider-side rate limits.
    """

    def __init__(self, request_timeout: float = 30.0) -> None:
        self._provider_name = "yahoo_finance"
        self.request_timeout = request_timeout

    @property
    def provider_name(self) -> str:
        """Name identifier of the data provider source."""
        return self._provider_name

    def fetch_raw(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_time: datetime | date | str | None = None,
        end_time: datetime | date | str | None = None,
        auto_adjust: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g. '^NSEI', 'SPY', 'AAPL').
            timeframe: Bar interval (defaults to '1d').
            start_time: Start date (inclusive).
            end_time: End date (exclusive for yfinance API).
            auto_adjust: Whether to adjust OHLC for splits and dividends.

        Returns:
            Tuple of (raw_record_dicts, provenance_dict).

        Raises:
            ValueError: If symbol or date parameters are invalid, or no data returned.
            RuntimeError: If network or provider rate limit errors occur.
        """
        sym = symbol.strip()
        if not sym:
            raise ValueError("Symbol must be a non-empty string.")

        # Format dates as YYYY-MM-DD
        start_str = self._format_date(start_time)
        end_str = self._format_date(end_time)

        logger.info(
            "Fetching historical data from Yahoo Finance: symbol=%s, interval=%s, "
            "start=%s, end=%s, auto_adjust=%s",
            sym,
            timeframe,
            start_str,
            end_str,
            auto_adjust,
        )

        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(
                start=start_str,
                end=end_str,
                interval=timeframe,
                auto_adjust=auto_adjust,
                timeout=self.request_timeout,
            )
        except Exception as exc:
            msg = str(exc)
            if "rate" in msg.lower() or "too many requests" in msg.lower():
                raise RuntimeError(
                    f"Yahoo Finance rate limit exceeded while querying '{sym}': {exc}"
                ) from exc
            if "timeout" in msg.lower():
                raise RuntimeError(
                    f"Timeout ({self.request_timeout}s) contacting Yahoo Finance for "
                    f"'{sym}': {exc}"
                ) from exc
            raise RuntimeError(
                f"Failed to retrieve data from Yahoo Finance for '{sym}': {exc}"
            ) from exc

        if df is None or df.empty:
            raise ValueError(
                f"No market data returned by Yahoo Finance for symbol '{sym}' "
                f"between {start_str or 'start'} and {end_str or 'end'}."
            )

        # Parse DataFrame rows into raw record dictionaries
        records: list[dict[str, Any]] = []
        has_tz = hasattr(df.index, "tz") and df.index.tz is not None
        original_tz = str(df.index.tz) if has_tz else "None"

        # Ensure index is datetime-like
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Convert index to timezone-aware UTC
        if df.index.tz is None:
            utc_index = df.index.tz_localize("UTC")
        else:
            utc_index = df.index.tz_convert("UTC")

        for dt, row in zip(utc_index, df.itertuples(index=False), strict=False):
            vol = float(row.Volume) if hasattr(row, "Volume") else 0.0
            records.append(
                {
                    "timestamp": dt.isoformat(),
                    "symbol": sym,
                    "open": float(row.Open),
                    "high": float(row.High),
                    "low": float(row.Low),
                    "close": float(row.Close),
                    "volume": vol,
                }
            )

        actual_start = records[0]["timestamp"] if records else None
        actual_end = records[-1]["timestamp"] if records else None

        provenance = {
            "provider": self.provider_name,
            "requested_symbol": sym,
            "returned_symbol": sym,
            "requested_start_date": start_str,
            "requested_end_date": end_str,
            "actual_start_date": actual_start,
            "actual_end_date": actual_end,
            "retrieval_timestamp_utc": datetime.now(UTC).isoformat(),
            "bar_interval": timeframe,
            "source_timezone": original_tz,
            "price_adjustment": "auto_adjusted" if auto_adjust else "unadjusted",
            "schema_version": "1.0.0",
            "row_count": len(records),
        }

        logger.info(
            "Successfully fetched %d bars for %s (actual range: %s to %s)",
            len(records),
            sym,
            actual_start,
            actual_end,
        )

        return records, provenance

    @staticmethod
    def _format_date(val: datetime | date | str | None) -> str | None:
        """Format an input date object or string into YYYY-MM-DD."""
        if val is None:
            return None
        if isinstance(val, (datetime, date)):
            return val.strftime("%Y-%m-%d")
        if isinstance(val, str):
            val_clean = val.strip()
            # If timestamp with time, extract YYYY-MM-DD
            if "T" in val_clean:
                return val_clean.split("T")[0]
            if " " in val_clean:
                return val_clean.split(" ")[0]
            return val_clean
        raise ValueError(f"Unsupported date format: {type(val)} ({val})")
