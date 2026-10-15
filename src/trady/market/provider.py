"""Market Quote and Candlestick Provider Service."""

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trady.market.universe import get_instrument_info
from trady.utils.logging import get_logger

logger = get_logger("market.provider")


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


class MarketQuoteService:
    """High-performance market quote and historical candle provider with caching."""

    def __init__(self, cache_ttl_seconds: int = 45) -> None:
        self.cache_ttl_seconds = cache_ttl_seconds
        self._quote_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._candle_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Fetch current market quote for symbol with delay and status metadata."""
        sym = symbol.strip().upper()
        now = time.time()

        # Check memory cache
        if sym in self._quote_cache:
            cache_ts, cached_data = self._quote_cache[sym]
            if (now - cache_ts) < self.cache_ttl_seconds:
                return cached_data

        inst = get_instrument_info(sym)

        # Attempt to fetch via yfinance
        try:
            import yfinance as yf

            ticker = yf.Ticker(sym)
            fast_info = getattr(ticker, "fast_info", None)

            price = None
            prev_close = None
            open_p = None
            high_p = None
            low_p = None
            vol = None

            if fast_info:
                try:
                    price = float(fast_info.last_price)
                    prev_close = float(fast_info.previous_close)
                    open_p = float(fast_info.open)
                    high_p = float(fast_info.day_high)
                    low_p = float(fast_info.day_low)
                    vol = float(fast_info.last_volume)
                except Exception:
                    pass

            if price is None or price <= 0:
                # Fallback to 5-day history
                hist = ticker.history(period="5d", interval="1d")
                if not hist.empty:
                    last_row = hist.iloc[-1]
                    price = float(last_row["Close"])
                    open_p = float(last_row["Open"])
                    high_p = float(last_row["High"])
                    low_p = float(last_row["Low"])
                    vol = float(last_row["Volume"])
                    prev_close = (
                        float(hist.iloc[-2]["Close"]) if len(hist) > 1 else price
                    )

            if price is not None and price > 0:
                prev_c = prev_close or price
                chg = price - prev_c
                chg_pct = (chg / prev_c) if prev_c > 0 else 0.0

                quote = {
                    "symbol": sym,
                    "name": inst.name,
                    "exchange": inst.exchange,
                    "currency": inst.currency.value,
                    "currency_symbol": inst.currency.symbol,
                    "price": round(price, 2),
                    "last_price": round(price, 2),
                    "open": round(open_p or price, 2),
                    "high": round(high_p or price, 2),
                    "low": round(low_p or price, 2),
                    "previous_close": round(prev_c, 2),
                    "volume": int(vol or 0),
                    "change": round(chg, 2),
                    "change_pct": round(chg_pct * 100.0, 2),
                    "data_delay": "Delayed (15m)",
                    "provider": "Yahoo Finance",
                    "status": "AVAILABLE",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                self._quote_cache[sym] = (now, quote)
                return quote

        except Exception as exc:
            logger.debug(
                "Live fetch for %s failed (%s), checking local files", sym, exc
            )

        # Fallback to local offline dataset if present
        # (e.g. data/processed/SPY_1d.parquet)
        local_quote = self._get_local_file_quote(sym, inst)
        if local_quote:
            self._quote_cache[sym] = (now, local_quote)
            return local_quote

        # Unavailable response without fabricating
        return {
            "symbol": sym,
            "name": inst.name,
            "exchange": inst.exchange,
            "currency": inst.currency.value,
            "currency_symbol": inst.currency.symbol,
            "price": 0.0,
            "last_price": 0.0,
            "status": "UNAVAILABLE",
            "message": f"Market quote currently unavailable for symbol '{sym}'.",
            "provider": "Yahoo Finance",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _get_local_file_quote(self, sym: str, inst: Any) -> dict[str, Any] | None:
        """Inspect local datasets for offline test & development environments."""
        candidates = [
            Path(f"data/processed/{sym}_1d.parquet"),
            Path(f"data/processed/{sym.lower()}.parquet"),
            Path("data/processed/SPY_1d.parquet"),
        ]
        for p in candidates:
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    if not df.empty and "close" in df.columns:
                        last = df.iloc[-1]
                        price = float(last["close"])
                        open_p = float(last.get("open", price))
                        high_p = float(last.get("high", price))
                        low_p = float(last.get("low", price))
                        vol = float(last.get("volume", 0.0))
                        prev = float(df.iloc[-2]["close"]) if len(df) > 1 else price
                        chg = price - prev
                        chg_pct = (chg / prev) if prev > 0 else 0.0

                        return {
                            "symbol": sym,
                            "name": inst.name,
                            "exchange": inst.exchange,
                            "currency": inst.currency.value,
                            "currency_symbol": inst.currency.symbol,
                            "price": round(price, 2),
                            "last_price": round(price, 2),
                            "open": round(open_p, 2),
                            "high": round(high_p, 2),
                            "low": round(low_p, 2),
                            "previous_close": round(prev, 2),
                            "volume": int(vol),
                            "change": round(chg, 2),
                            "change_pct": round(chg_pct * 100.0, 2),
                            "data_delay": "Offline Local Cache",
                            "provider": "Local Historical Store",
                            "status": "AVAILABLE",
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                except Exception:
                    pass
        return None

    def get_candles(self, symbol: str, timeframe: str = "1M") -> list[dict[str, Any]]:
        """Fetch historical candles with calculated technical indicators.

        Timeframes:
        - '1D': 1-day intraday (5m)
        - '1W': 5-day intraday (15m)
        - '1M': 1-month daily (1d)
        - '3M': 3-month daily (1d)
        - '1Y': 1-year daily (1d)
        - '5Y': 5-year weekly (1wk)
        """
        sym = symbol.strip().upper()
        cache_key = f"{sym}_{timeframe}"
        now = time.time()

        if cache_key in self._candle_cache:
            cache_ts, cached_candles = self._candle_cache[cache_key]
            if (now - cache_ts) < (self.cache_ttl_seconds * 4):
                return cached_candles

        period_map = {
            "1D": ("1d", "5m"),
            "1W": ("5d", "15m"),
            "1M": ("1mo", "1d"),
            "3M": ("3mo", "1d"),
            "1Y": ("1y", "1d"),
            "5Y": ("5y", "1wk"),
        }
        period, interval = period_map.get(timeframe.upper(), ("1mo", "1d"))

        df: pd.DataFrame | None = None
        try:
            import yfinance as yf

            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval=interval)
        except Exception as exc:
            logger.debug("Failed fetching candles for %s from yfinance: %s", sym, exc)

        # Fallback to local file if empty or failed
        if df is None or df.empty:
            candidates = [
                Path(f"data/processed/{sym}_1d.parquet"),
                Path(f"data/processed/{sym.lower()}.parquet"),
                Path("data/processed/SPY_1d.parquet"),
            ]
            for p in candidates:
                if p.exists():
                    try:
                        df = pd.read_parquet(p)
                        # Normalize columns
                        col_map = {c: c.capitalize() for c in df.columns}
                        df = df.rename(columns=col_map)
                        break
                    except Exception:
                        pass

        if df is None or df.empty:
            return []

        # Ensure required columns
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in df.columns:
                lower_col = col.lower()
                if lower_col in df.columns:
                    df[col] = df[lower_col]
                else:
                    return []

        close_s = df["Close"].astype(float)
        df["SMA20"] = close_s.rolling(window=20, min_periods=1).mean()
        df["SMA50"] = close_s.rolling(window=50, min_periods=1).mean()
        df["EMA20"] = close_s.ewm(span=20, adjust=False).mean()
        df["RSI14"] = compute_rsi(close_s, period=14)

        candles: list[dict[str, Any]] = []
        for idx, row in df.iterrows():
            raw_ts = (
                row.get("Timestamp")
                if "Timestamp" in row
                else row.get("timestamp", None)
            )
            if raw_ts is not None and not pd.isna(raw_ts):
                ts_str = (
                    raw_ts.isoformat() if hasattr(raw_ts, "isoformat") else str(raw_ts)
                )
            elif hasattr(idx, "isoformat"):
                ts_str = idx.isoformat()
            else:
                ts_str = str(idx)

            candles.append(
                {
                    "timestamp": ts_str,
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                    "sma20": round(float(row["SMA20"]), 2),
                    "sma50": round(float(row["SMA50"]), 2),
                    "ema20": round(float(row["EMA20"]), 2),
                    "rsi14": round(float(row["RSI14"]), 1),
                }
            )

        self._candle_cache[cache_key] = (now, candles)
        return candles
