"""Unit tests for TRADY Market Intelligence, Universe, Quotes, and Forecasting."""

import pytest

from trady.market.forecasting import RiskProbabilityEngine
from trady.market.provider import MarketQuoteService
from trady.market.universe import (
    CANONICAL_UNIVERSE,
    get_instrument_info,
    search_universe,
)
from trady.portfolio.models import Currency


def test_universe_search_and_metadata() -> None:
    """Verify searchable multi-market universe covers US and Indian instruments."""
    assert len(CANONICAL_UNIVERSE) >= 15

    # US symbol lookup
    spy = get_instrument_info("SPY")
    assert spy.currency == Currency.USD
    assert spy.exchange in ("NYSE Arca", "NYSE")

    # Indian symbol lookup
    infy = get_instrument_info("INFY.NS")
    assert infy.currency == Currency.INR
    assert infy.exchange == "NSE"

    # Search queries
    res_apple = search_universe("Apple")
    assert len(res_apple) >= 1
    assert res_apple[0]["symbol"] == "AAPL"

    res_tcs = search_universe("TCS")
    assert len(res_tcs) >= 1
    assert res_tcs[0]["symbol"] == "TCS.NS"


def test_market_quote_fetch_and_metadata() -> None:
    """Verify market quote service returns structured quote with provenance."""
    svc = MarketQuoteService(cache_ttl_seconds=60)
    quote = svc.get_quote("SPY")

    assert quote["symbol"] == "SPY"
    assert quote["currency"] == "USD"
    assert quote["currency_symbol"] == "$"
    assert "status" in quote
    assert "data_delay" in quote
    assert "provider" in quote
    assert "timestamp" in quote
    if quote["status"] == "AVAILABLE":
        assert quote["price"] > 0.0
        assert quote["open"] > 0.0


def test_candlestick_generation_and_technical_indicators() -> None:
    """Verify historical candles compute SMA20, SMA50, EMA20, and RSI14."""
    svc = MarketQuoteService()
    candles = svc.get_candles("SPY", timeframe="1M")

    assert len(candles) > 0
    c0 = candles[-1]
    assert "open" in c0
    assert "high" in c0
    assert "low" in c0
    assert "close" in c0
    assert "volume" in c0
    assert "sma20" in c0
    assert "sma50" in c0
    assert "ema20" in c0
    assert "rsi14" in c0
    assert 0.0 <= c0["rsi14"] <= 100.0


def test_risk_probability_engine_forecasting_and_risk_ratings() -> None:
    """Verify RiskProbabilityEngine computes directional probability and
    transparent risk ratings.
    """
    quote_svc = MarketQuoteService()
    engine = RiskProbabilityEngine(quote_service=quote_svc)

    result = engine.analyze_next_move(
        symbol="SPY", horizon_days=5, model_type="logistic_regression"
    )

    assert "status" in result
    assert "market_risk" in result
    risk = result["market_risk"]
    assert risk["risk_rating"] in (
        "LOW",
        "MODERATE",
        "HIGH",
        "VERY HIGH",
        "INSUFFICIENT EVIDENCE",
    )
    assert risk["annualized_volatility"] >= 0.0
    assert risk["volatility_regime"] in ("LOW", "NORMAL", "HIGH", "EXTREME")

    if result["status"] == "VALIDATED":
        f = result["forecast"]
        assert f["predicted_direction"] in ("UPWARD", "DOWNWARD", "NEUTRAL")
        assert 0.0 <= f["probability_up"] <= 100.0
        assert 0.0 <= f["probability_down"] <= 100.0
        assert pytest.approx(f["probability_up"] + f["probability_down"]) == 100.0
        assert "model_limitations" in result
        assert len(result["model_limitations"]) >= 2


def test_risk_probability_engine_unavailable_model_fails_honestly() -> None:
    """Verify that an untrained horizon (e.g. 99d) reports MODEL_UNAVAILABLE
    instead of fabricating.
    """
    engine = RiskProbabilityEngine()
    result = engine.analyze_next_move(
        symbol="SPY", horizon_days=99, model_type="logistic_regression"
    )

    assert result["status"] == "MODEL_UNAVAILABLE"
    assert "No serialized model found" in result["message"]
    # Still provides real market volatility
    assert "volatility" in result
