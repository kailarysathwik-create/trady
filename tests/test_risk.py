"""Deterministic unit tests for the TRADY Risk Engine and its constraints."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trady.backtesting.config import BacktestConfig
from trady.backtesting.engine import BacktestEngine
from trady.backtesting.portfolio import Position
from trady.backtesting.reporting import save_backtest_artifacts
from trady.backtesting.strategy import BaseStrategy, Order
from trady.risk.config import RiskConfig
from trady.risk.decision import RiskDecision
from trady.risk.engine import RiskEngine
from trady.risk.sizing import PositionSizer


def test_max_position_exposure_reduces_and_rejects() -> None:
    """Test that max_position_exposure strictly caps single-position dollar value."""
    config = RiskConfig(
        max_position_exposure=10_000.0,
        max_position_concentration=1.0,
        max_portfolio_exposure=2.0,
    )
    engine = RiskEngine(config=config)
    ts = datetime(2025, 1, 1, tzinfo=UTC)

    # 1. Proposed: BUY 150 shares @ $100 = $15,000 (exceeds $10,000 limit)
    order = Order(symbol="XYZ", action="BUY", quantity=150.0)
    approved_order, decision = engine.evaluate_order(
        order=order,
        timestamp=ts,
        current_price=100.0,
        current_equity=100_000.0,
        current_cash=100_000.0,
        current_drawdown=0.0,
        existing_positions={},
    )

    assert decision.status == "REDUCED"
    assert decision.constraint_triggered == "max_position_exposure"
    assert decision.approved_quantity == pytest.approx(100.0)
    assert decision.approved_value == pytest.approx(10_000.0)
    assert approved_order is not None
    assert approved_order.quantity == pytest.approx(100.0)
    assert "capped by max pos exposure" in decision.reason

    # 2. Existing: already holding 100 shares @ $100 ($10,000). Propose: BUY 20 shares.
    existing = {
        "XYZ": Position(
            symbol="XYZ",
            quantity=100.0,
            avg_entry_price=100.0,
            entry_time=ts,
            entry_bar_idx=0,
            entry_fee=1.0,
        )
    }
    order2 = Order(symbol="XYZ", action="BUY", quantity=20.0)
    approved2, decision2 = engine.evaluate_order(
        order=order2,
        timestamp=ts,
        current_price=100.0,
        current_equity=100_000.0,
        current_cash=90_000.0,
        current_drawdown=0.0,
        existing_positions=existing,
    )

    assert decision2.status == "REJECTED"
    assert decision2.constraint_triggered == "max_position_exposure"
    assert decision2.approved_quantity == 0.0
    assert approved2 is None
    assert "already at or exceeds" in decision2.reason


def test_max_position_concentration_constraint() -> None:
    """Test that max_position_concentration caps asset as a % of equity."""
    # Max 20% of equity in single asset
    config = RiskConfig(
        max_position_concentration=0.20,
        max_portfolio_exposure=2.0,
    )
    engine = RiskEngine(config=config)
    ts = datetime(2025, 1, 1, tzinfo=UTC)

    # Equity = $50,000. Max allowed in XYZ = $10,000 (200 shares @ $50).
    # Proposed: BUY 400 shares @ $50 ($20,000)
    order = Order(symbol="XYZ", action="BUY", quantity=400.0)
    approved, decision = engine.evaluate_order(
        order=order,
        timestamp=ts,
        current_price=50.0,
        current_equity=50_000.0,
        current_cash=50_000.0,
        current_drawdown=0.0,
        existing_positions={},
    )

    assert decision.status == "REDUCED"
    assert decision.constraint_triggered == "max_position_concentration"
    assert decision.approved_quantity == pytest.approx(200.0)
    assert decision.approved_value == pytest.approx(10_000.0)
    assert approved is not None
    assert approved.quantity == pytest.approx(200.0)


def test_max_portfolio_exposure_aggregate_leverage() -> None:
    """Test that max_portfolio_exposure caps total gross portfolio exposure."""
    # 100% unleveraged max gross portfolio exposure
    config = RiskConfig(
        max_portfolio_exposure=1.0,
        max_position_concentration=1.0,
        cash_buffer_pct=0.0,
    )
    engine = RiskEngine(config=config)
    ts = datetime(2025, 1, 1, tzinfo=UTC)

    # Equity = $100,000. Existing position in ABC: $70,000 (70% exposure).
    # Remaining capacity: $30,000.
    existing = {
        "ABC": Position(
            symbol="ABC",
            quantity=700.0,
            avg_entry_price=100.0,
            entry_time=ts,
            entry_bar_idx=0,
            entry_fee=1.0,
        )
    }

    # Propose: BUY 500 shares in XYZ @ $100 = $50,000
    order = Order(symbol="XYZ", action="BUY", quantity=500.0)
    approved, decision = engine.evaluate_order(
        order=order,
        timestamp=ts,
        current_price=100.0,
        current_equity=100_000.0,
        current_cash=30_000.0,
        current_drawdown=0.0,
        existing_positions=existing,
    )

    assert decision.status == "REDUCED"
    assert decision.constraint_triggered == "max_portfolio_exposure"
    assert decision.approved_quantity == pytest.approx(300.0)
    assert decision.approved_value == pytest.approx(30_000.0)


def test_max_simulated_drawdown_circuit_breaker() -> None:
    """Test that max_drawdown_limit blocks new risk but allows exits."""
    # 10% drawdown circuit breaker
    config = RiskConfig(max_drawdown_limit=0.10)
    engine = RiskEngine(config=config)
    ts = datetime(2025, 1, 1, tzinfo=UTC)

    # 1. In 12% drawdown: BUY order MUST be rejected
    buy_order = Order(symbol="XYZ", action="BUY", quantity=50.0)
    approved_buy, decision_buy = engine.evaluate_order(
        order=buy_order,
        timestamp=ts,
        current_price=100.0,
        current_equity=88_000.0,
        current_cash=88_000.0,
        current_drawdown=0.12,  # 12% drawdown > 10% limit
        existing_positions={},
    )
    assert decision_buy.status == "REJECTED"
    assert decision_buy.constraint_triggered == "max_drawdown_limit"
    assert approved_buy is None
    assert "breached limit" in decision_buy.reason

    # 2. In 12% drawdown: SELL order (reducing risk) MUST be accepted
    existing = {
        "XYZ": Position(
            symbol="XYZ",
            quantity=50.0,
            avg_entry_price=100.0,
            entry_time=ts,
            entry_bar_idx=0,
            entry_fee=1.0,
        )
    }
    sell_order = Order(symbol="XYZ", action="SELL", quantity=50.0)
    approved_sell, decision_sell = engine.evaluate_order(
        order=sell_order,
        timestamp=ts,
        current_price=100.0,
        current_equity=88_000.0,
        current_cash=88_000.0,
        current_drawdown=0.12,
        existing_positions=existing,
    )
    assert decision_sell.status == "ACCEPTED"
    assert approved_sell is not None
    assert approved_sell.quantity == 50.0


def test_position_sizing_rules() -> None:
    """Test fractional equity and fixed cash position sizing calculations."""
    # 1. Fractional equity sizing (10% of equity)
    cfg_frac = RiskConfig(sizing_method="fractional_equity", target_position_pct=0.10)
    sizer_frac = PositionSizer(cfg_frac)
    qty_frac = sizer_frac.calculate_size(price=50.0, equity=100_000.0, cash=50_000.0)
    # Target value: $10,000 -> 200 shares
    assert qty_frac == pytest.approx(200.0)

    # 2. Fixed cash sizing ($5,000 fixed allocation)
    cfg_cash = RiskConfig(sizing_method="fixed_cash", fixed_cash_amount=5_000.0)
    sizer_cash = PositionSizer(cfg_cash)
    qty_cash = sizer_cash.calculate_size(price=25.0, equity=100_000.0, cash=50_000.0)
    # $5,000 / $25 = 200 shares
    assert qty_cash == pytest.approx(200.0)


def test_volatility_based_sizing_atr_and_vol_target() -> None:
    """Test that ATR risk and volatility targeting scale inversely with volatility."""
    # 1. ATR Risk Sizing:
    # Equity: $100,000, 1% risk = $1,000 risk capital.
    # atr_multiplier = 2.0
    cfg_atr = RiskConfig(
        sizing_method="atr_risk",
        target_risk_pct=0.01,
        atr_multiplier=2.0,
    )
    sizer_atr = PositionSizer(cfg_atr)

    # Low Vol: ATR = $2.0 -> stop = $4.0 -> Q = 1000 / 4 = 250 shares
    qty_low_vol = sizer_atr.calculate_size(
        price=100.0, equity=100_000.0, cash=100_000.0, atr=2.0
    )
    assert qty_low_vol == pytest.approx(250.0)

    # High Vol: ATR = $5.0 -> stop = $10.0 -> Q = 1000 / 10 = 100 shares
    qty_high_vol = sizer_atr.calculate_size(
        price=100.0, equity=100_000.0, cash=100_000.0, atr=5.0
    )
    assert qty_high_vol == pytest.approx(100.0)
    # Higher volatility strictly decreases position size
    assert qty_high_vol < qty_low_vol

    # 2. Volatility Parity / Target Volatility:
    cfg_vol = RiskConfig(
        sizing_method="volatility_target",
        target_volatility=0.15,
        max_position_concentration=0.50,
    )
    sizer_vol = PositionSizer(cfg_vol)
    # Asset vol = 0.15 -> weight = 1.0 (capped at 0.50 max concentration)
    qty_norm = sizer_vol.calculate_size(
        price=100.0, equity=100_000.0, cash=100_000.0, volatility=0.15
    )
    assert qty_norm == pytest.approx(500.0)  # 50% of 100k = 50k / 100 = 500

    # Asset vol = 0.30 -> weight = 0.15 / 0.30 = 0.50 -> 50%
    qty_double_vol = sizer_vol.calculate_size(
        price=100.0, equity=100_000.0, cash=100_000.0, volatility=0.60
    )
    # 0.15 / 0.60 = 0.25 -> 25% of 100k = 25k / 100 = 250
    assert qty_double_vol == pytest.approx(250.0)


def test_transparent_decision_records_audit_trail() -> None:
    """Verify that RiskDecision records contain provenance and serialize cleanly."""
    ts = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    decision = RiskDecision(
        decision_id="risk_test01",
        timestamp=ts,
        symbol="SPY",
        action="BUY",
        status="REDUCED",
        requested_quantity=100.0,
        approved_quantity=60.0,
        requested_value=10_000.0,
        approved_value=6_000.0,
        price=100.0,
        reason="Reduced: capped by max concentration limit.",
        constraint_triggered="max_position_concentration",
        current_equity=50_000.0,
        current_drawdown=0.04,
        portfolio_exposure_pct=0.50,
        position_concentration_pct=0.12,
    )

    assert decision.is_reduced is True
    assert decision.is_accepted is False
    assert decision.is_rejected is False

    d = decision.to_dict()
    assert d["status"] == "REDUCED"
    assert d["approved_quantity"] == 60.0
    assert d["constraint_triggered"] == "max_position_concentration"
    assert d["timestamp"] == str(ts)


class AggressiveStrategy(BaseStrategy):
    """Strategy proposing an oversized order to test RiskEngine intercept."""

    def evaluate(
        self,
        bar_idx: int,
        bar_data: dict,
        prediction: float | None,
        position: Any,
        cash: float,
        current_equity: float,
    ) -> Order | None:
        if bar_idx == 0:
            # Propose massive order: 50k shares @ $100 = $5M (cash is 100k)
            return Order(symbol="SYNTH", action="BUY", quantity=50_000.0)
        return None


def test_zero_bypass_integration_with_backtest_engine(tmp_path: Path) -> None:
    """Test that BacktestEngine forces all strategy orders through RiskEngine."""
    base_time = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    records = []
    for i in range(5):
        records.append(
            {
                "timestamp": base_time + timedelta(days=i),
                "symbol": "SYNTH",
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1_000_000.0,
            }
        )
    bars = pd.DataFrame(records)

    # Restrict max concentration to 20% ($20,000 = 200 shares)
    risk_cfg = RiskConfig(
        max_position_concentration=0.20,
        max_portfolio_exposure=1.0,
    )
    risk_engine = RiskEngine(config=risk_cfg)

    bt_cfg = BacktestConfig(
        initial_cash=100_000.0,
        commission_bps=0.0,
        fixed_fee_per_order=0.0,
        slippage_bps=0.0,
    )

    engine = BacktestEngine(config=bt_cfg, risk_engine=risk_engine)
    strategy = AggressiveStrategy(config=bt_cfg)

    result = engine.run(data=bars, strategy=strategy, close_positions_at_end=False)

    # 1. Ensure Risk Engine intercepted the 50,000 share order
    assert len(result.risk_decisions) >= 1
    decision = result.risk_decisions[0]
    assert decision.status == "REDUCED"
    assert decision.requested_quantity == 50_000.0
    # Must be reduced to max concentration limit (200 shares)
    assert decision.approved_quantity == pytest.approx(200.0)

    # 2. Check risk decisions DataFrame
    risk_df = result.to_risk_decisions_dataframe()
    assert len(risk_df) >= 1
    assert "status" in risk_df.columns
    assert risk_df.iloc[0]["status"] == "REDUCED"

    # 3. Check artifact persistence
    artifacts = save_backtest_artifacts(result=result, output_dir=tmp_path)
    assert "risk_decisions.parquet" in artifacts
    assert artifacts["risk_decisions.parquet"].is_file()
    saved_risk_df = pd.read_parquet(artifacts["risk_decisions.parquet"])
    assert len(saved_risk_df) == len(risk_df)
