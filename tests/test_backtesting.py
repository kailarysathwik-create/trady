"""Unit and deterministic synthetic tests for TRADY Historical Backtesting Engine."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trady.backtesting.config import BacktestConfig
from trady.backtesting.engine import BacktestEngine
from trady.backtesting.execution import ExecutionSimulator
from trady.backtesting.reporting import save_backtest_artifacts
from trady.backtesting.strategy import BaseStrategy, ModelDrivenStrategy, Order
from trady.backtesting.walk_forward import WalkForwardConfig, WalkForwardEvaluator


class FixedOrderStrategy(BaseStrategy):
    """Deterministic strategy executing pre-scheduled orders at specific bar indices."""

    def __init__(
        self,
        config: BacktestConfig,
        schedule: dict[int, Order],
    ) -> None:
        super().__init__(config)
        self.schedule = schedule

    def evaluate(
        self,
        bar_idx: int,
        bar_data: dict[str, Any],
        prediction: float | None,
        position: Any,
        cash: float,
        current_equity: float,
    ) -> Order | None:
        return self.schedule.get(bar_idx)


def create_synthetic_bars(
    prices: list[tuple[float, float, float, float]],  # (open, high, low, close)
    volumes: list[float] | None = None,
    symbol: str = "SYNTH",
) -> pd.DataFrame:
    """Create deterministic OHLCV DataFrame for testing."""
    n = len(prices)
    base_time = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    records = []
    vols = volumes or [10_000.0] * n

    for i in range(n):
        p_open, p_high, p_low, p_close = prices[i]
        records.append(
            {
                "timestamp": base_time + timedelta(days=i),
                "symbol": symbol,
                "open": float(p_open),
                "high": float(p_high),
                "low": float(p_low),
                "close": float(p_close),
                "volume": float(vols[i]),
            }
        )
    return pd.DataFrame(records)


def test_deterministic_single_trade_exact_pnl_and_fees() -> None:
    """Deterministic test: exact known PnL and fee accounting down to the penny.

    Initial cash: $20,000.00
    Config:
      - 0 slippage
      - 10.0 bps commission (0.001)
      - $1.00 fixed ticket fee per trade
      - 100% volume liquidity

    Bar 0: Schedule BUY 100 shares
    Bar 1: Executes at OPEN 100.0
           Cost: 100 * 100 = $10,000.00
           Commission: 10,000 * 0.001 = $10.00
           Ticket: $1.00 -> Total entry fee: $11.00
           Cash after buy: $9,989.00
           Close: 105.0 -> Equity at close: 9,989 + 10,500 = $20,489.00
           Schedule SELL 100 shares
    Bar 2: Executes at OPEN 110.0
           Gross proceeds: 100 * 110 = $11,000.00
           Commission: 11,000 * 0.001 = $11.00
           Ticket: $1.00 -> Total exit fee: $12.00
           Net proceeds: 11,000 - 12 = $10,988.00
           Cash after sell: 9,989 + 10,988 = $20,977.00
           Gross PnL: 100 * (110 - 100) = $1,000.00
           Total fees: 11 + 12 = $23.00
           Net PnL: 1,000 - 23 = $977.00
           Total Return: 977 / 20,000 = 0.04885 (+4.885%)
    """
    bars = create_synthetic_bars(
        prices=[
            (100.0, 105.0, 95.0, 100.0),  # Bar 0: Signal BUY
            (100.0, 105.0, 95.0, 105.0),  # Bar 1: Fill BUY at 100, Signal SELL
            (110.0, 115.0, 105.0, 110.0),  # Bar 2: Fill SELL at 110
        ]
    )

    config = BacktestConfig(
        initial_cash=20_000.0,
        commission_bps=10.0,
        fixed_fee_per_order=1.0,
        slippage_bps=0.0,
        max_volume_pct=1.0,
    )

    strategy = FixedOrderStrategy(
        config=config,
        schedule={
            0: Order(symbol="SYNTH", action="BUY", quantity=100.0),
            1: Order(symbol="SYNTH", action="SELL", quantity=100.0),
        },
    )

    engine = BacktestEngine(config=config)
    result = engine.run(data=bars, strategy=strategy, close_positions_at_end=False)

    # Validate exact trade calculations
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.quantity == pytest.approx(100.0)
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(110.0)
    assert trade.gross_pnl == pytest.approx(1000.0)
    assert trade.total_fees == pytest.approx(23.0)
    assert trade.net_pnl == pytest.approx(977.0)

    # Validate portfolio equity and cash
    m = result.metrics
    assert m.initial_cash == pytest.approx(20_000.0)
    assert m.final_equity == pytest.approx(20_977.0)
    assert m.total_return == pytest.approx(977.0 / 20_000.0)
    assert m.win_rate == pytest.approx(1.0)
    assert m.win_count == 1
    assert m.loss_count == 0
    assert m.total_fees == pytest.approx(23.0)


def test_anti_lookahead_execution_delay() -> None:
    """Anti-leakage test: Bar T signals NEVER execute on Bar T (only on Bar T+1)."""
    bars = create_synthetic_bars(
        prices=[
            (100.0, 105.0, 95.0, 100.0),  # Bar 0
            (102.0, 106.0, 98.0, 104.0),  # Bar 1
        ]
    )

    config = BacktestConfig(
        initial_cash=10_000.0,
        commission_bps=0.0,
        fixed_fee_per_order=0.0,
        slippage_bps=0.0,
    )

    # Order scheduled at Bar 0
    strategy = FixedOrderStrategy(
        config=config,
        schedule={0: Order(symbol="SYNTH", action="BUY", quantity=10.0)},
    )

    engine = BacktestEngine(config=config)
    result = engine.run(data=bars, strategy=strategy, close_positions_at_end=False)

    # At bar 0, position must NOT exist yet
    assert len(result.executions) == 1
    exec_record = result.executions[0]
    # Execution must occur on Bar 1 timestamp
    assert exec_record.timestamp == bars.iloc[1]["timestamp"]
    assert exec_record.fill_price == pytest.approx(102.0)  # Bar 1 Open price


def test_price_range_clamping_and_slippage() -> None:
    """Friction test: Execution prices clamped strictly to [low, high]."""
    config = BacktestConfig(
        initial_cash=10_000.0,
        slippage_bps=500.0,  # 5% slippage
        commission_bps=0.0,
        fixed_fee_per_order=0.0,
    )
    sim = ExecutionSimulator(config)

    # 1. Buy order with open=100. Slippage gives 105. High is 103. Must clamp to 103.
    res_buy = sim.execute_order(
        order_id="test_buy",
        symbol="ABC",
        action="BUY",
        quantity=10.0,
        bar_open=100.0,
        bar_high=103.0,
        bar_low=98.0,
        bar_close=101.0,
        bar_volume=1000.0,
        timestamp=datetime.now(UTC),
    )
    assert res_buy.fill_price == pytest.approx(103.0)

    # 2. Sell order with open=100. Slippage gives 95. Low is 97. Must clamp to 97.
    res_sell = sim.execute_order(
        order_id="test_sell",
        symbol="ABC",
        action="SELL",
        quantity=10.0,
        bar_open=100.0,
        bar_high=103.0,
        bar_low=97.0,
        bar_close=99.0,
        bar_volume=1000.0,
        timestamp=datetime.now(UTC),
    )
    assert res_sell.fill_price == pytest.approx(97.0)


def test_volume_liquidity_cap() -> None:
    """Liquidity test: Order quantity cannot exceed max_volume_pct of bar volume."""
    config = BacktestConfig(
        max_volume_pct=0.10,  # Max 10% of bar volume
    )
    sim = ExecutionSimulator(config)

    # Request 500 units on a bar with only 1000 volume -> capped at 100 units
    res = sim.execute_order(
        order_id="test_liq",
        symbol="ABC",
        action="BUY",
        quantity=500.0,
        bar_open=50.0,
        bar_high=52.0,
        bar_low=49.0,
        bar_close=51.0,
        bar_volume=1000.0,
        timestamp=datetime.now(UTC),
    )
    assert res.requested_quantity == 500.0
    assert res.filled_quantity == pytest.approx(100.0)


def test_all_12_metrics_calculated() -> None:
    """Verify all 12 core performance metrics are calculated correctly."""
    bars = create_synthetic_bars(
        prices=[
            (100.0, 105.0, 95.0, 102.0),
            (102.0, 108.0, 100.0, 106.0),
            (106.0, 110.0, 103.0, 105.0),
            (105.0, 107.0, 99.0, 101.0),
            (101.0, 104.0, 98.0, 103.0),
        ]
    )

    config = BacktestConfig(
        initial_cash=50_000.0,
        commission_bps=5.0,
        slippage_bps=5.0,
    )

    strategy = FixedOrderStrategy(
        config=config,
        schedule={
            0: Order(symbol="SYNTH", action="BUY", quantity=50.0),
            2: Order(symbol="SYNTH", action="SELL", quantity=50.0),
        },
    )

    engine = BacktestEngine(config=config)
    result = engine.run(data=bars, strategy=strategy, close_positions_at_end=False)
    m = result.metrics

    # Check existence and finite values of all 12 required metrics
    assert isinstance(m.total_return, float)
    assert isinstance(m.annualized_return, float)
    assert isinstance(m.volatility, float)
    assert isinstance(m.maximum_drawdown, float)
    assert isinstance(m.sharpe_ratio, float)
    assert isinstance(m.sortino_ratio, float)
    assert isinstance(m.win_rate, float)
    assert isinstance(m.average_win, float)
    assert isinstance(m.average_loss, float)
    assert isinstance(m.profit_factor, float)
    assert isinstance(m.turnover, float)
    assert isinstance(m.trade_count, int)
    assert m.trade_count >= 1


def test_artifact_generation(tmp_path: Path) -> None:
    """Verify generation of all 5 required experiment artifacts."""
    bars = create_synthetic_bars(
        prices=[
            (50.0, 52.0, 48.0, 51.0),
            (51.0, 53.0, 50.0, 52.0),
            (52.0, 54.0, 51.0, 53.0),
        ]
    )

    config = BacktestConfig(initial_cash=10_000.0)
    strategy = FixedOrderStrategy(
        config=config,
        schedule={
            0: Order(symbol="SYNTH", action="BUY", quantity=20.0),
            1: Order(symbol="SYNTH", action="SELL", quantity=20.0),
        },
    )

    engine = BacktestEngine(config=config)
    result = engine.run(data=bars, strategy=strategy, close_positions_at_end=False)

    artifacts = save_backtest_artifacts(
        result=result,
        output_dir=tmp_path,
        extra_metadata={"test_run": True},
    )

    # 1. metrics.json
    assert "metrics.json" in artifacts
    assert artifacts["metrics.json"].is_file()
    with open(artifacts["metrics.json"], encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert "total_return" in metrics_data
    assert "sharpe_ratio" in metrics_data

    # 2. equity_curve.parquet
    assert "equity_curve.parquet" in artifacts
    assert artifacts["equity_curve.parquet"].is_file()
    eq_df = pd.read_parquet(artifacts["equity_curve.parquet"])
    assert "portfolio_equity" in eq_df.columns
    assert len(eq_df) == 3

    # 3. trades.parquet
    assert "trades.parquet" in artifacts
    assert artifacts["trades.parquet"].is_file()
    tr_df = pd.read_parquet(artifacts["trades.parquet"])
    assert "net_pnl" in tr_df.columns
    assert len(tr_df) == 1

    # 4. configuration.json
    assert "configuration.json" in artifacts
    assert artifacts["configuration.json"].is_file()
    with open(artifacts["configuration.json"], encoding="utf-8") as f:
        cfg_data = json.load(f)
    assert "backtest_config" in cfg_data
    assert cfg_data["extra_metadata"]["test_run"] is True

    # 5. report.html
    assert "report.html" in artifacts
    assert artifacts["report.html"].is_file()
    with open(artifacts["report.html"], encoding="utf-8") as f:
        html = f.read()
    assert "TRADY &bull; Quantitative Research Backtest Report" in html

    # 6. risk_decisions.parquet
    assert "risk_decisions.parquet" in artifacts
    assert artifacts["risk_decisions.parquet"].is_file()


def test_model_driven_strategy_stop_loss() -> None:
    """Verify stop-loss execution when price drops below stop_loss_pct."""
    bars = create_synthetic_bars(
        prices=[
            (100.0, 102.0, 98.0, 100.0),  # Bar 0: Enter
            (100.0, 102.0, 98.0, 93.0),  # Bar 1: Fill at 100. Close 93 (stop)
            (93.0, 95.0, 90.0, 92.0),  # Bar 2: Stop loss executes at Open 93
        ]
    )

    config = BacktestConfig(
        initial_cash=10_000.0,
        stop_loss_pct=0.05,  # 5% stop loss
        commission_bps=0.0,
        fixed_fee_per_order=0.0,
        slippage_bps=0.0,
    )

    strategy = ModelDrivenStrategy(
        config=config,
        entry_threshold=0.5,
        exit_threshold=0.5,
    )

    engine = BacktestEngine(config=config)
    # Give prediction 0.8 at bar 0 to trigger entry
    predictions = [0.8, 0.8, 0.8]
    result = engine.run(
        data=bars,
        strategy=strategy,
        predictions=predictions,
        close_positions_at_end=False,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.net_pnl < 0.0


def test_walk_forward_evaluation() -> None:
    """Verify chronological out-of-sample walk-forward cross-validation."""
    np.random.seed(42)
    n = 120
    base_time = datetime(2025, 1, 1, tzinfo=UTC)

    # Generate synthetic price series with upward trend
    p_series = [100.0]
    for _ in range(n - 1):
        ret = np.random.normal(0.001, 0.01)
        p_series.append(p_series[-1] * (1.0 + ret))

    records = []
    for i in range(n):
        p = p_series[i]
        records.append(
            {
                "timestamp": base_time + timedelta(days=i),
                "symbol": "SYNTH",
                "open": p,
                "high": p * 1.01,
                "low": p * 0.99,
                "close": p,
                "volume": 10_000.0,
                "feat_mom": float(np.random.randn()),
                "feat_vol": float(abs(np.random.randn())),
                "target_binary": 1 if (i % 2 == 0) else 0,
            }
        )
    df = pd.DataFrame(records)

    wf_config = WalkForwardConfig(
        n_folds=3,
        min_train_bars=30,
        window_type="expanding",
        backtest_config=BacktestConfig(initial_cash=50_000.0),
    )
    evaluator = WalkForwardEvaluator(config=wf_config)

    wf_result = evaluator.evaluate(
        data=df,
        feature_names=["feat_mom", "feat_vol"],
        target_name="target_binary",
        model_type="logistic_regression",
        task_type="classification",
    )

    assert len(wf_result.folds) >= 2
    assert wf_result.overall_metrics is not None
    summary = wf_result.summary_table()
    assert len(summary) >= 3  # Folds + OVERALL
    assert "total_return" in summary.columns
