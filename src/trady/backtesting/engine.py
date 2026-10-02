"""Chronological backtesting engine preventing look-ahead bias."""

import uuid
from dataclasses import dataclass

import pandas as pd

from trady.backtesting.config import BacktestConfig
from trady.backtesting.execution import ExecutionResult, ExecutionSimulator
from trady.backtesting.metrics import BacktestMetrics, calculate_metrics
from trady.backtesting.portfolio import EquityPoint, PortfolioTracker, TradeRecord
from trady.backtesting.strategy import BaseStrategy, Order


@dataclass
class BacktestResult:
    """Complete results and artifacts from a historical backtest run."""

    config: BacktestConfig
    metrics: BacktestMetrics
    equity_curve: list[EquityPoint]
    trades: list[TradeRecord]
    executions: list[ExecutionResult]

    def to_equity_dataframe(self) -> pd.DataFrame:
        """Convert equity curve history to pandas DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "cash",
                    "positions_value",
                    "portfolio_equity",
                    "daily_return",
                    "drawdown",
                    "gross_exposure",
                    "net_exposure",
                ]
            )
        return pd.DataFrame([pt.to_dict() for pt in self.equity_curve])

    def to_trades_dataframe(self) -> pd.DataFrame:
        """Convert roundtrip trades history to pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "trade_id",
                    "symbol",
                    "entry_time",
                    "exit_time",
                    "side",
                    "quantity",
                    "entry_price",
                    "exit_price",
                    "gross_pnl",
                    "total_fees",
                    "net_pnl",
                    "pnl_pct",
                    "holding_bars",
                    "exit_reason",
                ]
            )
        return pd.DataFrame([t.to_dict() for t in self.trades])


class BacktestEngine:
    """Orchestrates chronological execution and portfolio simulation."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.simulator = ExecutionSimulator(self.config)

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        predictions: pd.Series | list[float] | None = None,
        close_positions_at_end: bool = True,
    ) -> BacktestResult:
        """Execute a strictly chronological backtest over market data.

        Args:
            data: OHLCV DataFrame sorted chronologically. Required columns:
                timestamp, symbol, open, high, low, close, volume.
            strategy: Strategy generating next-bar orders from bar T information.
            predictions: Optional model predictions aligned row-for-row with data.
            close_positions_at_end: If True, liquidates remaining open positions
                at the final bar close.

        Returns:
            BacktestResult with metrics, equity curve, and trade records.
        """
        required_cols = {
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"Backtest data missing required columns: {missing}")

        n_bars = len(data)
        if n_bars == 0:
            raise ValueError("Cannot run backtest on empty DataFrame.")

        tracker = PortfolioTracker(initial_cash=self.config.initial_cash)
        executions: list[ExecutionResult] = []
        pending_orders: list[Order] = []

        # Convert predictions to sequence for indexing
        pred_list: list[float | None]
        if predictions is not None:
            if len(predictions) != n_bars:
                raise ValueError(
                    f"Predictions length ({len(predictions)}) does not match "
                    f"data length ({n_bars})."
                )
            if isinstance(predictions, pd.Series):
                pred_list = [
                    None if pd.isna(p) else float(p) for p in predictions.values
                ]
            else:
                pred_list = [None if p is None else float(p) for p in predictions]
        else:
            pred_list = [None] * n_bars

        # Chronological bar-by-bar iteration
        for t in range(n_bars):
            row = data.iloc[t]
            symbol = str(row["symbol"])
            b_ts = row["timestamp"]
            b_open = float(row["open"])
            b_high = float(row["high"])
            b_low = float(row["low"])
            b_close = float(row["close"])
            b_vol = float(row["volume"])

            bar_dict = {
                "symbol": symbol,
                "timestamp": b_ts,
                "open": b_open,
                "high": b_high,
                "low": b_low,
                "close": b_close,
                "volume": b_vol,
            }

            # Step 1: Execute pending orders generated from bar T-1 at bar T OPEN
            if pending_orders:
                for ord_req in pending_orders:
                    if ord_req.action == "BUY":
                        # Validate cash availability
                        if tracker.cash <= self.config.fixed_fee_per_order:
                            continue
                        exec_res = self.simulator.execute_order(
                            order_id=f"ord_{uuid.uuid4().hex[:8]}",
                            symbol=ord_req.symbol,
                            action="BUY",
                            quantity=ord_req.quantity,
                            bar_open=b_open,
                            bar_high=b_high,
                            bar_low=b_low,
                            bar_close=b_close,
                            bar_volume=b_vol,
                            timestamp=b_ts,
                        )
                        # Check cash constraint
                        if exec_res.total_cost > tracker.cash:
                            # Prorate quantity to fit cash
                            available_for_trade = (
                                tracker.cash - self.config.fixed_fee_per_order
                            ) * 0.99
                            if available_for_trade > 0:
                                adjusted_qty = available_for_trade / exec_res.fill_price
                                exec_res = self.simulator.execute_order(
                                    order_id=exec_res.order_id,
                                    symbol=ord_req.symbol,
                                    action="BUY",
                                    quantity=adjusted_qty,
                                    bar_open=b_open,
                                    bar_high=b_high,
                                    bar_low=b_low,
                                    bar_close=b_close,
                                    bar_volume=b_vol,
                                    timestamp=b_ts,
                                )
                        tracker.apply_execution(
                            exec_res, bar_idx=t, exit_reason=ord_req.exit_reason
                        )
                        executions.append(exec_res)

                    elif ord_req.action == "SELL":
                        if tracker.has_position(ord_req.symbol):
                            exec_res = self.simulator.execute_order(
                                order_id=f"ord_{uuid.uuid4().hex[:8]}",
                                symbol=ord_req.symbol,
                                action="SELL",
                                quantity=ord_req.quantity,
                                bar_open=b_open,
                                bar_high=b_high,
                                bar_low=b_low,
                                bar_close=b_close,
                                bar_volume=b_vol,
                                timestamp=b_ts,
                            )
                            tracker.apply_execution(
                                exec_res, bar_idx=t, exit_reason=ord_req.exit_reason
                            )
                            executions.append(exec_res)

                pending_orders = []

            # Step 2: Mark-to-market at bar T CLOSE
            tracker.mark_to_market(
                timestamp=b_ts,
                current_prices={symbol: b_close},
            )

            # Step 3: Strategy evaluates bar T state to generate orders for bar T+1
            if t < n_bars - 1:
                pos = tracker.positions.get(symbol)
                order = strategy.evaluate(
                    bar_idx=t,
                    bar_data=bar_dict,
                    prediction=pred_list[t],
                    position=pos,
                    cash=tracker.cash,
                    current_equity=tracker.current_equity,
                )
                if order is not None:
                    pending_orders.append(order)

        # Close out any remaining position on final bar if requested
        if close_positions_at_end:
            last_row = data.iloc[-1]
            last_sym = str(last_row["symbol"])
            if tracker.has_position(last_sym):
                pos = tracker.positions[last_sym]
                last_close = float(last_row["close"])
                last_vol = float(last_row["volume"])
                close_exec = self.simulator.execute_order(
                    order_id=f"ord_close_{uuid.uuid4().hex[:8]}",
                    symbol=last_sym,
                    action="SELL",
                    quantity=pos.quantity,
                    bar_open=last_close,
                    bar_high=float(last_row["high"]),
                    bar_low=float(last_row["low"]),
                    bar_close=last_close,
                    bar_volume=last_vol,
                    timestamp=last_row["timestamp"],
                )
                tracker.apply_execution(
                    close_exec, bar_idx=n_bars - 1, exit_reason="end_of_simulation"
                )
                executions.append(close_exec)
                # Final mark to market
                tracker.mark_to_market(
                    timestamp=last_row["timestamp"],
                    current_prices={last_sym: last_close},
                )

        metrics = calculate_metrics(
            equity_curve=tracker.equity_curve,
            trades=tracker.trades,
            initial_cash=self.config.initial_cash,
            periods_per_year=self.config.periods_per_year,
            risk_free_rate=self.config.risk_free_rate,
        )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            equity_curve=tracker.equity_curve,
            trades=tracker.trades,
            executions=executions,
        )
