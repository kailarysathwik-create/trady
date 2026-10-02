"""Performance metrics calculation for TRADY backtesting simulations."""

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from trady.backtesting.portfolio import EquityPoint, TradeRecord


@dataclass(frozen=True)
class BacktestMetrics:
    """Standardized performance and risk metrics of a simulation."""

    total_return: float
    annualized_return: float
    volatility: float
    maximum_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    turnover: float
    trade_count: int
    # Additional diagnostic metrics
    win_count: int
    loss_count: int
    total_fees: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    exposure_time_pct: float
    initial_cash: float
    final_equity: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to JSON-serializable dictionary."""
        d = asdict(self)
        # Handle inf / nan for clean JSON serialization
        for k, v in d.items():
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                d[k] = None
        return d


def calculate_metrics(
    equity_curve: list[EquityPoint],
    trades: list[TradeRecord],
    initial_cash: float,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.02,
) -> BacktestMetrics:
    """Calculate all standard portfolio performance and risk metrics.

    Args:
        equity_curve: Sequential equity points.
        trades: Roundtrip completed trades.
        initial_cash: Starting simulation cash balance.
        periods_per_year: Trading frequency scaling factor (default 252).
        risk_free_rate: Annualized benchmark rate.

    Returns:
        BacktestMetrics object containing all 12 core required metrics.
    """
    if not equity_curve:
        return BacktestMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            maximum_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            turnover=0.0,
            trade_count=0,
            win_count=0,
            loss_count=0,
            total_fees=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
            exposure_time_pct=0.0,
            initial_cash=initial_cash,
            final_equity=initial_cash,
        )

    final_equity = equity_curve[-1].portfolio_equity
    n_bars = len(equity_curve)

    # 1. Total Return
    total_return = (
        (final_equity - initial_cash) / initial_cash if initial_cash > 0 else 0.0
    )

    # 2. Annualized Return (CAGR)
    if n_bars > 0 and final_equity > 0 and initial_cash > 0:
        years = n_bars / periods_per_year
        if years > 0:
            annualized_return = (final_equity / initial_cash) ** (1.0 / years) - 1.0
        else:
            annualized_return = 0.0
    else:
        annualized_return = -1.0 if final_equity <= 0 else 0.0

    # 3. Volatility & Daily Returns
    daily_returns = np.array([pt.daily_return for pt in equity_curve], dtype=float)
    if len(daily_returns) > 1:
        daily_std = float(np.std(daily_returns, ddof=1))
        volatility = daily_std * math.sqrt(periods_per_year)
    else:
        daily_std = 0.0
        volatility = 0.0

    # 4. Maximum Drawdown (positive magnitude: 0.15 = 15% drop)
    drawdowns = [pt.drawdown for pt in equity_curve]
    min_dd = min(drawdowns) if drawdowns else 0.0
    maximum_drawdown = abs(float(min_dd))

    # 5. Sharpe Ratio
    if volatility > 1e-8:
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    else:
        sharpe_ratio = 0.0

    # 6. Sortino Ratio
    negative_returns = daily_returns[daily_returns < 0.0]
    if len(negative_returns) > 1:
        downside_std = float(np.std(negative_returns, ddof=1)) * math.sqrt(
            periods_per_year
        )
    elif len(negative_returns) == 1:
        downside_std = abs(float(negative_returns[0])) * math.sqrt(periods_per_year)
    else:
        downside_std = 0.0

    if downside_std > 1e-8:
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std
    else:
        sortino_ratio = 0.0

    # Trade statistics
    trade_count = len(trades)
    winning_trades = [t for t in trades if t.net_pnl > 0]
    losing_trades = [t for t in trades if t.net_pnl < 0]

    win_count = len(winning_trades)
    loss_count = len(losing_trades)

    # 7. Win Rate
    win_rate = win_count / trade_count if trade_count > 0 else 0.0

    # 8. Average Win
    gross_profit = sum(t.net_pnl for t in winning_trades)
    average_win = gross_profit / win_count if win_count > 0 else 0.0

    # 9. Average Loss
    gross_loss = abs(sum(t.net_pnl for t in losing_trades))
    average_loss = (
        sum(t.net_pnl for t in losing_trades) / loss_count if loss_count > 0 else 0.0
    )

    # 10. Profit Factor (gross profit / gross loss)
    if gross_loss > 1e-8:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 1e-8:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # 11. Turnover: Total Traded Value / (2 * Mean Equity)
    total_traded_value = sum(
        (t.entry_price * t.quantity) + (t.exit_price * t.quantity) for t in trades
    )
    mean_equity = (
        float(np.mean([pt.portfolio_equity for pt in equity_curve]))
        if equity_curve
        else initial_cash
    )
    turnover = total_traded_value / (2.0 * mean_equity) if mean_equity > 1e-8 else 0.0

    # Diagnostic sums
    total_fees = sum(t.total_fees for t in trades)
    net_profit = sum(t.net_pnl for t in trades)

    # Exposure percentage
    exposed_bars = sum(1 for pt in equity_curve if abs(pt.gross_exposure) > 1e-5)
    exposure_time_pct = exposed_bars / n_bars if n_bars > 0 else 0.0

    return BacktestMetrics(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        volatility=float(volatility),
        maximum_drawdown=float(maximum_drawdown),
        sharpe_ratio=float(sharpe_ratio),
        sortino_ratio=float(sortino_ratio),
        win_rate=float(win_rate),
        average_win=float(average_win),
        average_loss=float(average_loss),
        profit_factor=float(profit_factor),
        turnover=float(turnover),
        trade_count=trade_count,
        win_count=win_count,
        loss_count=loss_count,
        total_fees=float(total_fees),
        gross_profit=float(gross_profit),
        gross_loss=float(gross_loss),
        net_profit=float(net_profit),
        exposure_time_pct=float(exposure_time_pct),
        initial_cash=float(initial_cash),
        final_equity=float(final_equity),
    )
