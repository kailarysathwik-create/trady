"""Historical backtesting and quantitative simulation engine for TRADY.

SAFETY INVARIANT:
    All simulations are strictly virtual and educational. No real-money execution
    is implemented or permitted.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from trady.backtesting.config import BacktestConfig
from trady.backtesting.engine import BacktestEngine, BacktestResult
from trady.backtesting.execution import ExecutionResult, ExecutionSimulator
from trady.backtesting.metrics import BacktestMetrics, calculate_metrics
from trady.backtesting.portfolio import (
    EquityPoint,
    PortfolioTracker,
    Position,
    TradeRecord,
)
from trady.backtesting.reporting import generate_html_report, save_backtest_artifacts
from trady.backtesting.strategy import BaseStrategy, ModelDrivenStrategy, Order
from trady.backtesting.walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardFoldResult,
    WalkForwardResult,
)


@dataclass(frozen=True)
class SimulatedOrder:
    """Immutable virtual order for simulated paper trading."""

    order_id: str
    symbol: str
    action: Literal["BUY", "SELL"]
    quantity: float
    timestamp: datetime
    simulated: bool = True

    def __post_init__(self) -> None:
        if not self.simulated:
            raise ValueError(
                "SAFETY VIOLATION: Non-simulated orders are strictly forbidden in "
                "TRADY."
            )
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive.")


@dataclass(frozen=True)
class BacktestSummary:
    """Aggregated outcome metrics of a historical simulation."""

    run_id: str
    start_date: datetime
    end_date: datetime
    initial_cash: float
    ending_cash: float
    total_trades: int
    metadata: dict[str, Any]


__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestResult",
    "BacktestSummary",
    "BaseStrategy",
    "EquityPoint",
    "ExecutionResult",
    "ExecutionSimulator",
    "ModelDrivenStrategy",
    "Order",
    "PortfolioTracker",
    "Position",
    "SimulatedOrder",
    "TradeRecord",
    "WalkForwardConfig",
    "WalkForwardEvaluator",
    "WalkForwardFoldResult",
    "WalkForwardResult",
    "calculate_metrics",
    "generate_html_report",
    "save_backtest_artifacts",
]
