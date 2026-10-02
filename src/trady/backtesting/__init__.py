"""Backtesting and paper-trading simulation interface contracts for TRADY.

SAFETY INVARIANT:
    All simulations are strictly virtual and educational. No real-money execution
    is implemented or permitted.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable


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


@runtime_checkable
class BacktestEngine(Protocol):
    """Protocol defining the execution contract for simulation runners."""

    def run_simulation(self, dataset: Any, signal_generator: Any) -> BacktestSummary:
        """Execute a causal historical simulation."""
        ...


__all__ = ["BacktestEngine", "BacktestSummary", "SimulatedOrder"]
