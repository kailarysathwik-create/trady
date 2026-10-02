"""TRADY Risk Engine for simulated portfolio constraints and position sizing.

SAFETY INVARIANT:
    All risk evaluation is strictly virtual and educational. No real-money execution
    is implemented or permitted.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from trady.risk.config import RiskConfig
from trady.risk.decision import RiskDecision
from trady.risk.engine import RiskEngine
from trady.risk.sizing import PositionSizer


@dataclass(frozen=True)
class RiskCheckResult:
    """Result of a simulated risk validation (legacy contract)."""

    allowed: bool
    reason: str
    metrics: dict[str, float]


@runtime_checkable
class RiskController(Protocol):
    """Protocol for enforcing risk controls on simulated positions."""

    def evaluate_order(self, order: Any, current_state: Any) -> RiskCheckResult:
        """Evaluate whether a proposed simulated order conforms to risk rules."""
        ...

    def evaluate_portfolio(self, current_state: Any) -> RiskCheckResult:
        """Evaluate overall portfolio health against drawdown and leverage limits."""
        ...


__all__ = [
    "PositionSizer",
    "RiskCheckResult",
    "RiskConfig",
    "RiskController",
    "RiskDecision",
    "RiskEngine",
]
