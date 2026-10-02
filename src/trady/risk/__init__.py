"""Risk management and portfolio constraints for simulated trading in TRADY."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RiskCheckResult:
    """Result of a simulated risk validation."""

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


__all__ = ["RiskCheckResult", "RiskController"]
