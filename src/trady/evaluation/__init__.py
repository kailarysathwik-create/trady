"""Evaluation and statistical validation contracts for TRADY research.

Measures statistical significance and out-of-sample repeatability.
TRADY models do NOT claim or guarantee profitability.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MetricResult:
    """Individual statistical validation metric."""

    name: str
    value: float
    description: str


@runtime_checkable
class ModelEvaluator(Protocol):
    """Protocol for calculating statistical evaluation metrics on model predictions."""

    def evaluate(self, predictions: Any, targets: Any) -> list[MetricResult]:
        """Compute evaluation metrics comparing predictions against targets."""
        ...


__all__ = ["MetricResult", "ModelEvaluator"]
