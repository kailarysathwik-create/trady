"""Model interface contracts for TRADY quantitative research.

NOTE: Concrete machine-learning and statistical models are intentionally not implemented
in this initial foundation. This module establishes interface protocols.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable model metadata for reproducibility tracking."""

    model_id: str
    architecture: str
    hyperparameters: dict[str, Any]
    random_seed: int


@runtime_checkable
class ResearchModel(Protocol):
    """Protocol defining the core interface for research models."""

    @property
    def metadata(self) -> ModelMetadata:
        """Model provenance and metadata."""
        ...

    def fit(self, features: Any, targets: Any) -> None:
        """Fit model strictly on in-sample training data."""
        ...

    def predict(self, features: Any) -> Any:
        """Generate out-of-sample inferences or probabilistic forecasts."""
        ...


__all__ = ["ModelMetadata", "ResearchModel"]
