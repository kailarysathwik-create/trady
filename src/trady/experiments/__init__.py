"""Experiment tracking and research reproducibility contracts for TRADY."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExperimentRecord:
    """Immutable record tracking research model iterations and parameters."""

    experiment_id: str
    name: str
    created_at: datetime
    config_snapshot: dict[str, Any]
    git_hash: str | None
    metrics: dict[str, float]
    tags: tuple[str, ...] = ()


@runtime_checkable
class ExperimentTracker(Protocol):
    """Protocol for logging reproducible experiment artifacts."""

    def record_experiment(self, record: ExperimentRecord) -> None:
        """Persist an experiment snapshot."""
        ...

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve an experiment record by ID."""
        ...


__all__ = ["ExperimentRecord", "ExperimentTracker"]
