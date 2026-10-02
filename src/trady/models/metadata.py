"""Model metadata and experiment provenance tracking.

Stores full research provenance for each trained model:
- Experiment ID & Model ID
- Feature-set version & feature names
- Target definition (horizon, threshold, method)
- Training & validation temporal boundaries
- Model parameters & random seed
- Software environment versions
- Performance metrics
"""

import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import joblib
import lightgbm
import numpy
import pyarrow
import sklearn
import xgboost

import trady


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable audit metadata for trained research models."""

    experiment_id: str
    model_id: str
    model_type: str
    task_type: str
    feature_names: tuple[str, ...]
    feature_count: int
    target_name: str
    target_definition: dict[str, Any]
    train_period: dict[str, Any]
    val_period: dict[str, Any] | None
    test_period: dict[str, Any] | None
    model_parameters: dict[str, Any]
    random_seed: int
    feature_set_version: str = "v1"
    software_version: dict[str, str] | None = None
    metrics: dict[str, dict[str, float]] | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(UTC).isoformat())
        if self.software_version is None:
            object.__setattr__(
                self, "software_version", get_current_software_versions()
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for JSON persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Deserialize from dictionary."""
        data_copy = dict(data)
        if "feature_names" in data_copy and isinstance(
            data_copy["feature_names"], list
        ):
            data_copy["feature_names"] = tuple(data_copy["feature_names"])
        return cls(**data_copy)


def get_current_software_versions() -> dict[str, str]:
    """Capture runtime software and dependency versions."""
    return {
        "trady": getattr(trady, "__version__", "0.1.0"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }
