"""Abstract base class and contracts for TRADY quantitative models.

Enforces:
1. Feature schema validation: exact feature count, ordering, and names.
2. Target schema validation: target name and task type alignment.
3. Deterministic seeds and parameters.
4. Model serialization and deserialization with complete audit metadata.
5. Safety: Outputs statistical predictions only (no trading orders or execution).
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from trady.models.metadata import ModelMetadata


class TradyModel(ABC):
    """Base class for all TRADY quantitative research models."""

    def __init__(
        self,
        model_type: str,
        task_type: str,
        parameters: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.model_type = model_type
        self.task_type = task_type  # "classification" or "regression"
        self.parameters = parameters or {}
        self.random_seed = random_seed

        self.feature_names: tuple[str, ...] = ()
        self.target_name: str = ""
        self.target_definition: dict[str, Any] = {}
        self.metadata: ModelMetadata | None = None
        self.is_fitted: bool = False

    def validate_features(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Validate input feature matrix against trained schema.

        Args:
            X: 2D feature matrix.
            feature_names: Optional names of features in X.

        Returns:
            Validated 2D float64 numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before inference.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Feature matrix must be 2D, got shape {X_arr.shape}")

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature dimension mismatch: model expected {len(self.feature_names)} "
                f"features, received {X_arr.shape[1]}"
            )

        if feature_names is not None:
            names_tuple = tuple(feature_names)
            if names_tuple != self.feature_names:
                missing = set(self.feature_names) - set(names_tuple)
                extra = set(names_tuple) - set(self.feature_names)
                raise ValueError(
                    f"Feature schema mismatch! Missing: {missing}, Extra: {extra}. "
                    "Feature names and order must match the training schema exactly."
                )

        if np.isnan(X_arr).any():
            nan_count = int(np.isnan(X_arr).sum())
            raise ValueError(
                f"Feature matrix contains {nan_count} NaN values. "
                "Inputs must be clean before passing to model."
            )

        return X_arr

    def validate_targets(
        self,
        y: np.ndarray,
        target_name: str,
    ) -> np.ndarray:
        """Validate target vector against task type.

        Args:
            y: 1D target array.
            target_name: Name of target variable.

        Returns:
            Validated 1D numpy array.
        """
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError(f"Target array must be 1D, got shape {y_arr.shape}")

        if np.isnan(y_arr).any():
            nan_count = int(np.isnan(y_arr).sum())
            raise ValueError(
                f"Target vector '{target_name}' contains {nan_count} NaN values. "
                "Trailing or warmup rows must be dropped before training."
            )

        if self.task_type == "classification":
            unique_vals = set(np.unique(y_arr))
            if not unique_vals.issubset({0, 1}):
                raise ValueError(
                    f"Binary classification target '{target_name}' contains values "
                    f"{unique_vals}, expected only {{0, 1}}."
                )
            return y_arr.astype(np.int64)

        return y_arr.astype(np.float64)

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TradyModel":
        """Fit model strictly on training data."""
        pass

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Generate point predictions."""
        pass

    def predict_proba(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Generate class probability estimates for classification.

        Default fallback returns one-hot probabilities based on point predictions.
        """
        if self.task_type != "classification":
            raise NotImplementedError(
                "predict_proba() is only applicable to classification models."
            )
        preds = self.predict(X, feature_names=feature_names)
        return np.column_stack([1.0 - preds, preds])

    def save(self, file_path: str | Path) -> Path:
        """Serialize model weights and companion audit metadata."""
        out_p = Path(file_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, out_p)

        if self.metadata:
            meta_file = out_p.with_suffix(out_p.suffix + ".meta.json")
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata.to_dict(), f, indent=4)

        return out_p

    @classmethod
    def load(cls, file_path: str | Path) -> "TradyModel":
        """Load serialized model and companion metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model: TradyModel = joblib.load(path)

        meta_file = path.with_suffix(path.suffix + ".meta.json")
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta_dict = json.load(f)
            model.metadata = ModelMetadata.from_dict(meta_dict)

        return model
