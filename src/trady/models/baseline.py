"""Naive baseline models for quantitative finance research.

Critical for benchmarking: every ML model must beat a naive baseline
(e.g., majority class or zero return) to demonstrate predictive validity.
"""

from typing import Any

import numpy as np

from trady.models.base import TradyModel


class NaiveBaselineClassifier(TradyModel):
    """Naive classification baseline predicting majority class or class prior."""

    def __init__(
        self,
        strategy: str = "majority_class",
        constant_val: int = 1,
        random_seed: int = 42,
    ) -> None:
        super().__init__(
            model_type="naive_baseline",
            task_type="classification",
            parameters={"strategy": strategy, "constant_val": constant_val},
            random_seed=random_seed,
        )
        self.strategy = strategy
        self.constant_val = constant_val
        self.majority_class_: int = 0
        self.prior_p1_: float = 0.5

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "NaiveBaselineClassifier":
        """Compute training class frequencies."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)

        # Compute empirical statistics
        p1 = float(np.mean(y_clean == 1))
        self.prior_p1_ = p1
        self.majority_class_ = 1 if p1 >= 0.5 else 0

        return self

    def predict(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict class labels."""
        X_clean = self.validate_features(X, feature_names)
        n = len(X_clean)

        if self.strategy == "majority_class":
            return np.full(n, self.majority_class_, dtype=np.int64)
        elif self.strategy == "constant":
            return np.full(n, self.constant_val, dtype=np.int64)
        elif self.strategy == "prior":
            rng = np.random.default_rng(self.random_seed)
            return (rng.random(n) < self.prior_p1_).astype(np.int64)
        else:
            raise ValueError(f"Unknown baseline strategy '{self.strategy}'")

    def predict_proba(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict class probabilities."""
        X_clean = self.validate_features(X, feature_names)
        n = len(X_clean)

        if self.strategy in ("majority_class", "prior"):
            p1 = self.prior_p1_
            return np.column_stack([np.full(n, 1.0 - p1), np.full(n, p1)])
        elif self.strategy == "constant":
            p = float(self.constant_val)
            return np.column_stack([np.full(n, 1.0 - p), np.full(n, p)])
        else:
            return super().predict_proba(X, feature_names)


class NaiveBaselineRegressor(TradyModel):
    """Naive continuous baseline predicting mean, median, or zero return."""

    def __init__(
        self,
        strategy: str = "zero",
        constant_val: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        super().__init__(
            model_type="naive_baseline",
            task_type="regression",
            parameters={"strategy": strategy, "constant_val": constant_val},
            random_seed=random_seed,
        )
        self.strategy = strategy
        self.constant_val = constant_val
        self.mean_val_: float = 0.0
        self.median_val_: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "NaiveBaselineRegressor":
        """Compute historical target statistics."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)

        self.mean_val_ = float(np.mean(y_clean))
        self.median_val_ = float(np.median(y_clean))

        return self

    def predict(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict continuous targets."""
        X_clean = self.validate_features(X, feature_names)
        n = len(X_clean)

        if self.strategy == "zero":
            return np.zeros(n, dtype=np.float64)
        elif self.strategy == "mean":
            return np.full(n, self.mean_val_, dtype=np.float64)
        elif self.strategy == "median":
            return np.full(n, self.median_val_, dtype=np.float64)
        elif self.strategy == "constant":
            return np.full(n, self.constant_val, dtype=np.float64)
        else:
            raise ValueError(f"Unknown baseline strategy '{self.strategy}'")
