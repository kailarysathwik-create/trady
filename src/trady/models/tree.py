"""Random Forest models for TRADY quantitative research.

Non-linear tree ensemble supporting classification and regression with
deterministic seeds.
"""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from trady.models.base import TradyModel


class TradyRandomForestClassifier(TradyModel):
    """Random Forest Classifier with deterministic seed and feature importances."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float | None = "sqrt",
        random_seed: int = 42,
    ) -> None:
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        super().__init__(
            model_type="random_forest",
            task_type="classification",
            parameters=parameters,
            random_seed=random_seed,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.estimator_: RandomForestClassifier | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TradyRandomForestClassifier":
        """Fit Random Forest on in-sample training data."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)
        X_clean = self.validate_features(X_arr, self.feature_names)

        self.estimator_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self.estimator_.fit(X_clean, y_clean)
        return self

    def predict(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict binary class labels."""
        X_clean = self.validate_features(X, feature_names)
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.predict(X_clean).astype(np.int64)

    def predict_proba(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict class probabilities of shape (N, 2)."""
        X_clean = self.validate_features(X, feature_names)
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.predict_proba(X_clean).astype(np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get Gini feature importances."""
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.feature_importances_


class TradyRandomForestRegressor(TradyModel):
    """Random Forest Regressor with deterministic seed."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float | None = "sqrt",
        random_seed: int = 42,
    ) -> None:
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        super().__init__(
            model_type="random_forest",
            task_type="regression",
            parameters=parameters,
            random_seed=random_seed,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.estimator_: RandomForestRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TradyRandomForestRegressor":
        """Fit Random Forest on in-sample training data."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)
        X_clean = self.validate_features(X_arr, self.feature_names)

        self.estimator_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self.estimator_.fit(X_clean, y_clean)
        return self

    def predict(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict continuous target values."""
        X_clean = self.validate_features(X, feature_names)
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.predict(X_clean).astype(np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get impurity-based feature importances."""
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.feature_importances_
