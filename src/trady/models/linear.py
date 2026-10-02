"""Linear and logistic regression models for TRADY research.

Includes:
- TradyLogisticRegression: L2/L1 regularized classification with in-sample scaling.
- TradyRidgeRegression: L2 regularized linear regression with in-sample scaling.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from trady.models.base import TradyModel


class TradyLogisticRegression(TradyModel):
    """Logistic regression classifier with strictly in-sample feature scaling."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        scale_features: bool = True,
        random_seed: int = 42,
    ) -> None:
        parameters = {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": max_iter,
            "scale_features": scale_features,
        }
        super().__init__(
            model_type="logistic_regression",
            task_type="classification",
            parameters=parameters,
            random_seed=random_seed,
        )
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.scale_features = scale_features

        self.scaler_: StandardScaler | None = None
        self.estimator_: LogisticRegression | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TradyLogisticRegression":
        """Fit scaler and logistic regression on training data."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)
        X_clean = self.validate_features(X_arr, self.feature_names)

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_train_scaled = self.scaler_.fit_transform(X_clean)
        else:
            self.scaler_ = None
            X_train_scaled = X_clean

        kwargs: dict[str, Any] = {
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_seed,
        }
        if self.penalty != "l2":
            kwargs["penalty"] = self.penalty

        self.estimator_ = LogisticRegression(**kwargs)
        self.estimator_.fit(X_train_scaled, y_clean)
        return self

    def predict(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        X_clean = self.validate_features(X, feature_names)
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_clean)
        else:
            X_scaled = X_clean

        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.predict(X_scaled).astype(np.int64)

    def predict_proba(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict class probabilities of shape (N, 2)."""
        X_clean = self.validate_features(X, feature_names)
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_clean)
        else:
            X_scaled = X_clean

        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.predict_proba(X_scaled).astype(np.float64)


class TradyRidgeRegression(TradyModel):
    """Ridge regression for continuous target prediction with in-sample scaling."""

    def __init__(
        self,
        alpha: float = 1.0,
        scale_features: bool = True,
        random_seed: int = 42,
    ) -> None:
        parameters = {"alpha": alpha, "scale_features": scale_features}
        super().__init__(
            model_type="ridge_regression",
            task_type="regression",
            parameters=parameters,
            random_seed=random_seed,
        )
        self.alpha = alpha
        self.scale_features = scale_features

        self.scaler_: StandardScaler | None = None
        self.estimator_: Ridge | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TradyRidgeRegression":
        """Fit scaler and Ridge model on training data."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)
        X_clean = self.validate_features(X_arr, self.feature_names)

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_train_scaled = self.scaler_.fit_transform(X_clean)
        else:
            self.scaler_ = None
            X_train_scaled = X_clean

        self.estimator_ = Ridge(
            alpha=self.alpha,
            random_state=self.random_seed,
        )
        self.estimator_.fit(X_train_scaled, y_clean)
        return self

    def predict(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict continuous targets."""
        X_clean = self.validate_features(X, feature_names)
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_clean)
        else:
            X_scaled = X_clean

        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.predict(X_scaled).astype(np.float64)
