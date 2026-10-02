"""Gradient boosted tree models (LightGBM and XGBoost) for TRADY research.

Provides deterministic seeds, validation-set evaluation, and early stopping.
"""

import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from trady.models.base import TradyModel


class TradyLightGBMClassifier(TradyModel):
    """LightGBM Classifier with deterministic seed and early stopping support."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 10,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_seed: int = 42,
    ) -> None:
        parameters = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        super().__init__(
            model_type="lightgbm",
            task_type="classification",
            parameters=parameters,
            random_seed=random_seed,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.estimator_: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TradyLightGBMClassifier":
        """Fit LightGBM on in-sample training data with optional validation set."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)
        X_clean = self.validate_features(X_arr, self.feature_names)

        self.estimator_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_seed,
            verbosity=-1,
            n_jobs=-1,
        )

        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_clean = self.validate_features(X_val, self.feature_names)
            y_val_clean = self.validate_targets(y_val, target_name)
            eval_set = [(X_val_clean, y_val_clean)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.estimator_.fit(
                X_clean,
                y_clean,
                eval_set=eval_set,
            )
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
        """Get split-based feature importances."""
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.feature_importances_


class TradyLightGBMRegressor(TradyModel):
    """LightGBM Regressor with deterministic seed."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 10,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_seed: int = 42,
    ) -> None:
        parameters = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        super().__init__(
            model_type="lightgbm",
            task_type="regression",
            parameters=parameters,
            random_seed=random_seed,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.estimator_: lgb.LGBMRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TradyLightGBMRegressor":
        """Fit LightGBM Regressor."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)
        X_clean = self.validate_features(X_arr, self.feature_names)

        self.estimator_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_seed,
            verbosity=-1,
            n_jobs=-1,
        )

        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_clean = self.validate_features(X_val, self.feature_names)
            y_val_clean = self.validate_targets(y_val, target_name)
            eval_set = [(X_val_clean, y_val_clean)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.estimator_.fit(
                X_clean,
                y_clean,
                eval_set=eval_set,
            )
        return self

    def predict(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict continuous target."""
        X_clean = self.validate_features(X, feature_names)
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.predict(X_clean).astype(np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get split-based feature importances."""
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.feature_importances_


class TradyXGBoostClassifier(TradyModel):
    """XGBoost Classifier with deterministic seed."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_seed: int = 42,
    ) -> None:
        parameters = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        super().__init__(
            model_type="xgboost",
            task_type="classification",
            parameters=parameters,
            random_seed=random_seed,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.estimator_: xgb.XGBClassifier | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TradyXGBoostClassifier":
        """Fit XGBoost Classifier."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)
        X_clean = self.validate_features(X_arr, self.feature_names)

        self.estimator_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_seed,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )

        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_clean = self.validate_features(X_val, self.feature_names)
            y_val_clean = self.validate_targets(y_val, target_name)
            eval_set = [(X_val_clean, y_val_clean)]

        self.estimator_.fit(
            X_clean,
            y_clean,
            eval_set=eval_set,
            verbose=False,
        )
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


class TradyXGBoostRegressor(TradyModel):
    """XGBoost Regressor with deterministic seed."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_seed: int = 42,
    ) -> None:
        parameters = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        super().__init__(
            model_type="xgboost",
            task_type="regression",
            parameters=parameters,
            random_seed=random_seed,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.estimator_: xgb.XGBRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...] | list[str],
        target_name: str,
        target_definition: dict[str, Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TradyXGBoostRegressor":
        """Fit XGBoost Regressor."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.feature_names = tuple(feature_names)
        self.target_name = target_name
        self.target_definition = target_definition or {}

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Feature count mismatch between X and feature_names")

        self.is_fitted = True
        y_clean = self.validate_targets(y, target_name)
        X_clean = self.validate_features(X_arr, self.feature_names)

        self.estimator_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_seed,
            verbosity=0,
            n_jobs=-1,
        )

        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val_clean = self.validate_features(X_val, self.feature_names)
            y_val_clean = self.validate_targets(y_val, target_name)
            eval_set = [(X_val_clean, y_val_clean)]

        self.estimator_.fit(
            X_clean,
            y_clean,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        """Predict continuous target."""
        X_clean = self.validate_features(X, feature_names)
        if self.estimator_ is None:
            raise RuntimeError("Estimator is not fitted.")
        return self.estimator_.predict(X_clean).astype(np.float64)
