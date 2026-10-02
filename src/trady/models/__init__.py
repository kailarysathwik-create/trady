"""TRADY machine-learning model layer for quantitative research.

Provides:
- Model abstraction (TradyModel) with schema validation and persistence
- Naive baselines, Logistic Regression, Ridge, Random Forest, LightGBM, XGBoost
- Strictly chronological splitting with purge/embargo gaps
- Complete provenance metadata and evaluation metrics
"""

from trady.models.base import TradyModel
from trady.models.baseline import NaiveBaselineClassifier, NaiveBaselineRegressor
from trady.models.boosting import (
    TradyLightGBMClassifier,
    TradyLightGBMRegressor,
    TradyXGBoostClassifier,
    TradyXGBoostRegressor,
)
from trady.models.experiment import ExperimentResult, run_model_experiment
from trady.models.linear import TradyLogisticRegression, TradyRidgeRegression
from trady.models.metadata import ModelMetadata
from trady.models.metrics import (
    evaluate_classification,
    evaluate_model_predictions,
    evaluate_regression,
)
from trady.models.registry import AVAILABLE_MODELS, create_model, list_available_models
from trady.models.splitting import ChronologicalSplit, ChronologicalSplitter
from trady.models.tree import TradyRandomForestClassifier, TradyRandomForestRegressor

__all__ = [
    "AVAILABLE_MODELS",
    "ChronologicalSplit",
    "ChronologicalSplitter",
    "ExperimentResult",
    "ModelMetadata",
    "NaiveBaselineClassifier",
    "NaiveBaselineRegressor",
    "TradyLightGBMClassifier",
    "TradyLightGBMRegressor",
    "TradyLogisticRegression",
    "TradyModel",
    "TradyRandomForestClassifier",
    "TradyRandomForestRegressor",
    "TradyRidgeRegression",
    "TradyXGBoostClassifier",
    "TradyXGBoostRegressor",
    "create_model",
    "evaluate_classification",
    "evaluate_model_predictions",
    "evaluate_regression",
    "list_available_models",
    "run_model_experiment",
]
