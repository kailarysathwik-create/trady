"""Model factory and registry for TRADY quantitative models.

Instantiates and inspects model architectures across classification and regression.
"""

from typing import Any

from trady.models.base import TradyModel
from trady.models.baseline import NaiveBaselineClassifier, NaiveBaselineRegressor
from trady.models.boosting import (
    TradyLightGBMClassifier,
    TradyLightGBMRegressor,
    TradyXGBoostClassifier,
    TradyXGBoostRegressor,
)
from trady.models.linear import TradyLogisticRegression, TradyRidgeRegression
from trady.models.tree import TradyRandomForestClassifier, TradyRandomForestRegressor

AVAILABLE_MODELS: dict[str, dict[str, Any]] = {
    "baseline": {
        "description": (
            "Naive benchmark (majority class for classification, zero/mean for "
            "regression)"
        ),
        "classification_cls": NaiveBaselineClassifier,
        "regression_cls": NaiveBaselineRegressor,
    },
    "logistic_regression": {
        "description": "L2-regularized linear model with in-sample standardization",
        "classification_cls": TradyLogisticRegression,
        "regression_cls": TradyRidgeRegression,
    },
    "random_forest": {
        "description": "Bagged ensemble of decision trees with deterministic seed",
        "classification_cls": TradyRandomForestClassifier,
        "regression_cls": TradyRandomForestRegressor,
    },
    "lightgbm": {
        "description": "Fast gradient boosted decision trees with early stopping",
        "classification_cls": TradyLightGBMClassifier,
        "regression_cls": TradyLightGBMRegressor,
    },
    "xgboost": {
        "description": "Extreme gradient boosted decision trees",
        "classification_cls": TradyXGBoostClassifier,
        "regression_cls": TradyXGBoostRegressor,
    },
}


def create_model(
    model_type: str,
    task_type: str,
    parameters: dict[str, Any] | None = None,
    random_seed: int = 42,
) -> TradyModel:
    """Instantiate a model by type name and task type.

    Args:
        model_type: Architecture name ('baseline', 'logistic_regression',
            'random_forest', 'lightgbm', 'xgboost').
        task_type: Target task ('classification' or 'regression').
        parameters: Optional dictionary of hyperparameters.
        random_seed: Random seed for deterministic reproducibility.

    Returns:
        Instantiated TradyModel instance.
    """
    m_type = model_type.lower()
    if m_type not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available architectures: {list(AVAILABLE_MODELS.keys())}"
        )

    info = AVAILABLE_MODELS[m_type]
    params = dict(parameters or {})

    if task_type == "classification":
        model_cls = info["classification_cls"]
    elif task_type == "regression":
        model_cls = info["regression_cls"]
    else:
        msg = (
            f"Unsupported task_type '{task_type}'. "
            "Use 'classification' or 'regression'."
        )
        raise ValueError(msg)

    # Pass random_seed if accepted by model
    params["random_seed"] = random_seed
    return model_cls(**params)


def list_available_models() -> dict[str, str]:
    """Return dictionary of model type names and their descriptions."""
    return {name: info["description"] for name, info in AVAILABLE_MODELS.items()}
