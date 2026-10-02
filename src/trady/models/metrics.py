"""Evaluation metrics for TRADY quantitative models.

Covers:
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC, Brier score, Log Loss.
- Regression: MAE, RMSE, R², Directional Accuracy (sign hit rate), Pearson correlation.
"""

from typing import Any

import numpy as np
from sklearn import metrics


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute comprehensive classification performance and calibration metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted discrete labels (0 or 1).
        y_prob: Optional predicted probabilities of positive class (1.0).

    Returns:
        Dictionary of computed metric names and float values.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    if len(y_true_arr) == 0:
        return {}

    acc = float(metrics.accuracy_score(y_true_arr, y_pred_arr))
    prec = float(metrics.precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(metrics.recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(metrics.f1_score(y_true_arr, y_pred_arr, zero_division=0))

    result: dict[str, float] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)

        # ROC-AUC requires at least 2 distinct classes in y_true
        if len(np.unique(y_true_arr)) > 1:
            try:
                result["roc_auc"] = float(metrics.roc_auc_score(y_true_arr, y_prob_arr))
            except Exception:
                result["roc_auc"] = float("nan")
        else:
            result["roc_auc"] = float("nan")

        # Brier score (mean squared error of probabilistic predictions)
        result["brier_score"] = float(metrics.brier_score_loss(y_true_arr, y_prob_arr))

        # Log loss
        try:
            # Clip probabilities to avoid inf
            eps = 1e-15
            clipped_prob = np.clip(y_prob_arr, eps, 1.0 - eps)
            result["log_loss"] = float(
                metrics.log_loss(y_true_arr, clipped_prob, labels=[0, 1])
            )
        except Exception:
            result["log_loss"] = float("nan")

    return result


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute comprehensive regression metrics and directional statistics.

    Args:
        y_true: True continuous targets.
        y_pred: Predicted continuous targets.

    Returns:
        Dictionary of computed metric names and float values.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if len(y_true_arr) == 0:
        return {}

    mae = float(metrics.mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(metrics.mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))

    # R-squared can be arbitrarily negative if model is worse than horizontal mean
    try:
        r2 = float(metrics.r2_score(y_true_arr, y_pred_arr))
    except Exception:
        r2 = float("nan")

    # Directional accuracy: sign(y_pred) == sign(y_true)
    sign_true = np.sign(y_true_arr)
    sign_pred = np.sign(y_pred_arr)
    directional_accuracy = float(np.mean(sign_true == sign_pred))

    # Pearson correlation
    if len(y_true_arr) > 1 and np.std(y_true_arr) > 0 and np.std(y_pred_arr) > 0:
        corr_matrix = np.corrcoef(y_true_arr, y_pred_arr)
        pearson_corr = float(corr_matrix[0, 1])
    else:
        pearson_corr = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "pearson_corr": pearson_corr,
    }


def evaluate_model_predictions(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Dispatcher function to evaluate predictions based on task type."""
    if task_type == "classification":
        return evaluate_classification(y_true, y_pred, y_prob)
    elif task_type == "regression":
        return evaluate_regression(y_true, y_pred)
    else:
        msg = (
            f"Unsupported task_type '{task_type}'. "
            "Use 'classification' or 'regression'."
        )
        raise ValueError(msg)
