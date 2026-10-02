"""Tests for TRADY machine learning model layer.

Covers:
1. Chronological splitting (strictly ordered, no random shuffling, purge gap).
2. Metrics computation for classification and regression.
3. Feature schema validation and target schema validation.
4. Model architectures (Baseline, Logistic, Random Forest, LightGBM, XGBoost).
5. Model serialization and metadata roundtrip.
6. End-to-end experiment runner.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from trady.data.provider import SyntheticDataProvider
from trady.features.pipeline import FeaturePipeline
from trady.models.base import TradyModel
from trady.models.experiment import run_model_experiment
from trady.models.linear import TradyLogisticRegression
from trady.models.metrics import evaluate_classification, evaluate_regression
from trady.models.registry import create_model, list_available_models
from trady.models.splitting import ChronologicalSplitter
from trady.models.tree import TradyRandomForestClassifier
from trady.targets.config import TargetConfig
from trady.targets.pipeline import TargetPipeline

# -----------------------------------------------------------------------------
# 1. Chronological Splitting Tests
# -----------------------------------------------------------------------------


def test_chronological_splitting_strictly_ordered() -> None:
    """Chronological split must maintain train < val < test without shuffling."""
    timestamps = [f"2024-01-{i:03d}" for i in range(1, 101)]

    split = ChronologicalSplitter.split_by_ratio(
        timestamps=timestamps,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        purge_gap=0,
    )

    assert split.train_count == 60
    assert split.val_count == 20
    assert split.test_count == 20

    # Strict ordering: indices are contiguous and ascending
    np.testing.assert_array_equal(split.train_indices, np.arange(0, 60))
    np.testing.assert_array_equal(split.val_indices, np.arange(60, 80))
    np.testing.assert_array_equal(split.test_indices, np.arange(80, 100))

    assert split.train_end < split.val_start
    assert split.val_end < split.test_start


def test_chronological_splitting_purge_gap() -> None:
    """Purge gap must discard trailing bars before fold transitions."""
    timestamps = [f"2024-01-{i:03d}" for i in range(1, 101)]
    purge_gap = 5

    split = ChronologicalSplitter.split_by_ratio(
        timestamps=timestamps,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        purge_gap=purge_gap,
    )

    # Train had 60 bars, purged last 5 -> 55 bars (0 to 54)
    assert split.train_count == 55
    assert split.train_indices[-1] == 54

    # Validation had 20 bars (60..79), purged last 5 -> 15 bars (60..74)
    assert split.val_count == 15
    assert split.val_indices[0] == 60
    assert split.val_indices[-1] == 74

    # Test retains all 20 bars (80..99)
    assert split.test_count == 20
    assert split.test_indices[0] == 80


def test_chronological_splitting_rejects_unsorted_timestamps() -> None:
    """Splitter must raise ValueError if input timestamps are not sorted."""
    unsorted_ts = ["2024-01-05", "2024-01-02", "2024-01-03"]
    with pytest.raises(ValueError, match="chronological order"):
        ChronologicalSplitter.split_by_ratio(unsorted_ts, 0.5, 0.25, 0.25)


# -----------------------------------------------------------------------------
# 2. Evaluation Metrics Tests
# -----------------------------------------------------------------------------


def test_classification_metrics_computation() -> None:
    """Verify precision, recall, accuracy, roc_auc, and brier score."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.6, 0.85, 0.15])

    res = evaluate_classification(y_true, y_pred, y_prob)

    assert "accuracy" in res
    assert "precision" in res
    assert "recall" in res
    assert "f1" in res
    assert "roc_auc" in res
    assert "brier_score" in res
    assert "log_loss" in res

    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert res["brier_score"] >= 0.0


def test_regression_metrics_computation() -> None:
    """Verify MAE, RMSE, R², and directional accuracy."""
    y_true = np.array([0.02, -0.01, 0.05, -0.03])
    y_pred = np.array([0.018, -0.009, 0.045, -0.025])

    res = evaluate_regression(y_true, y_pred)

    assert "mae" in res
    assert "rmse" in res
    assert "r2" in res
    assert "directional_accuracy" in res

    assert math.isclose(res["directional_accuracy"], 1.0)
    assert res["mae"] < 0.01
    assert res["r2"] > 0.95


# -----------------------------------------------------------------------------
# 3. Schema Validation & Guardrails
# -----------------------------------------------------------------------------


def test_model_feature_schema_validation() -> None:
    """Model must strictly enforce feature dimensions, names, and NaNs."""
    model = TradyLogisticRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([1, 0, 1, 0])
    feat_names = ("feat_a", "feat_b")

    # Unfitted predict raises RuntimeError
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)

    model.fit(X, y, feature_names=feat_names, target_name="target_binary_up_1d")

    # Valid inference
    preds = model.predict(X, feature_names=feat_names)
    assert len(preds) == 4

    # Dimension mismatch
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        model.predict(np.array([[1.0, 2.0, 3.0]]))

    # Schema name mismatch
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        model.predict(X, feature_names=("wrong_col", "feat_b"))

    # Input contains NaN
    X_nan = np.array([[1.0, np.nan], [3.0, 4.0]])
    with pytest.raises(ValueError, match="contains 1 NaN values"):
        model.predict(X_nan)


def test_classification_model_rejects_continuous_target() -> None:
    """Classification model must reject non-binary target labels."""
    model = TradyLogisticRegression()
    X = np.ones((5, 2))
    y_cont = np.array([0.01, -0.02, 0.03, 0.04, -0.01])

    with pytest.raises(ValueError, match="Binary classification target"):
        model.fit(X, y_cont, ("f1", "f2"), target_name="target_fwd_ret_5d")


# -----------------------------------------------------------------------------
# 4. Model Architectures & Registry
# -----------------------------------------------------------------------------


def test_model_registry_contains_required_models() -> None:
    """Registry must support baseline, logreg, random forest, and boosting."""
    models = list_available_models()
    assert "baseline" in models
    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "lightgbm" in models
    assert "xgboost" in models


def test_all_models_fit_and_predict() -> None:
    """Verify all 5 supported architectures fit and predict on synthetic data."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(50, 4))
    y_clf = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    y_reg = (X[:, 0] * 0.5 + rng.normal(scale=0.1, size=50)).astype(np.float64)
    feats = ("f1", "f2", "f3", "f4")
    model_types = [
        "baseline",
        "logistic_regression",
        "random_forest",
        "lightgbm",
        "xgboost",
    ]

    # Classification
    for m_type in model_types:
        clf = create_model(m_type, "classification", random_seed=42)
        clf.fit(X, y_clf, feats, "target_clf")
        preds = clf.predict(X, feats)
        probs = clf.predict_proba(X, feats)
        assert preds.shape == (50,)
        assert probs.shape == (50, 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Regression
    for m_type in model_types:
        reg = create_model(m_type, "regression", random_seed=42)
        reg.fit(X, y_reg, feats, "target_reg")
        preds = reg.predict(X, feats)
        assert preds.shape == (50,)


# -----------------------------------------------------------------------------
# 5. Serialization and Roundtrip
# -----------------------------------------------------------------------------


def test_model_serialization_and_metadata_roundtrip(tmp_path: Path) -> None:
    """Trained model and metadata must serialize and deserialize identically."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([1, 0, 1, 0])
    feats = ("f1", "f2")

    model = TradyRandomForestClassifier(n_estimators=10, random_seed=99)
    model.fit(X, y, feats, "target_binary")

    out_file = tmp_path / "test_rf.joblib"
    saved = model.save(out_file)
    assert saved.exists()

    loaded = TradyModel.load(saved)
    assert loaded.model_type == "random_forest"
    assert loaded.feature_names == feats

    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


# -----------------------------------------------------------------------------
# 6. End-to-End Experiment Runner
# -----------------------------------------------------------------------------


def test_end_to_end_model_experiment(tmp_path: Path) -> None:
    """Run full pipeline: synthetic data -> features -> targets -> ML experiment."""
    provider = SyntheticDataProvider(seed=42)
    records, _ = provider.generate_valid_dataset(symbol="SPY", num_bars=100)
    from trady.data.normalizer import records_to_arrow_table

    market_table = records_to_arrow_table(records)

    # 1. Feature engine
    feat_pipe = FeaturePipeline()
    feat_ds, _ = feat_pipe.compute(market_table)

    # 2. Target engine
    target_pipe = TargetPipeline(
        config=TargetConfig(
            selected_targets=("target_binary_up_5d",),
            drop_unavailable_rows=False,
        )
    )
    modeling_ds, _ = target_pipe.compute(market_table, features_data=feat_ds)

    # 3. Model experiment runner
    result = run_model_experiment(
        dataset=modeling_ds,
        target_name="target_binary_up_5d",
        model_type="logistic_regression",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        output_dir=tmp_path,
        random_seed=42,
    )

    assert result.experiment_id.startswith("exp_")
    assert result.model.is_fitted is True
    assert result.saved_path is not None and result.saved_path.exists()

    # Metrics present for all 3 folds
    assert "train" in result.metrics
    assert "validation" in result.metrics
    assert "test" in result.metrics
    assert "accuracy" in result.metrics["test"]

    # Verify provenance metadata
    meta = result.model.metadata
    assert meta is not None
    assert meta.target_name == "target_binary_up_5d"
    assert meta.train_period["num_samples"] > 0
    assert meta.test_period["num_samples"] > 0
    assert "scikit_learn" in meta.software_version
