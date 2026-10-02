"""Experiment runner for dataset loading, chronological splitting, and training."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trady.models.base import TradyModel
from trady.models.metadata import ModelMetadata
from trady.models.metrics import evaluate_model_predictions
from trady.models.registry import create_model
from trady.models.splitting import ChronologicalSplit, ChronologicalSplitter
from trady.targets.dataset import ModelingDataset


@dataclass(frozen=True)
class ExperimentResult:
    """Structured result of a training and evaluation experiment."""

    experiment_id: str
    model_id: str
    model: TradyModel
    split: ChronologicalSplit
    metrics: dict[str, dict[str, float]]
    saved_path: Path | None = None

    def print_summary(self) -> None:
        """Print a formatted experiment evaluation summary."""
        print("=" * 70)
        print(f" EXPERIMENT REPORT: {self.experiment_id}")
        print(f" Model ID:          {self.model_id} ({self.model.model_type})")
        print(f" Target:            {self.model.target_name} ({self.model.task_type})")
        print(f" Features:          {len(self.model.feature_names)} features")
        print("-" * 70)
        print(" CHRONOLOGICAL PARTITIONS:")
        print(
            f"  - Train:      {self.split.train_count} samples "
            f"[{self.split.train_start} -> {self.split.train_end}]"
        )
        if self.split.val_count > 0:
            print(
                f"  - Validation: {self.split.val_count} samples "
                f"[{self.split.val_start} -> {self.split.val_end}]"
            )
        print(
            f"  - Test:       {self.split.test_count} samples "
            f"[{self.split.test_start} -> {self.split.test_end}]"
        )
        print("-" * 70)
        print(" EVALUATION METRICS:")
        for fold, fold_metrics in self.metrics.items():
            print(f"  [{fold.upper()}]:")
            for k, v in fold_metrics.items():
                print(f"    - {k:22s}: {v:.5f}")
        if self.saved_path:
            print(f" Model saved to:    {self.saved_path}")
        print("=" * 70)


def run_model_experiment(
    dataset: ModelingDataset,
    target_name: str,
    model_type: str,
    parameters: dict[str, Any] | None = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    purge_gap: int | None = None,
    random_seed: int = 42,
    output_dir: str | Path | None = "models",
    experiment_id: str | None = None,
) -> ExperimentResult:
    """Run an end-to-end model training and evaluation experiment.

    Args:
        dataset: ModelingDataset containing paired features and targets.
        target_name: Name of target column to predict.
        model_type: Architecture ('baseline', 'logistic_regression', 'random_forest',
            'lightgbm', 'xgboost').
        parameters: Optional hyperparameters for model.
        train_ratio: Fraction of records for chronological training.
        val_ratio: Fraction of records for validation.
        test_ratio: Fraction of records for test.
        purge_gap: Optional purge gap (if None, auto-inferred from target horizon).
        random_seed: Random seed for reproducibility.
        output_dir: Optional directory to persist trained model artifact.
        experiment_id: Optional custom experiment ID.

    Returns:
        ExperimentResult containing model, split, metrics, and persistence info.
    """
    if target_name not in dataset.target_names:
        raise KeyError(
            f"Target '{target_name}' not found in dataset. "
            f"Available targets: {dataset.target_names}"
        )

    # 1. Extract clean aligned modeling arrays
    X, y, timestamps, _ = dataset.get_modeling_arrays(target_name, drop_na=True)
    if len(X) < 10:
        raise ValueError(
            f"Insufficient valid rows ({len(X)}) for modeling after dropping NaNs."
        )

    # 2. Determine task type and target definition
    target_meta = dataset.target_metadatas.get(target_name)
    if target_meta is not None:
        target_def = {
            "name": target_meta.name,
            "horizon": target_meta.horizon,
            "target_type": target_meta.target_type,
            "threshold": target_meta.threshold,
            "calculation_method": target_meta.calculation_method,
        }
        task_type = (
            "classification" if target_meta.target_type == "binary" else "regression"
        )
        auto_purge_gap = target_meta.horizon
    else:
        # Fallback inspection
        unique_vals = set(np.unique(y))
        if unique_vals.issubset({0, 1}):
            task_type = "classification"
        else:
            task_type = "regression"
        target_def = {"name": target_name, "inferred_task": task_type}
        auto_purge_gap = 1

    effective_purge = purge_gap if purge_gap is not None else auto_purge_gap

    # 3. Chronological time-series split
    split = ChronologicalSplitter.split_by_ratio(
        timestamps=timestamps,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        purge_gap=effective_purge,
    )

    X_train = X[split.train_indices]
    y_train = y[split.train_indices]

    has_val = split.val_count > 0
    X_val = X[split.val_indices] if has_val else None
    y_val = y[split.val_indices] if has_val else None

    X_test = X[split.test_indices]
    y_test = y[split.test_indices]

    # 4. Instantiate and fit model
    model = create_model(
        model_type=model_type,
        task_type=task_type,
        parameters=parameters,
        random_seed=random_seed,
    )

    model.fit(
        X=X_train,
        y=y_train,
        feature_names=dataset.feature_names,
        target_name=target_name,
        target_definition=target_def,
        X_val=X_val,
        y_val=y_val,
    )

    # 5. Compute metrics across folds
    def _eval_fold(X_fold: np.ndarray, y_fold: np.ndarray) -> dict[str, float]:
        preds = model.predict(X_fold, dataset.feature_names)
        probs = None
        if task_type == "classification":
            prob_mat = model.predict_proba(X_fold, dataset.feature_names)
            probs = prob_mat[:, 1]
        return evaluate_model_predictions(task_type, y_fold, preds, probs)

    train_metrics = _eval_fold(X_train, y_train)
    val_metrics = _eval_fold(X_val, y_val) if has_val else {}
    test_metrics = _eval_fold(X_test, y_test)

    all_metrics = {"train": train_metrics}
    if has_val:
        all_metrics["validation"] = val_metrics
    all_metrics["test"] = test_metrics

    # 6. Assemble provenance metadata
    exp_id = experiment_id or f"exp_{uuid.uuid4().hex[:8]}"
    model_id = f"{model_type}_{target_name}_{uuid.uuid4().hex[:6]}"

    meta = ModelMetadata(
        experiment_id=exp_id,
        model_id=model_id,
        model_type=model_type,
        task_type=task_type,
        feature_names=dataset.feature_names,
        feature_count=len(dataset.feature_names),
        target_name=target_name,
        target_definition=target_def,
        train_period={
            "start": split.train_start,
            "end": split.train_end,
            "num_samples": split.train_count,
        },
        val_period=(
            {
                "start": split.val_start,
                "end": split.val_end,
                "num_samples": split.val_count,
            }
            if has_val
            else None
        ),
        test_period={
            "start": split.test_start,
            "end": split.test_end,
            "num_samples": split.test_count,
        },
        model_parameters=dict(model.parameters),
        random_seed=random_seed,
        metrics=all_metrics,
    )
    model.metadata = meta

    # 7. Persistence
    saved_path = None
    if output_dir is not None:
        out_path = Path(output_dir) / f"{model_id}.joblib"
        saved_path = model.save(out_path)

    return ExperimentResult(
        experiment_id=exp_id,
        model_id=model_id,
        model=model,
        split=split,
        metrics=all_metrics,
        saved_path=saved_path,
    )
