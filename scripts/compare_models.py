"""Deterministic model comparison script evaluating all TRADY architectures.

Compares:
1. Naive Baseline
2. Logistic Regression
3. Random Forest
4. LightGBM
5. XGBoost
"""

import sys
from pathlib import Path

from trady.models.experiment import run_model_experiment
from trady.targets.dataset import ModelingDataset


def main() -> int:
    dataset_path = Path("data/targets/SPY_1d_labeled.parquet")
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}", file=sys.stderr)
        return 1

    ds = ModelingDataset.load(dataset_path)
    models = ["baseline", "logistic_regression", "random_forest", "lightgbm", "xgboost"]
    target = "target_binary_up_5d"

    print("=" * 88)
    print(" TRADY MULTI-MODEL RESEARCH BENCHMARK COMPARISON")
    print(
        f" Target: {target} | Features: {len(ds.feature_names)} | "
        "Split: 60% Train / 20% Val / 20% Test"
    )
    print("=" * 88)
    header = (
        f"{'Model Architecture':<22} | {'Train Acc':<10} | {'Val Acc':<10} | "
        f"{'Test Acc':<10} | {'Test ROC-AUC':<12} | {'Test Brier':<10}"
    )
    print(header)
    print("-" * 88)

    for m in models:
        res = run_model_experiment(
            dataset=ds,
            target_name=target,
            model_type=m,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
            output_dir="models",
        )
        tr_acc = res.metrics["train"].get("accuracy", 0.0)
        va_acc = res.metrics["validation"].get("accuracy", 0.0)
        te_acc = res.metrics["test"].get("accuracy", 0.0)
        te_auc = res.metrics["test"].get("roc_auc", float("nan"))
        te_brier = res.metrics["test"].get("brier_score", 0.0)
        row = (
            f"{m:<22} | {tr_acc:<10.3f} | {va_acc:<10.3f} | "
            f"{te_acc:<10.3f} | {te_auc:<12.3f} | {te_brier:<10.4f}"
        )
        print(row)

    print("=" * 88)
    print(
        "NOTICE: Results are for quantitative research comparison only. "
        "No guaranteed profits or trading signals."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
