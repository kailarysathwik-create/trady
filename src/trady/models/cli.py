"""CLI command tree and handlers for TRADY machine-learning models."""

import argparse
import json
import sys
from pathlib import Path

from trady.models.base import TradyModel
from trady.models.experiment import run_model_experiment
from trady.models.metrics import evaluate_model_predictions
from trady.models.registry import list_available_models
from trady.targets.dataset import ModelingDataset


def add_model_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register 'models' command and its subcommands."""
    models_parser = subparsers.add_parser(
        "models",
        help="Machine learning model training, evaluation, and inspection",
    )
    model_subs = models_parser.add_subparsers(dest="model_subcommand", required=True)

    # 1. trady models list-types
    model_subs.add_parser(
        "list-types",
        help="List supported model architectures and descriptions",
    )

    # 2. trady models train
    train_parser = model_subs.add_parser(
        "train",
        help="Train a quantitative research model on a labeled dataset",
    )
    train_parser.add_argument(
        "dataset",
        help="Path to labeled modeling Parquet file (from Target Engine)",
    )
    train_parser.add_argument(
        "--model",
        "-m",
        default="logistic_regression",
        choices=[
            "baseline",
            "logistic_regression",
            "random_forest",
            "lightgbm",
            "xgboost",
        ],
        help="Model architecture (default: logistic_regression)",
    )
    train_parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Target column name to predict (e.g. target_binary_up_5d)",
    )
    train_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Training partition ratio (default: 0.6)",
    )
    train_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation partition ratio (default: 0.2)",
    )
    train_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test partition ratio (default: 0.2)",
    )
    train_parser.add_argument(
        "--purge-gap",
        type=int,
        default=None,
        help="Purge gap in bars before split boundaries (default: auto)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    train_parser.add_argument(
        "--output-dir",
        "-o",
        default="models",
        help="Directory to save trained model (default: models)",
    )

    # 3. trady models inspect
    inspect_parser = model_subs.add_parser(
        "inspect",
        help="Inspect trained model parameters and provenance metadata",
    )
    inspect_parser.add_argument(
        "model_file",
        help="Path to saved model file (.joblib)",
    )

    # 4. trady models evaluate
    eval_parser = model_subs.add_parser(
        "evaluate",
        help="Evaluate an existing trained model on a labeled dataset",
    )
    eval_parser.add_argument(
        "model_file",
        help="Path to saved model file (.joblib)",
    )
    eval_parser.add_argument(
        "dataset",
        help="Path to labeled modeling dataset Parquet file",
    )


def handle_model_command(args: argparse.Namespace) -> int:
    """Dispatch execution of models subcommands."""
    subcmd = args.model_subcommand

    if subcmd == "list-types":
        return _handle_list_types()
    elif subcmd == "train":
        return _handle_train(args)
    elif subcmd == "inspect":
        return _handle_inspect(args)
    elif subcmd == "evaluate":
        return _handle_evaluate(args)
    else:
        print(f"Unknown models subcommand: {subcmd}", file=sys.stderr)
        return 1


def _handle_list_types() -> int:
    models = list_available_models()
    print("=" * 70)
    print(f" SUPPORTED TRADY MODEL ARCHITECTURES ({len(models)} total)")
    print("=" * 70)
    for name, desc in models.items():
        print(f"  * {name:<22s} : {desc}")
    print("=" * 70)
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset file not found: {dataset_path}", file=sys.stderr)
        return 1

    dataset = ModelingDataset.load(dataset_path)
    result = run_model_experiment(
        dataset=dataset,
        target_name=args.target,
        model_type=args.model,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        purge_gap=args.purge_gap,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )
    result.print_summary()
    return 0


def _handle_inspect(args: argparse.Namespace) -> int:
    model_path = Path(args.model_file)
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    meta_file = model_path.with_suffix(model_path.suffix + ".meta.json")
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = None

    print("=" * 70)
    print(f" TRADY MODEL INSPECTION: {model_path.name}")
    print("=" * 70)
    if meta:
        print(f"  - Model ID:       {meta.get('model_id')}")
        print(f"  - Experiment ID:  {meta.get('experiment_id')}")
        print(f"  - Architecture:   {meta.get('model_type')}")
        print(f"  - Task Type:      {meta.get('task_type')}")
        print(f"  - Target Name:    {meta.get('target_name')}")
        print(f"  - Random Seed:    {meta.get('random_seed')}")
        print(f"  - Feature Count:  {meta.get('feature_count')}")
        feats = meta.get("feature_names", [])
        sample_feats = ", ".join(feats[:6])
        ellipsis_str = "..." if len(feats) > 6 else ""
        print(f"  - Features:       {sample_feats}{ellipsis_str}")
        print(f"  - Train Period:   {meta.get('train_period')}")
        print(f"  - Val Period:     {meta.get('val_period')}")
        print(f"  - Test Period:    {meta.get('test_period')}")
        print("  - Metrics:")
        metrics_dict = meta.get("metrics", {})
        for fold, f_metrics in metrics_dict.items():
            print(f"    [{fold.upper()}]:")
            for k, v in f_metrics.items():
                print(f"      - {k:20s}: {v:.5f}")
    else:
        print("  [Warning] Companion metadata file not found.")

    print("=" * 70)
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    model_path = Path(args.model_file)
    dataset_path = Path(args.dataset)

    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1
    if not dataset_path.exists():
        print(f"Dataset file not found: {dataset_path}", file=sys.stderr)
        return 1

    model = TradyModel.load(model_path)
    dataset = ModelingDataset.load(dataset_path)

    target_name = model.target_name
    X, y, _, _ = dataset.get_modeling_arrays(target_name, drop_na=True)

    preds = model.predict(X, dataset.feature_names)
    probs = None
    if model.task_type == "classification":
        prob_mat = model.predict_proba(X, dataset.feature_names)
        probs = prob_mat[:, 1]

    metrics = evaluate_model_predictions(model.task_type, y, preds, probs)

    print("=" * 70)
    print(f" EVALUATION ON DATASET: {dataset_path.name}")
    print(f" Model:  {model.model_type} ({model.target_name})")
    print(f" Samples: {len(X)}")
    print("-" * 70)
    for k, v in metrics.items():
        print(f"  - {k:22s}: {v:.5f}")
    print("=" * 70)
    return 0
