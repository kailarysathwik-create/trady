"""CLI commands for TRADY Feature Engine."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from trady.features.config import FeatureConfig
from trady.features.pipeline import FeaturePipeline
from trady.features.registry import registry
from trady.utils.logging import get_logger

logger = get_logger("cli.features")


def add_feature_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register feature subcommands with the main TRADY CLI parser."""
    feat_parser = subparsers.add_parser(
        "features",
        help="Feature engineering and transformation subsystem",
    )
    feat_subparsers = feat_parser.add_subparsers(
        dest="feature_command",
        required=True,
        help="Feature management and calculation commands",
    )

    # 1. trady features list
    list_p = feat_subparsers.add_parser(
        "list",
        help="List all registered features and their causal metadata",
    )
    list_p.add_argument(
        "--group",
        type=str,
        default=None,
        help="Filter features by group (e.g. PRICE, MOMENTUM, VOLATILITY)",
    )

    # 2. trady features compute <dataset>
    comp_p = feat_subparsers.add_parser(
        "compute",
        help="Compute causal features on a processed OHLCV dataset",
    )
    comp_p.add_argument(
        "dataset",
        type=str,
        help="Path to input processed market data (.parquet or .json)",
    )
    comp_p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Target Parquet path (default: data/features/<stem>_features.parquet)",
    )
    comp_p.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Specific feature groups to compute",
    )
    comp_p.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Explicit list of feature names to compute",
    )
    comp_p.add_argument(
        "--nan-policy",
        choices=["keep", "drop", "error"],
        default="keep",
        help="Policy for lookback NaNs (keep=preserve rows, drop=remove warmup)",
    )

    # 3. trady features inspect <dataset>
    insp_p = feat_subparsers.add_parser(
        "inspect",
        help="Inspect computed feature dataset and its audit metadata",
    )
    insp_p.add_argument(
        "dataset",
        type=str,
        help="Path to feature Parquet dataset",
    )


def handle_feature_command(args: argparse.Namespace) -> int:
    """Execute feature CLI commands.

    Returns:
        0 on success, non-zero on failure.
    """
    cmd = getattr(args, "feature_command", None)

    if cmd == "list":
        return _cmd_list(args.group)
    elif cmd == "compute":
        return _cmd_compute(args)
    elif cmd == "inspect":
        return _cmd_inspect(args.dataset)
    else:
        print(f"Unknown feature command: {cmd}")
        return 1


def _cmd_list(group: str | None) -> int:
    """List features formatted in a clean table."""
    features = registry.list_by_group(group) if group else registry.list_features()

    print("=" * 80)
    print(f" REGISTERED TRADY FEATURES ({len(features)} total)")
    print("=" * 80)
    fmt = "{:<20} | {:<14} | {:<8} | {:<14} | {:<8}"
    print(fmt.format("Feature", "Group", "Lookback", "Cols", "Status"))
    print("-" * 80)

    for m in features:
        cols_str = ",".join(m.required_columns)
        pit_str = "SAFE" if m.is_point_in_time else "UNSAFE"
        print(fmt.format(m.name, m.feature_group, m.lookback_period, cols_str, pit_str))

    print("=" * 80)
    return 0


def _cmd_compute(args: argparse.Namespace) -> int:
    """Execute feature computation pipeline."""
    input_path = Path(args.dataset)
    if not input_path.exists():
        print(f"Error: Input dataset not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        stem = input_path.stem
        out_path = Path("data/features") / f"{stem}_features.parquet"

    groups = (
        tuple(args.groups)
        if args.groups
        else ("PRICE", "MOMENTUM", "VOLATILITY", "VOLUME", "PRICE_STRUCTURE")
    )
    features = tuple(args.features) if args.features else None

    config = FeatureConfig(
        enabled_groups=groups,
        selected_features=features,
        nan_policy=args.nan_policy,
        output_dir=str(out_path.parent),
    )

    print(f"Computing features for '{input_path}'...")
    print(f"  - NaN Policy:    {config.nan_policy}")
    print(f"  - Output Target: {out_path}")

    pipeline = FeaturePipeline(config=config)
    try:
        table, report = pipeline.compute(input_path)
    except Exception as e:
        print(f"\n[FAIL] Feature calculation error: {e}")
        return 1

    saved_path = pipeline.save(table, out_path, source_path=input_path)

    print("\nFeature Computation Report:")
    print(f"  - Valid:         {'YES' if report.is_valid else 'NO'}")
    print(f"  - Records:       {report.total_records}")
    print(f"  - Features:      {report.feature_count}")
    print(f"  - Symbols:       {', '.join(report.symbols)}")

    if report.warnings:
        print("  - Warnings:")
        for w in report.warnings:
            print(f"      * {w}")

    if not report.is_valid:
        print("  - Errors:")
        for err in report.errors:
            print(f"      * {err}")
        return 1

    print(f"\n[OK] Feature dataset saved to {saved_path}")
    return 0


def _cmd_inspect(dataset_path: str) -> int:
    """Inspect a computed feature Parquet file and companion metadata."""
    path = Path(dataset_path)
    if not path.exists():
        print(f"Error: File not found: {path}")
        return 1

    table = pq.read_table(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    print("=" * 70)
    print(f" FEATURE DATASET INSPECTION: {path.name}")
    print("=" * 70)
    print(f"  - Record Count:  {len(table)}")
    print(f"  - Total Columns: {len(table.column_names)}")
    feature_cols = [c for c in table.column_names if c not in ("timestamp", "symbol")]
    preview = ", ".join(feature_cols[:8])
    if len(feature_cols) > 8:
        preview += "..."
    print(f"  - Features ({len(feature_cols)}): {preview}")

    if "symbol" in table.column_names:
        symbols = sorted(set(table.column("symbol").to_pylist()))
        print(f"  - Symbols:       {', '.join(symbols)}")

    if "timestamp" in table.column_names and len(table) > 0:
        ts = table.column("timestamp").to_pylist()
        print(f"  - Date Range:    {ts[0]} -> {ts[-1]}")

    if meta_path.exists():
        print("\n  Companion Audit Metadata:")
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta_dict = json.load(f)
            print(json.dumps(meta_dict, indent=4))
        except Exception as e:
            print(f"  (Failed to read metadata: {e})")

    print("=" * 70)
    return 0
