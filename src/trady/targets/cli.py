"""CLI commands for TRADY Target Engine."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from trady.targets.config import TargetConfig
from trady.targets.pipeline import TargetPipeline
from trady.targets.registry import target_registry
from trady.utils.logging import get_logger

logger = get_logger("cli.targets")


def add_target_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register target subcommands with the main TRADY CLI parser."""
    tgt_parser = subparsers.add_parser(
        "targets",
        help="Target and label engineering subsystem for supervised learning",
    )
    tgt_subparsers = tgt_parser.add_subparsers(
        dest="target_command",
        required=True,
        help="Target calculation and inspection commands",
    )

    # 1. trady targets list
    tgt_subparsers.add_parser(
        "list",
        help="List all registered forward-looking targets and their horizons",
    )

    # 2. trady targets compute <dataset>
    comp_p = tgt_subparsers.add_parser(
        "compute",
        help="Compute forward targets on processed market data or feature datasets",
    )
    comp_p.add_argument(
        "dataset",
        type=str,
        help="Path to input market data Parquet file",
    )
    comp_p.add_argument(
        "--features",
        type=str,
        default=None,
        help="Optional path to computed feature Parquet file to pair with targets",
    )
    comp_p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Destination Parquet path (default: data/targets/<stem>_labeled.parquet)",
    )
    comp_p.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Explicit list of target names to compute",
    )
    comp_p.add_argument(
        "--drop-unavailable",
        action="store_true",
        help="Drop final trailing rows where future targets are NaN",
    )

    # 3. trady targets inspect <dataset>
    insp_p = tgt_subparsers.add_parser(
        "inspect",
        help="Inspect computed target/modeling dataset and report class balance",
    )
    insp_p.add_argument(
        "dataset",
        type=str,
        help="Path to labeled modeling Parquet dataset",
    )


def handle_target_command(args: argparse.Namespace) -> int:
    """Execute target CLI commands.

    Returns:
        0 on success, non-zero on failure.
    """
    cmd = getattr(args, "target_command", None)

    if cmd == "list":
        return _cmd_list()
    elif cmd == "compute":
        return _cmd_compute(args)
    elif cmd == "inspect":
        return _cmd_inspect(args.dataset)
    else:
        print(f"Unknown target command: {cmd}")
        return 1


def _cmd_list() -> int:
    """List registered targets in tabular format."""
    targets = target_registry.list_targets()

    print("=" * 80)
    print(f" REGISTERED TRADY TARGETS ({len(targets)} total)")
    print("=" * 80)
    fmt = "{:<24} | {:<12} | {:<8} | {:<10} | {:<12}"
    print(fmt.format("Target Name", "Type", "Horizon", "Threshold", "Method"))
    print("-" * 80)

    for m in targets:
        thresh_str = str(m.threshold) if m.threshold is not None else "None"
        method_short = m.calculation_method.replace("forward_", "")
        print(
            fmt.format(
                m.name, m.target_type, f"{m.horizon} bars", thresh_str, method_short
            )
        )

    print("=" * 80)
    return 0


def _cmd_compute(args: argparse.Namespace) -> int:
    """Compute configured targets on dataset."""
    input_path = Path(args.dataset)
    if not input_path.exists():
        print(f"Error: Dataset not found: {input_path}")
        return 1

    if args.output:
        out_path = Path(args.output)
    else:
        stem = input_path.stem
        out_path = Path("data/targets") / f"{stem}_labeled.parquet"

    selected = (
        tuple(args.targets)
        if args.targets
        else (
            "target_fwd_ret_1d",
            "target_fwd_ret_5d",
            "target_binary_up_5d",
            "target_fwd_vol_5d",
        )
    )

    config = TargetConfig(
        selected_targets=selected,
        drop_unavailable_rows=args.drop_unavailable,
        output_dir=str(out_path.parent),
    )

    print(f"Computing targets for '{input_path}'...")
    print(f"  - Targets:          {', '.join(selected)}")
    print(f"  - Drop Unavailable: {config.drop_unavailable_rows}")
    print(f"  - Output Target:    {out_path}")

    pipeline = TargetPipeline(config=config)
    try:
        modeling_ds, report = pipeline.compute(
            market_data=input_path,
            features_data=args.features,
        )
    except Exception as e:
        print(f"\n[FAIL] Target calculation error: {e}")
        return 1

    saved = modeling_ds.save(out_path, source_path=input_path)

    print("\nTarget Computation Report:")
    print(f"  - Valid:         {'YES' if report.is_valid else 'NO'}")
    print(f"  - Total Records: {report.total_records}")
    print(f"  - Features:      {len(modeling_ds.feature_names)}")
    print(f"  - Targets:       {len(modeling_ds.target_names)}")
    print(f"  - Symbols:       {', '.join(report.symbols)}")

    print("\nUnavailable (Trailing) Row Counts:")
    for t_name, count in report.unavailable_counts.items():
        print(f"    * {t_name:<22}: {count} trailing rows")

    print(f"\n[OK] Labeled modeling dataset saved to {saved}")
    return 0


def _cmd_inspect(dataset_path: str) -> int:
    """Inspect modeling dataset and display summary statistics."""
    path = Path(dataset_path)
    if not path.exists():
        print(f"Error: File not found: {path}")
        return 1

    table = pq.read_table(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    print("=" * 70)
    print(f" MODELING DATASET INSPECTION: {path.name}")
    print("=" * 70)
    print(f"  - Record Count:  {len(table)}")
    print(f"  - Total Columns: {len(table.column_names)}")

    meta_dict = None
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta_dict = json.load(f)
        except Exception:
            meta_dict = None

    if meta_dict:
        feats = meta_dict.get("feature_names", [])
        tgts = meta_dict.get("target_names", [])
        sample_feats = ", ".join(feats[:6])
        ellipsis_str = "..." if len(feats) > 6 else ""
        print(f"  - Features ({len(feats)}): {sample_feats}{ellipsis_str}")
        print(f"  - Targets ({len(tgts)}):  {', '.join(tgts)}")

    print("=" * 70)
    return 0
