"""CLI commands for TRADY End-to-End Quantitative Research Pipeline."""

import argparse
import sys

from trady.pipeline.config import PipelineConfig
from trady.pipeline.runner import PipelineRunner
from trady.utils.logging import get_logger

logger = get_logger("cli.pipeline")


def add_pipeline_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register 'pipeline' subcommands with main TRADY CLI."""
    pipe_parser = subparsers.add_parser(
        "pipeline",
        help="End-to-end quantitative research workflow orchestration",
    )
    pipe_subs = pipe_parser.add_subparsers(dest="pipeline_subcommand", required=True)

    # 1. trade9 pipeline run
    run_parser = pipe_subs.add_parser(
        "run",
        help="Execute full quantitative research workflow from data to backtest",
    )
    run_parser.add_argument(
        "--symbol",
        "-s",
        type=str,
        default="^NSEI",
        help="Market symbol to download and research (default: ^NSEI)",
    )
    run_parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format",
    )
    run_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format",
    )
    run_parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="Bar timeframe interval (default: 1d)",
    )
    run_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Optional path to existing processed market Parquet file to skip download",
    )
    run_parser.add_argument(
        "--model",
        "-m",
        type=str,
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
    run_parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="target_binary_up_5d",
        help="Target column to predict (default: target_binary_up_5d)",
    )
    run_parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Initial capital in base currency (default: 100,000.0)",
    )
    run_parser.add_argument(
        "--commission-bps",
        type=float,
        default=5.0,
        help="Commission in basis points (default: 5.0 bps)",
    )
    run_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5.0 bps)",
    )
    run_parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop loss decimal (e.g. 0.05 for 5%%)",
    )
    run_parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit decimal (e.g. 0.10 for 10%%)",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing market datasets if re-downloading",
    )
    run_parser.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Auto-adjust OHLC prices for corporate actions",
    )


def handle_pipeline_command(args: argparse.Namespace) -> int:
    """Dispatch pipeline commands."""
    subcmd = getattr(args, "pipeline_subcommand", None)
    if subcmd == "run":
        cfg = PipelineConfig(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            timeframe=args.timeframe,
            dataset_path=args.dataset,
            model_name=args.model,
            target_name=args.target,
            initial_cash=args.initial_cash,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            overwrite=args.overwrite,
            auto_adjust=args.auto_adjust,
        )
        runner = PipelineRunner(config=cfg)
        try:
            res = runner.run()
            return 0 if res.status == "SUCCESS" else 1
        except Exception as e:
            print(f"\n[FATAL] Pipeline Execution Failed: {e}", file=sys.stderr)
            return 1
    else:
        print(f"Unknown pipeline subcommand: {subcmd}", file=sys.stderr)
        return 1
