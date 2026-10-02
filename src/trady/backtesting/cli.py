"""CLI command tree and handlers for TRADY historical backtesting simulations."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq

from trady.backtesting.config import BacktestConfig
from trady.backtesting.engine import BacktestEngine
from trady.backtesting.reporting import save_backtest_artifacts
from trady.backtesting.strategy import ModelDrivenStrategy
from trady.backtesting.walk_forward import WalkForwardConfig, WalkForwardEvaluator
from trady.models.base import TradyModel


def add_backtest_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register 'backtest' command and its subcommands."""
    bt_parser = subparsers.add_parser(
        "backtest",
        help="Historical backtesting simulation and strategy evaluation",
    )
    bt_subs = bt_parser.add_subparsers(dest="backtest_subcommand", required=True)

    # 1. trady backtest run
    run_parser = bt_subs.add_parser(
        "run",
        help="Run a historical backtest on market or modeling data",
    )
    run_parser.add_argument(
        "dataset",
        help="Path to Parquet market data or modeling dataset",
    )
    run_parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Path to saved TradyModel (.joblib) for signal generation",
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
        "--fixed-fee",
        type=float,
        default=1.0,
        help="Fixed fee per trade in base currency (default: $1.00)",
    )
    run_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Bid/Ask slippage in basis points (default: 5.0 bps)",
    )
    run_parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.5,
        help="Model prediction entry threshold (default: 0.5)",
    )
    run_parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.5,
        help="Model prediction exit threshold (default: 0.5)",
    )
    run_parser.add_argument(
        "--holding-bars",
        type=int,
        default=None,
        help="Maximum holding period in bars (default: None)",
    )
    run_parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop loss percentage as decimal (e.g. 0.05 for 5%%)",
    )
    run_parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit percentage as decimal (e.g. 0.10 for 10%%)",
    )
    run_parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to save artifacts (default: reports/backtests/<timestamp>)",
    )

    # 2. trady backtest inspect
    inspect_parser = bt_subs.add_parser(
        "inspect",
        help="Inspect artifacts and metrics from a prior backtest run",
    )
    inspect_parser.add_argument(
        "artifact_dir",
        help="Path to folder containing backtest artifacts",
    )

    # 3. trady backtest walk-forward
    wf_parser = bt_subs.add_parser(
        "walk-forward",
        help="Run chronological out-of-sample walk-forward cross-validation",
    )
    wf_parser.add_argument(
        "dataset",
        help="Path to modeling dataset Parquet file",
    )
    wf_parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Target column name to predict",
    )
    wf_parser.add_argument(
        "--model-type",
        "-m",
        default="logistic_regression",
        choices=[
            "baseline",
            "logistic_regression",
            "random_forest",
            "lightgbm",
            "xgboost",
        ],
        help="Model architecture for walk-forward evaluation",
    )
    wf_parser.add_argument(
        "--folds",
        "-k",
        type=int,
        default=3,
        help="Number of chronological folds (default: 3)",
    )
    wf_parser.add_argument(
        "--window-type",
        default="expanding",
        choices=["expanding", "rolling"],
        help="Walk-forward window type (default: expanding)",
    )
    wf_parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to save walk-forward artifacts",
    )


def handle_backtest_command(args: argparse.Namespace) -> int:
    """Dispatch backtest command to appropriate handler."""
    subcmd = args.backtest_subcommand
    if subcmd == "run":
        return _handle_run(args)
    elif subcmd == "inspect":
        return _handle_inspect(args)
    elif subcmd == "walk-forward":
        return _handle_walk_forward(args)
    else:
        print(f"Unknown backtest subcommand: {subcmd}", file=sys.stderr)
        return 1


def _handle_run(args: argparse.Namespace) -> int:
    data_path = Path(args.dataset)
    if not data_path.is_file():
        print(f"Error: Dataset file not found: {data_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(data_path)
    # Ensure chronological sorting
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    config = BacktestConfig(
        initial_cash=args.initial_cash,
        commission_bps=args.commission_bps,
        fixed_fee_per_order=args.fixed_fee,
        slippage_bps=args.slippage_bps,
        holding_period_bars=args.holding_bars,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )

    predictions = None
    is_classification = True

    if args.model is not None:
        model_path = Path(args.model)
        if not model_path.is_file():
            print(f"Error: Model file not found: {model_path}", file=sys.stderr)
            return 1
        if hasattr(TradyModel, "load"):
            model: TradyModel = TradyModel.load(model_path)
        else:
            model_payload = joblib.load(model_path)
            if isinstance(model_payload, dict):
                model = model_payload.get("model", model_payload)
            else:
                model = model_payload
        is_classification = model.task_type == "classification"

        # Check feature presence
        feature_cols = list(model.feature_names)
        missing_feats = set(feature_cols) - set(df.columns)
        if missing_feats:
            print(
                f"Error: Dataset missing model features: {missing_feats}",
                file=sys.stderr,
            )
            return 1

        X = df[feature_cols].to_numpy()
        # Clean any NaNs with 0.0
        X_clean = pd.DataFrame(X).ffill().bfill().to_numpy()
        if is_classification:
            predictions = model.predict_proba(X_clean)[:, 1]
        else:
            predictions = model.predict(X_clean)

    strategy = ModelDrivenStrategy(
        config=config,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        is_classification=is_classification,
    )

    engine = BacktestEngine(config=config)
    result = engine.run(
        data=df,
        strategy=strategy,
        predictions=predictions,
        close_positions_at_end=True,
    )

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports") / "backtests" / f"run_{timestamp_str}"

    extra_meta = {
        "dataset_file": str(data_path),
        "model_file": str(args.model) if args.model else None,
        "entry_threshold": args.entry_threshold,
        "exit_threshold": args.exit_threshold,
    }
    artifacts = save_backtest_artifacts(
        result=result,
        output_dir=out_dir,
        extra_metadata=extra_meta,
    )

    m = result.metrics
    print("=" * 70)
    print(" TRADY HISTORICAL BACKTEST COMPLETED")
    print("=" * 70)
    print(f" Initial Capital:    ${m.initial_cash:,.2f}")
    print(f" Final Equity:       ${m.final_equity:,.2f}")
    print(f" Total Return:       {m.total_return * 100.0:+.2f}%")
    print(f" Annualized Return:  {m.annualized_return * 100.0:+.2f}%")
    print(f" Annual Volatility:  {m.volatility * 100.0:.2f}%")
    print(f" Sharpe Ratio:       {m.sharpe_ratio:.2f}")
    print(f" Sortino Ratio:      {m.sortino_ratio:.2f}")
    print(f" Max Drawdown:       {m.maximum_drawdown * 100.0:.2f}%")
    print(f" Completed Trades:   {m.trade_count}")
    print(f" Win Rate:           {m.win_rate * 100.0:.1f}%")
    print(f" Profit Factor:      {m.profit_factor:.2f}")
    print(f" Total Fees Paid:    ${m.total_fees:,.2f}")
    print(f" Portfolio Turnover: {m.turnover:.2f}x")
    print("-" * 70)
    print(" SAVED EXPERIMENT ARTIFACTS:")
    for name, path in artifacts.items():
        print(f"  - {name:22s}: {path}")
    print("=" * 70)
    return 0


def _handle_inspect(args: argparse.Namespace) -> int:
    art_dir = Path(args.artifact_dir)
    if not art_dir.is_dir():
        print(f"Error: Artifact directory not found: {art_dir}", file=sys.stderr)
        return 1

    metrics_file = art_dir / "metrics.json"
    if not metrics_file.is_file():
        print(f"Error: metrics.json not found in {art_dir}", file=sys.stderr)
        return 1

    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)

    cfg_file = art_dir / "configuration.json"
    cfg = {}
    if cfg_file.is_file():
        with open(cfg_file, encoding="utf-8") as f:
            cfg = json.load(f)

    print("=" * 70)
    print(f" TRADY BACKTEST ARTIFACT INSPECTION: {art_dir.name}")
    print("=" * 70)
    if cfg and "backtest_config" in cfg:
        init_c = cfg["backtest_config"].get("initial_cash", 0.0)
        print(f" Initial Capital:    ${init_c:,.2f}")
    print(f" Total Return:       {metrics.get('total_return', 0.0) * 100.0:+.2f}%")
    print(f" Annualized Return:  {metrics.get('annualized_return', 0.0) * 100.0:+.2f}%")
    print(f" Sharpe Ratio:       {metrics.get('sharpe_ratio', 0.0):.2f}")
    print(f" Sortino Ratio:      {metrics.get('sortino_ratio', 0.0):.2f}")
    print(f" Max Drawdown:       {metrics.get('maximum_drawdown', 0.0) * 100.0:.2f}%")
    print(f" Completed Trades:   {metrics.get('trade_count', 0)}")
    print(f" Win Rate:           {metrics.get('win_rate', 0.0) * 100.0:.1f}%")
    print(f" Profit Factor:      {metrics.get('profit_factor', 0.0):.2f}")
    print(f" Total Fees:         ${metrics.get('total_fees', 0.0):,.2f}")
    print(f" Portfolio Turnover: {metrics.get('turnover', 0.0):.2f}x")
    print("-" * 70)

    trades_file = art_dir / "trades.parquet"
    if trades_file.is_file():
        trades_df = pd.read_parquet(trades_file)
        print(f" Roundtrip Trades:   {len(trades_df)} records loaded")
        if not trades_df.empty:
            print(" Recent 5 Trades:")
            print(
                trades_df[
                    ["symbol", "entry_time", "exit_time", "net_pnl", "exit_reason"]
                ]
                .tail(5)
                .to_string(index=False)
            )

    html_file = art_dir / "report.html"
    if html_file.is_file():
        print(f"\n Full HTML Report:   {html_file}")
    print("=" * 70)
    return 0


def _handle_walk_forward(args: argparse.Namespace) -> int:
    data_path = Path(args.dataset)
    if not data_path.is_file():
        print(f"Error: Dataset file not found: {data_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(data_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Identify features
    table = pq.read_table(data_path)
    schema_meta = table.schema.metadata or {}
    meta_json = schema_meta.get(b"trady_metadata")
    if meta_json:
        dataset_meta = json.loads(meta_json.decode("utf-8"))
        features = list(dataset_meta.get("feature_names", []))
    else:
        # Fallback: all numeric columns except target and metadata
        non_feat = {
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            args.target,
        }
        features = [
            c
            for c in df.columns
            if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])
        ]

    wf_config = WalkForwardConfig(
        n_folds=args.folds,
        window_type=args.window_type,
    )
    evaluator = WalkForwardEvaluator(config=wf_config)

    wf_result = evaluator.evaluate(
        data=df,
        feature_names=features,
        target_name=args.target,
        model_type=args.model_type,
        task_type="classification",
    )

    summary_df = wf_result.summary_table()
    print("=" * 70)
    print(f" WALK-FORWARD EVALUATION: {args.folds} FOLDS ({args.window_type})")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports") / "walk_forward" / f"wf_{timestamp_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f" Walk-forward summary saved to: {summary_path}")
    print("=" * 70)
    return 0
