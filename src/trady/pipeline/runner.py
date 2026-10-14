"""Orchestrator for TRADY End-to-End Quantitative Research Pipeline."""

import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trady.backtesting.config import BacktestConfig
from trady.backtesting.engine import BacktestEngine
from trady.backtesting.reporting import save_backtest_artifacts
from trady.backtesting.strategy import ModelDrivenStrategy
from trady.data.ingestion import download_and_ingest_market_data
from trady.features.config import FeatureConfig
from trady.features.pipeline import FeaturePipeline
from trady.models.base import TradyModel
from trady.models.experiment import ExperimentResult, run_model_experiment
from trady.pipeline.config import PipelineConfig
from trady.targets.config import TargetConfig
from trady.targets.pipeline import TargetPipeline
from trady.utils.logging import get_logger

logger = get_logger("pipeline.runner")


@dataclass(frozen=True)
class PipelineResult:
    """Comprehensive result of end-to-end quantitative research execution."""

    symbol: str
    status: str
    data_file: str
    feature_file: str
    target_file: str
    model_file: str
    experiment_id: str
    backtest_dir: str
    html_report: str
    model_metrics: dict[str, Any]
    backtest_metrics: dict[str, Any]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline result to serializable dict."""
        return asdict(self)

    def print_summary(self) -> None:
        """Print formatted research pipeline summary."""
        print("=" * 75)
        print(f" TRADY QUANTITATIVE RESEARCH PIPELINE: {self.symbol}")
        print("=" * 75)
        print(f" Status:             {self.status}")
        print(f" Execution Time:     {self.duration_seconds:.2f}s")
        print("-" * 75)
        print(" ARTIFACT PROVENANCE:")
        print(f"  - Market Data:     {self.data_file}")
        print(f"  - Feature Store:   {self.feature_file}")
        print(f"  - Target Dataset:  {self.target_file}")
        print(f"  - Serialized Model:{self.model_file}")
        print(f"  - Backtest Suite:  {self.backtest_dir}")
        print(f"  - HTML Report:     {self.html_report}")
        print("-" * 75)
        print(" OUT-OF-SAMPLE MODEL PERFORMANCE:")
        test_m = self.model_metrics.get("test", {})
        for k, v in test_m.items():
            if isinstance(v, (int, float)):
                print(f"  - {k:<18}: {v:.4f}")
        print("-" * 75)
        print(" PAPER BACKTEST SIMULATION PERFORMANCE:")
        bm = self.backtest_metrics
        tot_ret = bm.get("total_return", 0.0)
        cagr = bm.get("cagr", 0.0)
        sharpe = bm.get("sharpe_ratio", 0.0)
        sortino = bm.get("sortino_ratio", 0.0)
        max_dd = bm.get("max_drawdown", 0.0)
        win_rate = bm.get("win_rate", 0.0)
        profit_factor = bm.get("profit_factor", 0.0)
        trade_count = bm.get("total_trades", 0)

        print(f"  - Total Return:    {tot_ret:+.2%}")
        print(f"  - CAGR:            {cagr:+.2%}")
        print(f"  - Sharpe Ratio:    {sharpe:.2f}")
        print(f"  - Sortino Ratio:   {sortino:.2f}")
        print(f"  - Max Drawdown:    {max_dd:.2%}")
        print(f"  - Win Rate:        {win_rate:.1%}")
        print(f"  - Profit Factor:   {profit_factor:.2f}")
        print(f"  - Total Trades:    {trade_count}")
        print("=" * 75)


class PipelineRunner:
    """Coordinates the 5 stages of reproducible quantitative research."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self) -> PipelineResult:
        """Execute full quantitative research pipeline from raw data to backtest."""
        start_time = time.time()
        cfg = self.config

        print(
            "==========================================================================="
        )
        print(f" EXECUTING TRADY RESEARCH PIPELINE: {cfg.symbol} ({cfg.model_name})")
        print(
            "==========================================================================="
        )

        # ---------------------------------------------------------------------
        # STAGE 1: Market Data Ingestion & Validation
        # ---------------------------------------------------------------------
        print("\n[STAGE 1/5] Market Data Ingestion & Validation...")
        if cfg.dataset_path and Path(cfg.dataset_path).is_file():
            processed_data_path = Path(cfg.dataset_path)
            print(f"  -> Reusing existing validated dataset: {processed_data_path}")
        else:
            ingest_res = download_and_ingest_market_data(
                symbol=cfg.symbol,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                timeframe=cfg.timeframe,
                overwrite=cfg.overwrite,
                auto_adjust=cfg.auto_adjust,
            )
            processed_data_path = Path(ingest_res["processed_file"])
            print(
                f"  -> Downloaded and normalized: {processed_data_path} "
                f"({ingest_res['record_count']} bars)"
            )

        stem = processed_data_path.stem

        # ---------------------------------------------------------------------
        # STAGE 2: Causal Feature Engineering (43 Features)
        # ---------------------------------------------------------------------
        print("\n[STAGE 2/5] Computing Causal Feature Matrix...")
        feat_out_path = Path("data/features") / f"{stem}_features.parquet"
        feat_config = FeatureConfig(
            enabled_groups=cfg.feature_groups,
            nan_policy="keep",
            output_dir=str(feat_out_path.parent),
        )
        feat_pipeline = FeaturePipeline(config=feat_config)
        feat_table, feat_report = feat_pipeline.compute(processed_data_path)
        saved_feat_path = feat_pipeline.save(
            feat_table, feat_out_path, source_path=processed_data_path
        )
        print(
            f"  -> Features computed: {saved_feat_path} "
            f"({feat_report.feature_count} features, {feat_report.total_records} rows)"
        )

        # ---------------------------------------------------------------------
        # STAGE 3: Forward Target Generation & Alignment
        # ---------------------------------------------------------------------
        print(f"\n[STAGE 3/5] Generating Forward Targets ({cfg.target_name})...")
        target_out_path = Path("data/targets") / f"{stem}_labeled.parquet"
        target_config = TargetConfig(
            selected_targets=(cfg.target_name,),
            drop_unavailable_rows=True,
            output_dir=str(target_out_path.parent),
        )
        target_pipeline = TargetPipeline(config=target_config)
        modeling_ds, target_report = target_pipeline.compute(
            market_data=processed_data_path,
            features_data=saved_feat_path,
        )
        saved_target_path = modeling_ds.save(
            target_out_path, source_path=processed_data_path
        )
        print(
            f"  -> Target dataset generated: {saved_target_path} "
            f"({target_report.total_records} aligned rows)"
        )

        # ---------------------------------------------------------------------
        # STAGE 4: ML Model Training & Purged Temporal Validation
        # ---------------------------------------------------------------------
        print(f"\n[STAGE 4/5] Training ML Model ({cfg.model_name})...")
        ds = modeling_ds
        exp_res: ExperimentResult = run_model_experiment(
            dataset=ds,
            target_name=cfg.target_name,
            model_type=cfg.model_name,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            purge_gap=cfg.purge_gap,
            random_seed=cfg.seed,
            output_dir="models",
        )
        saved_model_path = (
            exp_res.saved_path or Path("models") / f"{exp_res.model_id}.joblib"
        )
        print(f"  -> Trained model saved: {saved_model_path}")
        test_acc = exp_res.metrics.get("test", {}).get("accuracy")
        if test_acc is not None:
            print(f"  -> Test Partition Accuracy: {test_acc:.2%}")

        # ---------------------------------------------------------------------
        # STAGE 5: Historical Paper Backtest Simulation & Risk Controls
        # ---------------------------------------------------------------------
        print("\n[STAGE 5/5] Running Event-Driven Backtest Simulation...")
        df = pd.read_parquet(saved_target_path)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        bt_config = BacktestConfig(
            initial_cash=cfg.initial_cash,
            commission_bps=cfg.commission_bps,
            fixed_fee_per_order=cfg.fixed_fee,
            slippage_bps=cfg.slippage_bps,
            holding_period_bars=cfg.holding_bars,
            stop_loss_pct=cfg.stop_loss,
            take_profit_pct=cfg.take_profit,
        )

        model: TradyModel = exp_res.model
        feature_cols = list(model.feature_names)
        X = df[feature_cols].to_numpy()
        X_clean = pd.DataFrame(X).ffill().bfill().to_numpy()

        if model.task_type == "classification":
            predictions = model.predict_proba(X_clean)[:, 1]
            is_classif = True
        else:
            predictions = model.predict(X_clean)
            is_classif = False

        strategy = ModelDrivenStrategy(
            config=bt_config,
            entry_threshold=cfg.entry_threshold,
            exit_threshold=cfg.exit_threshold,
            is_classification=is_classif,
        )

        engine = BacktestEngine(config=bt_config)
        bt_result = engine.run(
            data=df,
            strategy=strategy,
            predictions=predictions,
            close_positions_at_end=True,
        )

        timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backtest_out_dir = Path("reports") / "backtests" / f"run_{timestamp_str}"
        saved_files = save_backtest_artifacts(
            result=bt_result,
            output_dir=backtest_out_dir,
            extra_metadata={
                "symbol": cfg.symbol,
                "model_id": exp_res.model_id,
                "experiment_id": exp_res.experiment_id,
                "dataset_file": str(saved_target_path),
            },
        )
        print(f"  -> Backtest completed: {saved_files.get('report.html')}")

        duration = time.time() - start_time
        res = PipelineResult(
            symbol=cfg.symbol,
            status="SUCCESS",
            data_file=str(processed_data_path),
            feature_file=str(saved_feat_path),
            target_file=str(saved_target_path),
            model_file=str(saved_model_path),
            experiment_id=exp_res.experiment_id,
            backtest_dir=str(backtest_out_dir),
            html_report=str(saved_files.get("report.html", "")),
            model_metrics=exp_res.metrics,
            backtest_metrics=bt_result.metrics.to_dict(),
            duration_seconds=duration,
        )

        print("\n[OK] Quantitative Research Pipeline Finished Successfully!")
        res.print_summary()
        return res
