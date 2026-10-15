"""Unit and integration tests for TRADY Quantitative Research Pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from trady.assistant.tools import get_backtest_metrics
from trady.cli import build_parser
from trady.data.ingestion import BASKETS, download_and_ingest_multiple_symbols
from trady.pipeline.config import PipelineConfig
from trady.pipeline.runner import PipelineResult, PipelineRunner


def test_baskets_definition() -> None:
    """Verify predefined baskets exist and contain valid tickers."""
    assert "nifty50_top5" in BASKETS
    assert "^NSEI" in BASKETS["nifty50_top5"]
    assert "us_tech_top5" in BASKETS
    assert "SPY" in BASKETS["us_tech_top5"]


def test_multi_symbol_ingestion_mocked() -> None:
    """Verify batch ingestion loops over symbols and reports results."""
    mock_df = pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [105.0, 107.0],
            "Low": [98.0, 101.0],
            "Close": [104.0, 106.0],
            "Volume": [10000.0, 15000.0],
        },
        index=pd.to_datetime(["2023-01-02", "2023-01-03"]).tz_localize("UTC"),
    )

    with patch("yfinance.Ticker") as mock_ticker:
        inst = MagicMock()
        inst.history.return_value = mock_df
        mock_ticker.return_value = inst

        res = download_and_ingest_multiple_symbols(
            symbols=["TESTA", "TESTB"],
            start_date="2023-01-01",
            end_date="2023-01-04",
            preview_only=True,
        )

        assert len(res) == 2
        assert res[0]["symbol"] == "TESTA"
        assert res[0]["record_count"] == 2
        assert res[1]["symbol"] == "TESTB"
        assert res[1]["record_count"] == 2


def test_pipeline_runner_with_existing_dataset() -> None:
    """Verify end-to-end pipeline execution reusing existing validated data."""
    dataset_file = Path("data/processed/SPY_1d.parquet")
    assert dataset_file.exists(), (
        "SPY_1d.parquet required for pipeline integration test"
    )

    config = PipelineConfig(
        symbol="SPY",
        dataset_path=str(dataset_file),
        model_name="logistic_regression",
        target_name="target_binary_up_5d",
        seed=42,
    )

    runner = PipelineRunner(config=config)
    result = runner.run()

    assert isinstance(result, PipelineResult)
    assert result.status == "SUCCESS"
    assert result.symbol == "SPY"
    assert Path(result.data_file).exists()
    assert Path(result.feature_file).exists()
    assert Path(result.target_file).exists()
    assert Path(result.model_file).exists()
    assert Path(result.backtest_dir).exists()
    assert Path(result.html_report).exists()

    # Check metrics
    assert "test" in result.model_metrics
    assert "accuracy" in result.model_metrics["test"]
    assert "total_return" in result.backtest_metrics
    assert "sharpe_ratio" in result.backtest_metrics
    assert result.duration_seconds > 0


def test_pipeline_cli_parser() -> None:
    """Verify CLI parser handles pipeline arguments."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "pipeline",
            "run",
            "--symbol",
            "SPY",
            "--dataset",
            "data/processed/SPY_1d.parquet",
            "--model",
            "logistic_regression",
            "--target",
            "target_binary_up_5d",
        ]
    )
    assert args.command == "pipeline"
    assert args.pipeline_subcommand == "run"
    assert args.symbol == "SPY"
    assert args.model == "logistic_regression"


def test_assistant_retrieves_nested_backtest_metrics(tmp_path: Path) -> None:
    """Verify assistant tool extracts real metrics from nested metrics.json."""
    backtest_dir = tmp_path / "reports" / "backtests" / "run_20261014_120000"
    backtest_dir.mkdir(parents=True)
    (backtest_dir / "report.html").write_text("<html>Report</html>", encoding="utf-8")
    import json

    sample_metrics = {
        "total_return": 0.125,
        "annualized_return": 0.25,
        "sharpe_ratio": 1.45,
        "sortino_ratio": 2.10,
        "max_drawdown": 0.085,
        "win_rate": 0.70,
        "profit_factor": 2.15,
        "total_trades": 12,
    }
    (backtest_dir / "metrics.json").write_text(
        json.dumps(sample_metrics), encoding="utf-8"
    )

    info = get_backtest_metrics(project_dir=tmp_path)
    assert info["status"] == "success"
    assert info["total_html_reports"] >= 1
    rep = info["reports"][0]
    assert "metrics" in rep
    assert rep["metrics"]["sharpe_ratio"] == 1.45
    assert rep["metrics"]["total_trades"] == 12


def test_pipeline_runner_strictly_out_of_sample_and_risk_controls() -> None:
    """Verify automated pipeline evaluates strictly out-of-sample with risk."""
    import json

    import pytest

    dataset_file = Path("data/processed/SPY_1d.parquet")
    if not dataset_file.exists():
        pytest.skip("SPY_1d.parquet required for pipeline integration test")

    config = PipelineConfig(
        symbol="SPY",
        dataset_path=str(dataset_file),
        model_name="logistic_regression",
        target_name="target_binary_up_5d",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        max_drawdown_limit=0.15,
        max_position_concentration=0.20,
        seed=42,
    )

    runner = PipelineRunner(config=config)
    result = runner.run()

    assert result.status == "SUCCESS"
    bt_dir = Path(result.backtest_dir)
    cfg_file = bt_dir / "configuration.json"
    assert cfg_file.exists()

    with open(cfg_file, encoding="utf-8") as f:
        cfg_data = json.load(f)

    # 1. Verify strict out-of-sample evaluation metadata
    extra = cfg_data.get("extra_metadata", {})
    assert extra.get("evaluation_mode") == "STRICT_OUT_OF_SAMPLE"
    assert extra.get("test_bars") > 0
    assert extra.get("train_bars") > extra.get("test_bars")

    # 2. Verify equity curve bars match test slice count exactly
    equity_df = pd.read_parquet(bt_dir / "equity_curve.parquet")
    assert len(equity_df) == extra.get("test_bars")

    # 3. Verify risk configuration reached the engine and was serialized
    bt_cfg = cfg_data.get("backtest_config", {})
    assert bt_cfg.get("risk_config") is not None
    assert bt_cfg["risk_config"]["max_drawdown_limit"] == 0.15
    assert bt_cfg["risk_config"]["max_position_concentration"] == 0.20


def test_pipeline_runner_insufficient_test_data_raises() -> None:
    """Verify that insufficient test data produces a clear error."""
    import pytest

    dataset_file = Path("data/processed/SPY_1d.parquet")
    if not dataset_file.exists():
        pytest.skip("SPY_1d.parquet required for pipeline test")

    # Extreme train ratio leaving only 1-2 test bars
    config = PipelineConfig(
        symbol="SPY",
        dataset_path=str(dataset_file),
        train_ratio=0.98,
        val_ratio=0.01,
        test_ratio=0.01,
    )
    runner = PipelineRunner(config=config)
    with pytest.raises(
        ValueError, match="Insufficient out-of-sample test observations"
    ):
        runner.run()


def test_pipeline_config_invariants_raise_on_invalid() -> None:
    """Verify invalid split ratios or risk bounds raise immediate ValueError."""
    import pytest

    # Ratios do not sum to 1.0
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        PipelineConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.3)

    # Invalid concentration
    with pytest.raises(ValueError, match="max_position_concentration"):
        PipelineConfig(max_position_concentration=-0.5)

    # Invalid drawdown limit
    with pytest.raises(ValueError, match="max_drawdown_limit"):
        PipelineConfig(max_drawdown_limit=1.5)
