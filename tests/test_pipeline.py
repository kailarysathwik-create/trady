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
