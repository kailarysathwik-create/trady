"""Configuration model for end-to-end TRADY quantitative research pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Configures multi-stage quantitative pipeline execution."""

    symbol: str = "^NSEI"
    start_date: str | None = None
    end_date: str | None = None
    timeframe: str = "1d"
    dataset_path: str | None = None
    model_name: str = "logistic_regression"
    target_name: str = "target_binary_up_5d"
    feature_groups: tuple[str, ...] = (
        "PRICE",
        "MOMENTUM",
        "VOLATILITY",
        "VOLUME",
        "PRICE_STRUCTURE",
    )
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    purge_gap: int | None = None
    seed: int = 42
    initial_cash: float = 100_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    fixed_fee: float = 1.0
    entry_threshold: float = 0.5
    exit_threshold: float = 0.5
    holding_bars: int | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    output_dir: str = "reports/pipeline"
    overwrite: bool = False
    auto_adjust: bool = False
