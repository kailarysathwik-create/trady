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
    max_position_concentration: float = 0.25  # Max 25% single position concentration
    max_portfolio_exposure: float = 1.0  # Max 100% gross exposure (unleveraged)
    max_drawdown_limit: float | None = (
        0.20  # 20% drawdown circuit breaker active by default
    )
    cash_buffer_pct: float = 0.01  # 1% cash buffer for fees
    risk_config: object | None = None

    def __post_init__(self) -> None:
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-4:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio:.4f} "
                f"({self.train_ratio} + {self.val_ratio} + {self.test_ratio})"
            )
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0.0 and 1.0.")
        if not (0.0 < self.test_ratio < 1.0):
            raise ValueError("test_ratio must be between 0.0 and 1.0.")
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError("val_ratio must be between 0.0 and 1.0.")
        if not (0.0 < self.max_position_concentration <= 1.0):
            raise ValueError("max_position_concentration must be in (0.0, 1.0].")
        if self.max_portfolio_exposure <= 0:
            raise ValueError("max_portfolio_exposure must be strictly positive.")
        if self.max_drawdown_limit is not None and not (
            0.0 < self.max_drawdown_limit < 1.0
        ):
            raise ValueError("max_drawdown_limit must be in (0.0, 1.0) if enabled.")
        if not (0.0 <= self.cash_buffer_pct < 0.5):
            raise ValueError("cash_buffer_pct must be in [0.0, 0.5).")
