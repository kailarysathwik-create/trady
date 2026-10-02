"""Configuration schemas for TRADY historical backtesting simulations."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters governing simulation capital, frictions, execution, and risk."""

    initial_cash: float = 100_000.0
    commission_bps: float = 5.0  # 5 bps = 0.0005 (0.05%)
    fixed_fee_per_order: float = 1.0  # $1 fixed fee per order
    slippage_bps: float = 5.0  # 5 bps = 0.0005 (0.05%)
    max_volume_pct: float = 0.10  # Max 10% of bar volume per trade
    position_size_type: str = "fractional_equity"  # "fractional_equity" | "fixed_cash"
    position_size_value: float = 0.95  # 95% allocation (leaves 5% cash buffer)
    max_gross_leverage: float = 1.0  # Gross exposure limit (1.0 = unleveraged)
    allow_shorting: bool = False  # Long-only by default
    holding_period_bars: int | None = None  # Optional fixed holding duration
    stop_loss_pct: float | None = None  # E.g. 0.05 for 5% trailing/fixed stop
    take_profit_pct: float | None = None  # E.g. 0.10 for 10% profit target
    risk_free_rate: float = 0.02  # 2.0% annual benchmark for Sharpe / Sortino
    periods_per_year: int = 252  # Trading days per year
    random_seed: int = 42
    risk_config: Any | None = None

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be strictly positive.")
        if self.commission_bps < 0:
            raise ValueError("commission_bps cannot be negative.")
        if self.fixed_fee_per_order < 0:
            raise ValueError("fixed_fee_per_order cannot be negative.")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative.")
        if not (0.0 < self.max_volume_pct <= 1.0):
            raise ValueError("max_volume_pct must be in (0.0, 1.0].")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be strictly positive.")

    @property
    def commission_rate(self) -> float:
        """Commission as decimal fraction (e.g. 5 bps -> 0.0005)."""
        return self.commission_bps / 10_000.0

    @property
    def slippage_rate(self) -> float:
        """Slippage as decimal fraction (e.g. 5 bps -> 0.0005)."""
        return self.slippage_bps / 10_000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        return asdict(self)
