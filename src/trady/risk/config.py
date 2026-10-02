"""Configuration schemas for the TRADY Risk Engine."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RiskConfig:
    """Configurable research risk constraints and position sizing rules."""

    max_position_exposure: float | None = None
    max_position_concentration: float = 1.0  # Max single position concentration (1.0 = 100%)
    max_portfolio_exposure: float = 1.0  # Max 100% gross exposure (unleveraged)
    max_drawdown_limit: float | None = (
        None  # Drawdown circuit breaker (e.g. 0.15 for 15%)
    )
    sizing_method: str = "fractional_equity"  # Sizing method identifier
    target_position_pct: float = 0.10  # 10% target allocation
    fixed_cash_amount: float = 10_000.0  # Dollar allocation for fixed cash sizing
    target_risk_pct: float = 0.01  # 1% equity risk per trade for ATR sizing
    atr_multiplier: float = 2.0  # ATR stop distance multiplier
    target_volatility: float = 0.15  # 15% annualized volatility target
    allow_shorting: bool = False  # Long-only by default
    cash_buffer_pct: float = 0.01  # 1% cash buffer for execution fees

    def __post_init__(self) -> None:
        if self.max_position_exposure is not None and self.max_position_exposure <= 0:
            raise ValueError("max_position_exposure must be strictly positive.")
        if not (0.0 < self.max_position_concentration <= 1.0):
            raise ValueError("max_position_concentration must be in (0.0, 1.0].")
        if self.max_portfolio_exposure <= 0:
            raise ValueError("max_portfolio_exposure must be strictly positive.")
        if self.max_drawdown_limit is not None and not (
            0.0 < self.max_drawdown_limit < 1.0
        ):
            raise ValueError("max_drawdown_limit must be in (0.0, 1.0).")
        if not (0.0 < self.target_position_pct <= 1.0):
            raise ValueError("target_position_pct must be in (0.0, 1.0].")
        if self.fixed_cash_amount <= 0:
            raise ValueError("fixed_cash_amount must be strictly positive.")
        if not (0.0 < self.target_risk_pct <= 1.0):
            raise ValueError("target_risk_pct must be in (0.0, 1.0].")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be strictly positive.")
        if self.target_volatility <= 0:
            raise ValueError("target_volatility must be strictly positive.")
        if not (0.0 <= self.cash_buffer_pct < 0.5):
            raise ValueError("cash_buffer_pct must be in [0.0, 0.5).")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        return asdict(self)
