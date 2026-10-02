"""Position sizing rules and volatility-based risk allocation."""

import math

from trady.risk.config import RiskConfig


class PositionSizer:
    """Computes target order share quantities based on configured sizing rules."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def calculate_size(
        self,
        price: float,
        equity: float,
        cash: float,
        atr: float | None = None,
        volatility: float | None = None,
    ) -> float:
        """Compute recommended position quantity.

        Args:
            price: Current asset price. Must be strictly positive.
            equity: Current mark-to-market portfolio equity.
            cash: Current liquid cash balance.
            atr: Optional Average True Range (for ATR risk sizing).
            volatility: Optional annualized return volatility (for vol targeting).

        Returns:
            Computed target share quantity (float >= 0.0).
        """
        if price <= 0:
            return 0.0
        if equity <= 0:
            return 0.0

        method = self.config.sizing_method

        if method == "fixed_cash":
            target_cash = min(
                self.config.fixed_cash_amount,
                cash * (1.0 - self.config.cash_buffer_pct),
            )
            return max(0.0, target_cash / price)

        elif method == "atr_risk":
            if atr is not None and atr > 0:
                risk_capital = equity * self.config.target_risk_pct
                stop_distance = atr * self.config.atr_multiplier
                if stop_distance > 0:
                    raw_qty = risk_capital / stop_distance
                    # Bound raw quantity so total value does not exceed available cash
                    max_affordable_qty = (
                        cash * (1.0 - self.config.cash_buffer_pct)
                    ) / price
                    return max(0.0, min(raw_qty, max_affordable_qty))
            # Fallback if ATR is not provided or zero
            target_val = equity * self.config.target_position_pct
            return max(0.0, target_val / price)

        elif method == "volatility_target":
            if volatility is not None and volatility > 0 and not math.isnan(volatility):
                # Volatility parity: weight = target_vol / asset_vol
                vol_weight = self.config.target_volatility / volatility
                # Cap weight at max_position_concentration
                vol_weight = min(vol_weight, self.config.max_position_concentration)
                target_val = equity * vol_weight
                max_affordable_val = cash * (1.0 - self.config.cash_buffer_pct)
                target_val = min(target_val, max_affordable_val)
                return max(0.0, target_val / price)
            # Fallback to fractional equity if vol unavailable
            target_val = equity * self.config.target_position_pct
            return max(0.0, target_val / price)

        else:  # "fractional_equity" (default)
            target_val = equity * self.config.target_position_pct
            max_affordable_val = cash * (1.0 - self.config.cash_buffer_pct)
            target_val = min(target_val, max_affordable_val)
            return max(0.0, target_val / price)
