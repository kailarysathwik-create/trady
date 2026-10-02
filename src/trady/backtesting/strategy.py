"""Strategy abstraction and model-driven strategy implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from trady.backtesting.config import BacktestConfig
from trady.backtesting.portfolio import Position


@dataclass(frozen=True)
class Order:
    """Virtual trading order generated at bar T to be executed at bar T+1."""

    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: float
    order_type: str = "MARKET_OPEN"
    exit_reason: str = "signal"


class BaseStrategy(ABC):
    """Abstract base class for backtesting strategies."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        bar_idx: int,
        bar_data: dict[str, Any],
        prediction: float | None,
        position: Position | None,
        cash: float,
        current_equity: float,
    ) -> Order | None:
        """Evaluate market state and optional model prediction at bar T.

        Args:
            bar_idx: Zero-based index of current bar T.
            bar_data: Bar T OHLCV and timestamp.
            prediction: Optional model prediction at bar T (using data <= T).
            position: Currently open position for symbol, if any.
            cash: Current liquid cash balance.
            current_equity: Current mark-to-market portfolio equity.

        Returns:
            Order to be executed at bar T+1, or None if no action.
        """
        pass


class ModelDrivenStrategy(BaseStrategy):
    """Strategy driven by model forecasts with risk management rules.

    Evaluates:
    1. Stop-loss: Exit if unrealized loss exceeds stop_loss_pct.
    2. Take-profit: Exit if unrealized gain exceeds take_profit_pct.
    3. Holding period: Exit if holding duration >= holding_period_bars.
    4. Model signal:
       - Entry: If no position and prediction >= entry_threshold.
       - Exit: If in position and prediction <= exit_threshold.
    """

    def __init__(
        self,
        config: BacktestConfig,
        entry_threshold: float = 0.5,
        exit_threshold: float = 0.5,
        is_classification: bool = True,
    ) -> None:
        super().__init__(config)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.is_classification = is_classification

    def _calculate_order_quantity(
        self,
        close_price: float,
        cash: float,
        current_equity: float,
    ) -> float:
        """Compute target order quantity based on position sizing config."""
        if close_price <= 0:
            return 0.0

        if self.config.position_size_type == "fractional_equity":
            target_val = current_equity * self.config.position_size_value
        elif self.config.position_size_type == "fixed_cash":
            target_val = min(self.config.position_size_value, cash * 0.99)
        else:
            target_val = cash * 0.95

        # Bound by available cash (accounting for approx fees)
        target_val = min(target_val, cash * 0.99)
        if target_val <= 0:
            return 0.0

        return target_val / close_price

    def evaluate(
        self,
        bar_idx: int,
        bar_data: dict[str, Any],
        prediction: float | None,
        position: Position | None,
        cash: float,
        current_equity: float,
    ) -> Order | None:
        """Evaluate strategy rules at bar T."""
        symbol = bar_data["symbol"]
        close_price = float(bar_data["close"])

        # Check existing position for risk exits
        if position is not None and position.quantity > 0:
            # 1. Stop loss check
            if self.config.stop_loss_pct is not None:
                pnl_pct = (close_price - position.avg_entry_price) / (
                    position.avg_entry_price
                )
                if pnl_pct <= -self.config.stop_loss_pct:
                    return Order(
                        symbol=symbol,
                        action="SELL",
                        quantity=position.quantity,
                        exit_reason="stop_loss",
                    )

            # 2. Take profit check
            if self.config.take_profit_pct is not None:
                pnl_pct = (close_price - position.avg_entry_price) / (
                    position.avg_entry_price
                )
                if pnl_pct >= self.config.take_profit_pct:
                    return Order(
                        symbol=symbol,
                        action="SELL",
                        quantity=position.quantity,
                        exit_reason="take_profit",
                    )

            # 3. Holding period check
            if self.config.holding_period_bars is not None:
                bars_held = bar_idx - position.entry_bar_idx
                if bars_held >= self.config.holding_period_bars:
                    return Order(
                        symbol=symbol,
                        action="SELL",
                        quantity=position.quantity,
                        exit_reason="max_holding_period",
                    )

            # 4. Model exit signal
            if prediction is not None:
                should_exit = False
                if self.is_classification and prediction <= self.exit_threshold:
                    should_exit = True
                elif not self.is_classification and prediction <= self.exit_threshold:
                    should_exit = True

                if should_exit:
                    return Order(
                        symbol=symbol,
                        action="SELL",
                        quantity=position.quantity,
                        exit_reason="signal_exit",
                    )

            return None

        # No open position: check entry condition
        if prediction is not None:
            should_enter = False
            if self.is_classification and prediction >= self.entry_threshold:
                should_enter = True
            elif not self.is_classification and prediction >= self.entry_threshold:
                should_enter = True

            if should_enter:
                qty = self._calculate_order_quantity(
                    close_price=close_price,
                    cash=cash,
                    current_equity=current_equity,
                )
                if qty > 0:
                    return Order(
                        symbol=symbol,
                        action="BUY",
                        quantity=qty,
                        exit_reason="signal_entry",
                    )

        return None
