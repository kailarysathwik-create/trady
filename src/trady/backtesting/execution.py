"""Execution simulator implementing realistic market frictions.

Enforces:
1. Next-bar execution: signals generated at bar T close execute at bar T+1 open.
2. Bid/Ask slippage and market impact.
3. Realistic commissions (proportional bps + fixed ticket fee).
4. Physical price range clamping: fill price cannot exceed [low, high] of the bar.
5. Bar volume liquidity constraints: cannot trade more than max_volume_pct.
"""

from dataclasses import dataclass
from typing import Any

from trady.backtesting.config import BacktestConfig


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of an executed virtual order."""

    order_id: str
    symbol: str
    timestamp: Any
    action: str  # "BUY" or "SELL"
    requested_quantity: float
    filled_quantity: float
    base_price: float
    fill_price: float
    fee: float
    slippage_cost: float
    traded_value: float

    @property
    def total_cost(self) -> float:
        """Total cash outlay required for BUY, or net proceeds from SELL."""
        if self.action == "BUY":
            return self.traded_value + self.fee
        else:
            return self.traded_value - self.fee


class ExecutionSimulator:
    """Simulates market fills incorporating transaction costs and slippage."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def execute_order(
        self,
        order_id: str,
        symbol: str,
        action: str,
        quantity: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        timestamp: Any,
    ) -> ExecutionResult:
        """Simulate next-bar execution at open with slippage and clamping.

        Args:
            order_id: Unique identifier of the order.
            symbol: Ticker symbol.
            action: "BUY" or "SELL".
            quantity: Requested share/unit quantity.
            bar_open: Open price of the execution bar.
            bar_high: High price of the execution bar.
            bar_low: Low price of the execution bar.
            bar_close: Close price of the execution bar.
            bar_volume: Total traded volume of the execution bar.
            timestamp: Timestamp of the execution bar.

        Returns:
            ExecutionResult containing exact fill details, fees, and prices.
        """
        if quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {quantity}")

        # 1. Volume liquidity constraint
        max_allowed_qty = bar_volume * self.config.max_volume_pct
        filled_qty = min(quantity, max_allowed_qty)

        # 2. Slippage application on open price
        base_price = bar_open
        slip_rate = self.config.slippage_rate

        if action == "BUY":
            # Buy orders execute at higher ask price
            raw_fill_price = base_price * (1.0 + slip_rate)
            # Cannot execute higher than the bar's high
            fill_price = min(raw_fill_price, bar_high)
        elif action == "SELL":
            # Sell orders execute at lower bid price
            raw_fill_price = base_price * (1.0 - slip_rate)
            # Cannot execute lower than the bar's low
            fill_price = max(raw_fill_price, bar_low)
        else:
            msg = f"Invalid order action: '{action}'. Must be 'BUY' or 'SELL'."
            raise ValueError(msg)

        # 3. Traded value and fee calculation
        traded_value = filled_qty * fill_price
        slippage_cost = filled_qty * abs(fill_price - base_price)

        proportional_fee = traded_value * self.config.commission_rate
        total_fee = proportional_fee + self.config.fixed_fee_per_order

        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            requested_quantity=quantity,
            filled_quantity=filled_qty,
            base_price=base_price,
            fill_price=fill_price,
            fee=total_fee,
            slippage_cost=slippage_cost,
            traded_value=traded_value,
        )
