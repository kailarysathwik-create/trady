"""Portfolio state tracking, mark-to-market accounting, and trade logging."""

import uuid
from dataclasses import dataclass
from typing import Any

from trady.backtesting.execution import ExecutionResult


@dataclass
class Position:
    """Current open position in an instrument."""

    symbol: str
    quantity: float
    avg_entry_price: float
    entry_time: Any
    entry_bar_idx: int
    entry_fee: float


@dataclass(frozen=True)
class TradeRecord:
    """Complete roundtrip trade record with PnL, duration, and fees."""

    trade_id: str
    symbol: str
    entry_time: Any
    exit_time: Any
    side: str  # "LONG"
    quantity: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    total_fees: float
    net_pnl: float
    pnl_pct: float
    holding_bars: int
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert trade record to dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time),
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "gross_pnl": self.gross_pnl,
            "total_fees": self.total_fees,
            "net_pnl": self.net_pnl,
            "pnl_pct": self.pnl_pct,
            "holding_bars": self.holding_bars,
            "exit_reason": self.exit_reason,
        }


@dataclass(frozen=True)
class EquityPoint:
    """Snapshot of portfolio equity and exposure at timestamp T."""

    timestamp: Any
    cash: float
    positions_value: float
    portfolio_equity: float
    daily_return: float
    drawdown: float
    gross_exposure: float
    net_exposure: float

    def to_dict(self) -> dict[str, Any]:
        """Convert equity snapshot to dictionary."""
        return {
            "timestamp": str(self.timestamp),
            "cash": self.cash,
            "positions_value": self.positions_value,
            "portfolio_equity": self.portfolio_equity,
            "daily_return": self.daily_return,
            "drawdown": self.drawdown,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
        }


class PortfolioTracker:
    """Manages cash, open positions, completed trades, and equity curve."""

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[EquityPoint] = []
        self.high_water_mark: float = initial_cash

    @property
    def current_equity(self) -> float:
        """Current equity based on last mark-to-market."""
        if self.equity_curve:
            return self.equity_curve[-1].portfolio_equity
        return self.cash

    def has_position(self, symbol: str) -> bool:
        """Return True if symbol currently has an open position."""
        return symbol in self.positions and self.positions[symbol].quantity > 0

    def get_position_quantity(self, symbol: str) -> float:
        """Return currently held quantity of symbol."""
        if symbol in self.positions:
            return self.positions[symbol].quantity
        return 0.0

    def apply_execution(
        self,
        exec_res: ExecutionResult,
        bar_idx: int,
        exit_reason: str = "signal",
    ) -> None:
        """Update portfolio state following order execution."""
        sym = exec_res.symbol

        if exec_res.action == "BUY":
            # Deduct cash (traded_value + fee)
            total_cost = exec_res.traded_value + exec_res.fee
            self.cash -= total_cost

            if sym in self.positions and self.positions[sym].quantity > 0:
                # Add to existing position
                pos = self.positions[sym]
                total_qty = pos.quantity + exec_res.filled_quantity
                new_avg_price = (
                    (pos.quantity * pos.avg_entry_price)
                    + (exec_res.filled_quantity * exec_res.fill_price)
                ) / total_qty
                pos.quantity = total_qty
                pos.avg_entry_price = new_avg_price
                pos.entry_fee += exec_res.fee
            else:
                # Open new position
                self.positions[sym] = Position(
                    symbol=sym,
                    quantity=exec_res.filled_quantity,
                    avg_entry_price=exec_res.fill_price,
                    entry_time=exec_res.timestamp,
                    entry_bar_idx=bar_idx,
                    entry_fee=exec_res.fee,
                )

        elif exec_res.action == "SELL":
            # Realize sale proceeds (traded_value - fee)
            net_proceeds = exec_res.traded_value - exec_res.fee
            self.cash += net_proceeds

            if sym in self.positions and self.positions[sym].quantity > 0:
                pos = self.positions[sym]
                close_qty = min(pos.quantity, exec_res.filled_quantity)

                gross_pnl = close_qty * (exec_res.fill_price - pos.avg_entry_price)
                # Pro-rate entry fee if partial close
                entry_fee_share = pos.entry_fee * (close_qty / pos.quantity)
                total_fees = entry_fee_share + exec_res.fee
                net_pnl = gross_pnl - total_fees
                pnl_pct = (
                    net_pnl / (close_qty * pos.avg_entry_price)
                    if pos.avg_entry_price > 0
                    else 0.0
                )
                holding_bars = bar_idx - pos.entry_bar_idx

                trade = TradeRecord(
                    trade_id=f"trd_{uuid.uuid4().hex[:8]}",
                    symbol=sym,
                    entry_time=pos.entry_time,
                    exit_time=exec_res.timestamp,
                    side="LONG",
                    quantity=close_qty,
                    entry_price=pos.avg_entry_price,
                    exit_price=exec_res.fill_price,
                    gross_pnl=gross_pnl,
                    total_fees=total_fees,
                    net_pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    holding_bars=holding_bars,
                    exit_reason=exit_reason,
                )
                self.trades.append(trade)

                # Update remaining position
                remaining_qty = pos.quantity - close_qty
                if remaining_qty <= 1e-7:
                    del self.positions[sym]
                else:
                    pos.quantity = remaining_qty
                    pos.entry_fee -= entry_fee_share

    def mark_to_market(
        self,
        timestamp: Any,
        current_prices: dict[str, float],
    ) -> EquityPoint:
        """Calculate mark-to-market equity and drawdown at end of bar."""
        positions_val = 0.0
        gross_exp = 0.0
        net_exp = 0.0

        for sym, pos in self.positions.items():
            if sym in current_prices and pos.quantity > 0:
                val = pos.quantity * current_prices[sym]
                positions_val += val
                gross_exp += abs(val)
                net_exp += val

        equity = self.cash + positions_val

        # Drawdown calculation
        if equity > self.high_water_mark:
            self.high_water_mark = equity
        drawdown = (
            (equity - self.high_water_mark) / self.high_water_mark
            if self.high_water_mark > 0
            else 0.0
        )

        # Daily return relative to previous equity
        if self.equity_curve:
            prev_equity = self.equity_curve[-1].portfolio_equity
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        else:
            daily_ret = (
                (equity - self.initial_cash) / self.initial_cash
                if self.initial_cash > 0
                else 0.0
            )

        pt = EquityPoint(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=positions_val,
            portfolio_equity=equity,
            daily_return=daily_ret,
            drawdown=drawdown,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
        )
        self.equity_curve.append(pt)
        return pt
