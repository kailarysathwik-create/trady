"""Data models for TRADE9 Virtual Money and Paper Exchange."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Currency(StrEnum):
    """Supported virtual currencies for isolated paper trading."""

    INR = "INR"
    USD = "USD"

    @property
    def symbol(self) -> str:
        return "₹" if self == Currency.INR else "$"


class OrderAction(StrEnum):
    """Direction for paper orders."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Execution type for paper orders."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    """Lifecycle status for paper orders."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Position:
    """An open virtual holding in a specific instrument."""

    symbol: str
    currency: Currency
    quantity: float
    average_entry_price: float
    current_price: float
    realized_pnl: float = 0.0
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def cost_basis(self) -> float:
        """Total virtual capital committed to holding."""
        return self.quantity * self.average_entry_price

    @property
    def market_value(self) -> float:
        """Current virtual valuation based on latest market price."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Paper profit or loss at current market price."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Percentage unrealized gain or loss."""
        if self.cost_basis <= 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "currency": self.currency.value,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "current_price": self.current_price,
            "cost_basis": self.cost_basis,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "realized_pnl": self.realized_pnl,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        return cls(
            symbol=data["symbol"],
            currency=Currency(data["currency"]),
            quantity=float(data["quantity"]),
            average_entry_price=float(data["average_entry_price"]),
            current_price=float(data.get("current_price", data["average_entry_price"])),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
        )


@dataclass
class PaperOrder:
    """A virtual order submitted to the paper exchange simulator."""

    order_id: str
    symbol: str
    currency: Currency
    action: OrderAction
    order_type: OrderType
    quantity: float
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    filled_at: str | None = None
    filled_price: float | None = None
    filled_quantity: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    reserved_cash: float = 0.0
    rejection_reason: str | None = None
    # Immutable audit trail preserving the exact AI forecast and risk decision
    forecast_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["currency"] = self.currency.value
        d["action"] = self.action.value
        d["order_type"] = self.order_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperOrder":
        return cls(
            order_id=data["order_id"],
            symbol=data["symbol"],
            currency=Currency(data["currency"]),
            action=OrderAction(data["action"]),
            order_type=OrderType(data["order_type"]),
            quantity=float(data["quantity"]),
            limit_price=(
                float(data["limit_price"])
                if data.get("limit_price") is not None
                else None
            ),
            status=OrderStatus(data["status"]),
            created_at=data["created_at"],
            filled_at=data.get("filled_at"),
            filled_price=(
                float(data["filled_price"])
                if data.get("filled_price") is not None
                else None
            ),
            filled_quantity=float(data.get("filled_quantity", 0.0)),
            fees=float(data.get("fees", 0.0)),
            slippage=float(data.get("slippage", 0.0)),
            reserved_cash=float(data.get("reserved_cash", 0.0)),
            rejection_reason=data.get("rejection_reason"),
            forecast_metadata=data.get("forecast_metadata"),
        )


@dataclass
class VirtualWallet:
    """Persistent, isolated paper wallet denominated in a specific currency."""

    currency: Currency
    initial_balance: float
    available_cash: float
    reserved_cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    orders: list[PaperOrder] = field(default_factory=list)
    total_fees: float = 0.0
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    equity_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.peak_equity <= 0.0:
            self.peak_equity = self.total_equity

    @property
    def positions_market_value(self) -> float:
        """Sum of market values for all open positions."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_cost_basis(self) -> float:
        """Sum of cost bases for all open positions."""
        return sum(pos.cost_basis for pos in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized paper profit/loss across all holdings."""
        return self.positions_market_value - self.total_cost_basis

    @property
    def total_cash(self) -> float:
        """Available cash plus cash reserved for pending limit orders."""
        return self.available_cash + self.reserved_cash

    @property
    def total_equity(self) -> float:
        """Total virtual portfolio value: total cash + open positions value."""
        return self.total_cash + self.positions_market_value

    @property
    def total_return_pct(self) -> float:
        """Cumulative return percentage relative to initial balance."""
        if self.initial_balance <= 0:
            return 0.0
        return (self.total_equity - self.initial_balance) / self.initial_balance

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from the peak equity value."""
        if self.peak_equity <= 0:
            return 0.0
        dd = (self.peak_equity - self.total_equity) / self.peak_equity
        return max(0.0, dd)

    def record_equity_snapshot(self) -> None:
        """Record a timestamped equity point and update peak and drawdown."""
        curr_eq = self.total_equity
        if curr_eq > self.peak_equity:
            self.peak_equity = curr_eq

        dd = self.current_drawdown
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        self.equity_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "available_cash": self.available_cash,
                "reserved_cash": self.reserved_cash,
                "positions_market_value": self.positions_market_value,
                "total_equity": curr_eq,
                "unrealized_pnl": self.total_unrealized_pnl,
                "realized_pnl": self.realized_pnl,
                "total_fees": self.total_fees,
                "drawdown": dd,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency.value,
            "initial_balance": self.initial_balance,
            "available_cash": self.available_cash,
            "cash_balance": self.available_cash,
            "reserved_cash": self.reserved_cash,
            "positions": {k: pos.to_dict() for k, pos in self.positions.items()},
            "orders": [ord.to_dict() for ord in self.orders],
            "total_fees": self.total_fees,
            "realized_pnl": self.realized_pnl,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "positions_market_value": self.positions_market_value,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_equity": self.total_equity,
            "total_return_pct": self.total_return_pct,
            "current_drawdown": self.current_drawdown,
            "equity_history": self.equity_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VirtualWallet":
        currency = Currency(data["currency"])
        positions = {
            k: Position.from_dict(v) for k, v in data.get("positions", {}).items()
        }
        orders = [PaperOrder.from_dict(v) for v in data.get("orders", [])]
        wallet = cls(
            currency=currency,
            initial_balance=float(data["initial_balance"]),
            available_cash=float(data["available_cash"]),
            reserved_cash=float(data.get("reserved_cash", 0.0)),
            positions=positions,
            orders=orders,
            total_fees=float(data.get("total_fees", 0.0)),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            peak_equity=float(data.get("peak_equity", 0.0)),
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            equity_history=data.get("equity_history", []),
        )
        return wallet
