"""Risk decision records and transparent audit trail."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RiskDecision:
    """Immutable audit record detailing the risk engine's decision for a proposed order.

    Explicitly records whether the order was ACCEPTED, REDUCED, or REJECTED,
    including the triggered constraint, adjusted quantities, and mathematical snapshot.
    """

    decision_id: str
    timestamp: Any
    symbol: str
    action: str  # "BUY" or "SELL"
    status: str  # "ACCEPTED", "REDUCED", "REJECTED"
    requested_quantity: float
    approved_quantity: float
    requested_value: float
    approved_value: float
    price: float
    reason: str
    constraint_triggered: str | None
    current_equity: float
    current_drawdown: float
    portfolio_exposure_pct: float
    position_concentration_pct: float

    def __post_init__(self) -> None:
        if self.status not in ("ACCEPTED", "REDUCED", "REJECTED"):
            raise ValueError(
                f"Invalid RiskDecision status '{self.status}'. "
                "Must be 'ACCEPTED', 'REDUCED', or 'REJECTED'."
            )
        if self.approved_quantity < 0:
            raise ValueError("approved_quantity cannot be negative.")

    @property
    def is_accepted(self) -> bool:
        """Return True if order was approved without reduction."""
        return self.status == "ACCEPTED"

    @property
    def is_reduced(self) -> bool:
        """Return True if order quantity was scaled down due to risk rules."""
        return self.status == "REDUCED"

    @property
    def is_rejected(self) -> bool:
        """Return True if order was completely blocked."""
        return self.status == "REJECTED"

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to dictionary representation."""
        d = asdict(self)
        d["timestamp"] = str(self.timestamp)
        return d
