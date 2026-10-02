"""TRADY Risk Engine enforcing portfolio limits and audit trail decisions."""

import uuid
from typing import Any

from trady.backtesting.strategy import Order
from trady.risk.config import RiskConfig
from trady.risk.decision import RiskDecision
from trady.risk.sizing import PositionSizer


class RiskEngine:
    """Independent risk controller sitting between strategies and simulated execution.

    Enforces:
    1. Maximum simulated drawdown circuit breakers.
    2. Maximum single-position dollar exposure.
    3. Maximum single-position equity concentration.
    4. Maximum aggregate portfolio gross leverage.
    5. Liquid cash buffers for transaction costs.
    6. Volatility-based and equity-fraction position sizing.

    Produces transparent decision records for every evaluated order.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.sizer = PositionSizer(self.config)
        self.decisions: list[RiskDecision] = []

    def size_order(
        self,
        price: float,
        equity: float,
        cash: float,
        atr: float | None = None,
        volatility: float | None = None,
    ) -> float:
        """Calculate target share quantity using configured sizing rules."""
        return self.sizer.calculate_size(
            price=price,
            equity=equity,
            cash=cash,
            atr=atr,
            volatility=volatility,
        )

    def evaluate_order(
        self,
        order: Order,
        timestamp: Any,
        current_price: float,
        current_equity: float,
        current_cash: float,
        current_drawdown: float,
        existing_positions: dict[str, Any],
        market_context: dict[str, Any] | None = None,
    ) -> tuple[Order | None, RiskDecision]:
        """Evaluate a proposed simulated order against all risk constraints.

        Args:
            order: Proposed virtual trading order.
            timestamp: Timestamp of the evaluation bar.
            current_price: Mark price of the instrument.
            current_equity: Mark-to-market portfolio equity.
            current_cash: Available liquid cash balance.
            current_drawdown: Current drawdown from peak (positive magnitude).
            existing_positions: Dictionary of active Position objects.
            market_context: Optional context containing volatility/ATR metrics.

        Returns:
            Tuple of (approved_order or None, RiskDecision audit record).
        """
        decision_id = f"risk_{uuid.uuid4().hex[:8]}"
        symbol = order.symbol
        action = order.action
        requested_qty = order.quantity
        requested_val = requested_qty * current_price

        # Current snapshot metrics
        cur_pos_qty = 0.0
        if symbol in existing_positions:
            pos = existing_positions[symbol]
            cur_pos_qty = getattr(pos, "quantity", 0.0)

        cur_pos_val = cur_pos_qty * current_price
        cur_gross_val = sum(
            getattr(p, "quantity", 0.0) * current_price
            for p in existing_positions.values()
        )
        cur_portfolio_exp_pct = (
            (cur_gross_val / current_equity) if current_equity > 0 else 0.0
        )
        cur_concentration_pct = (
            (cur_pos_val / current_equity) if current_equity > 0 else 0.0
        )

        def make_decision(
            status: str,
            approved_qty: float,
            reason: str,
            constraint: str | None = None,
        ) -> RiskDecision:
            approved_val = approved_qty * current_price
            dec = RiskDecision(
                decision_id=decision_id,
                timestamp=timestamp,
                symbol=symbol,
                action=action,
                status=status,
                requested_quantity=float(requested_qty),
                approved_quantity=float(approved_qty),
                requested_value=float(requested_val),
                approved_value=float(approved_val),
                price=float(current_price),
                reason=reason,
                constraint_triggered=constraint,
                current_equity=float(current_equity),
                current_drawdown=float(current_drawdown),
                portfolio_exposure_pct=float(cur_portfolio_exp_pct),
                position_concentration_pct=float(cur_concentration_pct),
            )
            self.decisions.append(dec)
            return dec

        # 1. Sanity check
        if requested_qty <= 0 or current_price <= 0:
            dec = make_decision(
                status="REJECTED",
                approved_qty=0.0,
                reason="Rejected: order quantity and price must be positive.",
                constraint="order_sanity",
            )
            return None, dec

        # 2. Position Exits / Reductions (SELL)
        if action == "SELL":
            if cur_pos_qty <= 0:
                if not self.config.allow_shorting:
                    dec = make_decision(
                        status="REJECTED",
                        approved_qty=0.0,
                        reason="Rejected: shorting disabled and no long position.",
                        constraint="short_selling_disabled",
                    )
                    return None, dec
            elif requested_qty > cur_pos_qty and not self.config.allow_shorting:
                approved_qty = cur_pos_qty
                dec = make_decision(
                    status="REDUCED",
                    approved_qty=approved_qty,
                    reason=(
                        f"Reduced: sell ({requested_qty:.4f}) exceeds held position "
                        f"({cur_pos_qty:.4f}). Clamped to full exit."
                    ),
                    constraint="held_position_limit",
                )
                approved_order = Order(
                    symbol=symbol,
                    action="SELL",
                    quantity=approved_qty,
                    order_type=order.order_type,
                    exit_reason=order.exit_reason,
                )
                return approved_order, dec

            # Normal closing order conforms to risk rules
            dec = make_decision(
                status="ACCEPTED",
                approved_qty=requested_qty,
                reason="Accepted: risk-reducing order complies with portfolio rules.",
            )
            return order, dec

        # 3. Position Entries / Additions (BUY)
        # Constraint A: Maximum Simulated Drawdown Circuit Breaker
        if self.config.max_drawdown_limit is not None:
            dd_mag = abs(current_drawdown)
            if dd_mag >= self.config.max_drawdown_limit:
                limit_pct = self.config.max_drawdown_limit
                dec = make_decision(
                    status="REJECTED",
                    approved_qty=0.0,
                    reason=(
                        f"Rejected: drawdown ({dd_mag:.2%}) breached limit "
                        f"({limit_pct:.2%}). Circuit breaker active."
                    ),
                    constraint="max_drawdown_limit",
                )
                return None, dec

        approved_qty = requested_qty
        reduction_reasons: list[str] = []
        triggered_constraint: str | None = None

        # Constraint B: Maximum Position Exposure (Dollar Cap)
        if self.config.max_position_exposure is not None:
            proj_pos_val = cur_pos_val + (approved_qty * current_price)
            if proj_pos_val > self.config.max_position_exposure:
                rem_exp = max(0.0, self.config.max_position_exposure - cur_pos_val)
                max_exp_qty = rem_exp / current_price
                if max_exp_qty <= 1e-7:
                    limit_val = self.config.max_position_exposure
                    dec = make_decision(
                        status="REJECTED",
                        approved_qty=0.0,
                        reason=(
                            f"Rejected: position (${cur_pos_val:,.2f}) already at "
                            f"or exceeds limit (${limit_val:,.2f})."
                        ),
                        constraint="max_position_exposure",
                    )
                    return None, dec
                approved_qty = min(approved_qty, max_exp_qty)
                exp_lim = self.config.max_position_exposure
                reduction_reasons.append(
                    f"capped by max pos exposure (${exp_lim:,.0f})"
                )
                triggered_constraint = "max_position_exposure"

        # Constraint C: Maximum Position Concentration (% of Equity)
        max_pos_val = current_equity * self.config.max_position_concentration
        proj_pos_val = cur_pos_val + (approved_qty * current_price)
        if proj_pos_val > max_pos_val:
            rem_conc_val = max(0.0, max_pos_val - cur_pos_val)
            max_conc_qty = rem_conc_val / current_price
            if max_conc_qty <= 1e-7:
                conc_lim = self.config.max_position_concentration
                dec = make_decision(
                    status="REJECTED",
                    approved_qty=0.0,
                    reason=(
                        f"Rejected: concentration ({cur_concentration_pct:.1%}) "
                        f"at max limit ({conc_lim:.1%})."
                    ),
                    constraint="max_position_concentration",
                )
                return None, dec
            approved_qty = min(approved_qty, max_conc_qty)
            conc_pct = self.config.max_position_concentration
            reduction_reasons.append(f"capped by max concentration ({conc_pct:.1%})")
            triggered_constraint = "max_position_concentration"

        # Constraint D: Maximum Portfolio Gross Exposure (% of Equity)
        max_gross_val = current_equity * self.config.max_portfolio_exposure
        proj_gross_val = cur_gross_val + (approved_qty * current_price)
        if proj_gross_val > max_gross_val:
            rem_port_val = max(0.0, max_gross_val - cur_gross_val)
            max_port_qty = rem_port_val / current_price
            if max_port_qty <= 1e-7:
                port_lim = self.config.max_portfolio_exposure
                dec = make_decision(
                    status="REJECTED",
                    approved_qty=0.0,
                    reason=(
                        f"Rejected: leverage ({cur_portfolio_exp_pct:.1%}) "
                        f"at max limit ({port_lim:.1%})."
                    ),
                    constraint="max_portfolio_exposure",
                )
                return None, dec
            approved_qty = min(approved_qty, max_port_qty)
            port_pct = self.config.max_portfolio_exposure
            reduction_reasons.append(f"capped by max exposure ({port_pct:.1%})")
            triggered_constraint = "max_portfolio_exposure"

        # Constraint E: Available Cash & Buffer
        available_cash = current_cash * (1.0 - self.config.cash_buffer_pct)
        order_cost = approved_qty * current_price
        if order_cost > available_cash:
            max_cash_qty = max(0.0, available_cash / current_price)
            if max_cash_qty <= 1e-7:
                dec = make_decision(
                    status="REJECTED",
                    approved_qty=0.0,
                    reason="Rejected: insufficient liquid cash to satisfy buffer.",
                    constraint="cash_buffer",
                )
                return None, dec
            approved_qty = min(approved_qty, max_cash_qty)
            reduction_reasons.append("capped by available cash buffer")
            triggered_constraint = "cash_buffer"

        # 4. Final outcome
        if approved_qty < (requested_qty - 1e-7):
            reasons_text = ", ".join(reduction_reasons)
            reason_str = (
                f"Reduced: order size scaled from {requested_qty:.4f} "
                f"to {approved_qty:.4f} ({reasons_text})."
            )
            dec = make_decision(
                status="REDUCED",
                approved_qty=approved_qty,
                reason=reason_str,
                constraint=triggered_constraint,
            )
            approved_order = Order(
                symbol=symbol,
                action="BUY",
                quantity=approved_qty,
                order_type=order.order_type,
                exit_reason=order.exit_reason,
            )
            return approved_order, dec

        dec = make_decision(
            status="ACCEPTED",
            approved_qty=approved_qty,
            reason="Accepted: order conforms to all portfolio and position risk rules.",
        )
        return order, dec
