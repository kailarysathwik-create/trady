"""Paper Order Execution Simulator with realistic frictions and cash controls."""

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from trady.portfolio.models import (
    Currency,
    OrderAction,
    OrderStatus,
    OrderType,
    PaperOrder,
    Position,
)
from trady.portfolio.wallet import WalletManager, get_instrument_currency
from trady.utils.logging import get_logger

logger = get_logger("portfolio.simulator")


class PaperExchangeSimulator:
    """Simulates realistic paper order routing, friction modeling, and execution."""

    def __init__(
        self,
        wallet_manager: WalletManager,
        commission_bps: float = 10.0,  # 0.10% commission
        fixed_fee_inr: float = 20.0,  # ₹20 flat fee (standard discount broker)
        fixed_fee_usd: float = 1.0,  # $1.00 flat fee
        slippage_bps: float = 5.0,  # 0.05% typical slippage on liquid names
    ) -> None:
        self.wallet_manager = wallet_manager
        self.commission_bps = commission_bps
        self.fixed_fee_inr = fixed_fee_inr
        self.fixed_fee_usd = fixed_fee_usd
        self.slippage_bps = slippage_bps

    def calculate_fees(self, notional: float, currency: Currency) -> float:
        """Calculate simulated commission and fixed exchange transaction fee."""
        pct_fee = notional * (self.commission_bps / 10_000.0)
        fixed = self.fixed_fee_inr if currency == Currency.INR else self.fixed_fee_usd
        return round(pct_fee + fixed, 2)

    def calculate_slippage_price(self, price: float, action: OrderAction) -> float:
        """Apply directional bid-ask slippage penalty."""
        slip_mult = self.slippage_bps / 10_000.0
        if action == OrderAction.BUY:
            return round(price * (1.0 + slip_mult), 4)
        return round(price * (1.0 - slip_mult), 4)

    def estimate_order(
        self,
        symbol: str,
        action: OrderAction | str,
        quantity: float,
        price: float,
    ) -> dict[str, Any]:
        """Provide detailed pre-trade cost and fee breakdown."""
        act = OrderAction(action) if isinstance(action, str) else action
        if quantity <= 0:
            raise ValueError("Order quantity must be positive.")
        if price <= 0:
            raise ValueError("Reference market price must be positive.")

        currency = get_instrument_currency(symbol)
        wallet = self.wallet_manager.get_wallet(currency)

        est_fill_price = self.calculate_slippage_price(price, act)
        notional = quantity * est_fill_price
        fees = self.calculate_fees(notional, currency)

        if act == OrderAction.BUY:
            total_required = notional + fees
            has_sufficient = wallet.available_cash >= total_required
        else:
            total_required = 0.0  # Selling generates cash
            curr_pos = wallet.positions.get(symbol)
            has_sufficient = (curr_pos is not None) and (curr_pos.quantity >= quantity)

        return {
            "symbol": symbol,
            "currency": currency.value,
            "action": act.value,
            "quantity": quantity,
            "reference_price": price,
            "estimated_fill_price": est_fill_price,
            "estimated_notional": notional,
            "estimated_fees": fees,
            "estimated_slippage": abs(est_fill_price - price) * quantity,
            "total_capital_required": total_required if act == OrderAction.BUY else 0.0,
            "available_virtual_cash": wallet.available_cash,
            "sufficient_resources": has_sufficient,
        }

    def submit_order(
        self,
        symbol: str,
        action: OrderAction | str,
        order_type: OrderType | str,
        quantity: float,
        current_market_price: float,
        limit_price: float | None = None,
        forecast_metadata: dict[str, Any] | None = None,
    ) -> PaperOrder:
        """Submit and process a simulated paper order.

        Safety:
        - NEVER contacts real brokers.
        - Enforces cash and position constraints.
        - Preserves immutable AI forecast metadata.
        """
        act = OrderAction(action) if isinstance(action, str) else action
        ot = OrderType(order_type) if isinstance(order_type, str) else order_type

        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if current_market_price <= 0:
            raise ValueError(
                f"Market price must be positive, got {current_market_price}"
            )

        currency = get_instrument_currency(symbol)
        wallet = self.wallet_manager.get_wallet(currency)

        order_id = f"ord_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

        # Market Order Execution
        if ot == OrderType.MARKET:
            fill_price = self.calculate_slippage_price(current_market_price, act)
            notional = quantity * fill_price
            fees = self.calculate_fees(notional, currency)
            slip = abs(fill_price - current_market_price) * quantity

            if act == OrderAction.BUY:
                total_cost = notional + fees
                if total_cost > wallet.available_cash:
                    order = PaperOrder(
                        order_id=order_id,
                        symbol=symbol,
                        currency=currency,
                        action=act,
                        order_type=ot,
                        quantity=quantity,
                        status=OrderStatus.REJECTED,
                        rejection_reason=(
                            f"Insufficient virtual cash. Required: "
                            f"{currency.symbol}{total_cost:,.2f}, Available: "
                            f"{currency.symbol}{wallet.available_cash:,.2f}"
                        ),
                        forecast_metadata=forecast_metadata,
                    )
                    wallet.orders.append(order)
                    self.wallet_manager.persist()
                    return order

                # Execute BUY
                wallet.available_cash -= total_cost
                wallet.total_fees += fees

                if symbol in wallet.positions:
                    pos = wallet.positions[symbol]
                    new_qty = pos.quantity + quantity
                    new_avg = (
                        (pos.quantity * pos.average_entry_price)
                        + (quantity * fill_price)
                    ) / new_qty
                    pos.quantity = new_qty
                    pos.average_entry_price = new_avg
                    pos.current_price = current_market_price
                    pos.updated_at = datetime.now(UTC).isoformat()
                else:
                    wallet.positions[symbol] = Position(
                        symbol=symbol,
                        currency=currency,
                        quantity=quantity,
                        average_entry_price=fill_price,
                        current_price=current_market_price,
                    )

            else:  # SELL
                curr_pos = wallet.positions.get(symbol)
                if not curr_pos or curr_pos.quantity < quantity:
                    held = curr_pos.quantity if curr_pos else 0.0
                    order = PaperOrder(
                        order_id=order_id,
                        symbol=symbol,
                        currency=currency,
                        action=act,
                        order_type=ot,
                        quantity=quantity,
                        status=OrderStatus.REJECTED,
                        rejection_reason=(
                            f"Insufficient position quantity. Held: {held}, "
                            f"Attempted: {quantity}"
                        ),
                        forecast_metadata=forecast_metadata,
                    )
                    wallet.orders.append(order)
                    self.wallet_manager.persist()
                    return order

                # Execute SELL
                net_proceeds = notional - fees
                wallet.available_cash += net_proceeds
                wallet.total_fees += fees

                trade_pnl = (
                    (quantity * fill_price)
                    - (quantity * curr_pos.average_entry_price)
                    - fees
                )
                wallet.realized_pnl += trade_pnl
                curr_pos.realized_pnl += trade_pnl

                if curr_pos.quantity == quantity:
                    del wallet.positions[symbol]
                else:
                    curr_pos.quantity -= quantity
                    curr_pos.current_price = current_market_price
                    curr_pos.updated_at = datetime.now(UTC).isoformat()

            # Record filled order
            now_str = datetime.now(UTC).isoformat()
            order = PaperOrder(
                order_id=order_id,
                symbol=symbol,
                currency=currency,
                action=act,
                order_type=ot,
                quantity=quantity,
                status=OrderStatus.FILLED,
                created_at=now_str,
                filled_at=now_str,
                filled_price=fill_price,
                filled_quantity=quantity,
                fees=fees,
                slippage=slip,
                forecast_metadata=forecast_metadata,
            )
            wallet.orders.append(order)
            wallet.record_equity_snapshot()
            self.wallet_manager.persist()
            return order

        # Limit Order Execution
        if ot == OrderType.LIMIT:
            if limit_price is None or limit_price <= 0:
                raise ValueError("Valid limit_price required for LIMIT orders.")

            est_notional = quantity * limit_price
            est_fees = self.calculate_fees(est_notional, currency)
            total_reserve = est_notional + est_fees

            if act == OrderAction.BUY:
                if total_reserve > wallet.available_cash:
                    order = PaperOrder(
                        order_id=order_id,
                        symbol=symbol,
                        currency=currency,
                        action=act,
                        order_type=ot,
                        quantity=quantity,
                        limit_price=limit_price,
                        status=OrderStatus.REJECTED,
                        rejection_reason=(
                            f"Insufficient virtual cash to reserve for limit order. "
                            f"Required: {currency.symbol}{total_reserve:,.2f}, "
                            f"Available: {currency.symbol}{wallet.available_cash:,.2f}"
                        ),
                        forecast_metadata=forecast_metadata,
                    )
                    wallet.orders.append(order)
                    self.wallet_manager.persist()
                    return order

                # Lock cash in reserved_cash
                wallet.available_cash -= total_reserve
                wallet.reserved_cash += total_reserve
            else:  # SELL limit order
                curr_pos = wallet.positions.get(symbol)
                if not curr_pos or curr_pos.quantity < quantity:
                    held = curr_pos.quantity if curr_pos else 0.0
                    order = PaperOrder(
                        order_id=order_id,
                        symbol=symbol,
                        currency=currency,
                        action=act,
                        order_type=ot,
                        quantity=quantity,
                        limit_price=limit_price,
                        status=OrderStatus.REJECTED,
                        rejection_reason=(
                            f"Insufficient position quantity. Held: {held}, "
                            f"Attempted: {quantity}"
                        ),
                        forecast_metadata=forecast_metadata,
                    )
                    wallet.orders.append(order)
                    self.wallet_manager.persist()
                    return order

            order = PaperOrder(
                order_id=order_id,
                symbol=symbol,
                currency=currency,
                action=act,
                order_type=ot,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                reserved_cash=total_reserve if act == OrderAction.BUY else 0.0,
                forecast_metadata=forecast_metadata,
            )
            wallet.orders.append(order)
            self.wallet_manager.persist()
            return order

        raise ValueError(f"Unsupported order type: {ot}")

    def cancel_order(self, order_id: str) -> PaperOrder:
        """Cancel a pending paper limit order and release any reserved virtual cash."""
        for wallet in (self.wallet_manager.inr_wallet, self.wallet_manager.usd_wallet):
            for order in wallet.orders:
                if order.order_id == order_id:
                    if order.status != OrderStatus.PENDING:
                        raise ValueError(
                            f"Cannot cancel order in status '{order.status.value}'"
                        )

                    order.status = OrderStatus.CANCELLED
                    if order.reserved_cash > 0:
                        wallet.reserved_cash -= order.reserved_cash
                        wallet.available_cash += order.reserved_cash
                        order.reserved_cash = 0.0

                    self.wallet_manager.persist()
                    logger.info(
                        "Cancelled paper order %s, released reserved cash", order_id
                    )
                    return order

        raise KeyError(f"Paper order '{order_id}' not found.")

    def process_pending_limit_orders(
        self, price_map: dict[str, float]
    ) -> list[PaperOrder]:
        """Check all pending limit orders against latest prices and
        fill executable ones.
        """
        filled_orders: list[PaperOrder] = []

        for wallet in (self.wallet_manager.inr_wallet, self.wallet_manager.usd_wallet):
            for order in wallet.orders:
                if order.status != OrderStatus.PENDING or order.limit_price is None:
                    continue

                symbol = order.symbol
                if symbol not in price_map or price_map[symbol] <= 0:
                    continue

                mkt_price = float(price_map[symbol])
                should_fill = False

                # Buy Limit fills when market drops to or below limit
                if order.action == OrderAction.BUY and mkt_price <= order.limit_price:
                    should_fill = True
                # Sell Limit fills when market rises to or above limit
                elif (
                    order.action == OrderAction.SELL and mkt_price >= order.limit_price
                ):
                    should_fill = True

                if should_fill:
                    # Fill at the limit price (or better)
                    fill_price = order.limit_price
                    notional = order.quantity * fill_price
                    fees = self.calculate_fees(notional, order.currency)

                    if order.action == OrderAction.BUY:
                        # Release reserved cash and deduct actual cost
                        wallet.reserved_cash -= order.reserved_cash
                        actual_cost = notional + fees
                        diff = order.reserved_cash - actual_cost
                        wallet.available_cash += (
                            diff  # refund surplus if limit filled cheaper
                        )
                        wallet.total_fees += fees

                        if symbol in wallet.positions:
                            pos = wallet.positions[symbol]
                            new_qty = pos.quantity + order.quantity
                            new_avg = (
                                (pos.quantity * pos.average_entry_price)
                                + (order.quantity * fill_price)
                            ) / new_qty
                            pos.quantity = new_qty
                            pos.average_entry_price = new_avg
                            pos.current_price = mkt_price
                        else:
                            wallet.positions[symbol] = Position(
                                symbol=symbol,
                                currency=order.currency,
                                quantity=order.quantity,
                                average_entry_price=fill_price,
                                current_price=mkt_price,
                            )
                    else:  # Sell Limit
                        net_proceeds = notional - fees
                        wallet.available_cash += net_proceeds
                        wallet.total_fees += fees

                        curr_pos = wallet.positions.get(symbol)
                        if curr_pos:
                            trade_pnl = (
                                (order.quantity * fill_price)
                                - (order.quantity * curr_pos.average_entry_price)
                                - fees
                            )
                            wallet.realized_pnl += trade_pnl
                            curr_pos.realized_pnl += trade_pnl

                            if curr_pos.quantity == order.quantity:
                                del wallet.positions[symbol]
                            else:
                                curr_pos.quantity -= order.quantity
                                curr_pos.current_price = mkt_price

                    order.status = OrderStatus.FILLED
                    order.filled_at = datetime.now(UTC).isoformat()
                    order.filled_price = fill_price
                    order.filled_quantity = order.quantity
                    order.fees = fees
                    order.reserved_cash = 0.0

                    wallet.record_equity_snapshot()
                    filled_orders.append(order)

        if filled_orders:
            self.wallet_manager.persist()

        return filled_orders
