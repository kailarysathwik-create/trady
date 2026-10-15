"""Unit and integration tests for TRADE9 Virtual Money and Paper Exchange."""

from pathlib import Path

import pytest

from trady.portfolio.models import (
    Currency,
    OrderStatus,
)
from trady.portfolio.simulator import PaperExchangeSimulator
from trady.portfolio.wallet import WalletManager, get_instrument_currency


@pytest.fixture
def temp_wallet_mgr(tmp_path: Path) -> WalletManager:
    """Fixture providing an isolated WalletManager stored in a temp directory."""
    store_file = tmp_path / "test_portfolio.json"
    return WalletManager(
        storage_path=store_file,
        initial_inr=100_000.0,
        initial_usd=10_000.0,
    )


@pytest.fixture
def sim(temp_wallet_mgr: WalletManager) -> PaperExchangeSimulator:
    """Fixture providing PaperExchangeSimulator with deterministic frictions."""
    return PaperExchangeSimulator(
        wallet_manager=temp_wallet_mgr,
        commission_bps=10.0,  # 0.10%
        fixed_fee_inr=20.0,
        fixed_fee_usd=1.0,
        slippage_bps=5.0,  # 0.05%
    )


def test_currency_mapping_rules() -> None:
    """Verify instrument tickers map strictly to their home currency."""
    assert get_instrument_currency("RELIANCE.NS") == Currency.INR
    assert get_instrument_currency("TCS.BO") == Currency.INR
    assert get_instrument_currency("^NSEI") == Currency.INR
    assert get_instrument_currency("NIFTY50") == Currency.INR

    assert get_instrument_currency("SPY") == Currency.USD
    assert get_instrument_currency("AAPL") == Currency.USD
    assert get_instrument_currency("NVDA") == Currency.USD
    assert get_instrument_currency("^GSPC") == Currency.USD


def test_inr_and_usd_wallets_remain_strictly_separate(
    temp_wallet_mgr: WalletManager, sim: PaperExchangeSimulator
) -> None:
    """Verify INR and USD wallets do not mix or bleed balances into each other."""
    inr_w = temp_wallet_mgr.inr_wallet
    usd_w = temp_wallet_mgr.usd_wallet

    assert inr_w.currency == Currency.INR
    assert inr_w.available_cash == 100_000.0
    assert inr_w.total_equity == 100_000.0

    assert usd_w.currency == Currency.USD
    assert usd_w.available_cash == 10_000.0
    assert usd_w.total_equity == 10_000.0

    # Buying an Indian stock affects only the INR wallet
    ord_inr = sim.submit_order(
        symbol="INFY.NS",
        action="BUY",
        order_type="MARKET",
        quantity=10.0,
        current_market_price=1500.0,
    )
    assert ord_inr.status == OrderStatus.FILLED
    assert ord_inr.currency == Currency.INR

    # INR wallet balance reduced; USD wallet untouched
    assert inr_w.available_cash < 100_000.0
    assert "INFY.NS" in inr_w.positions
    assert usd_w.available_cash == 10_000.0
    assert len(usd_w.positions) == 0

    # Buying a US stock affects only the USD wallet
    ord_usd = sim.submit_order(
        symbol="SPY",
        action="BUY",
        order_type="MARKET",
        quantity=5.0,
        current_market_price=500.0,
    )
    assert ord_usd.status == OrderStatus.FILLED
    assert ord_usd.currency == Currency.USD

    assert usd_w.available_cash < 10_000.0
    assert "SPY" in usd_w.positions
    assert "SPY" not in inr_w.positions


def test_virtual_balances_persist_across_restarts(tmp_path: Path) -> None:
    """Verify wallet balances, positions, and order history survive restarts."""
    store_file = tmp_path / "persist_portfolio.json"

    # Session 1: Create and execute
    mgr1 = WalletManager(storage_path=store_file)
    sim1 = PaperExchangeSimulator(wallet_manager=mgr1)
    sim1.submit_order(
        symbol="AAPL",
        action="BUY",
        order_type="MARKET",
        quantity=10.0,
        current_market_price=200.0,
    )
    saved_usd_cash = mgr1.usd_wallet.available_cash
    assert saved_usd_cash < 10_000.0

    # Session 2: Reload from disk
    mgr2 = WalletManager(storage_path=store_file)
    assert mgr2.usd_wallet.available_cash == pytest.approx(saved_usd_cash)
    assert "AAPL" in mgr2.usd_wallet.positions
    assert mgr2.usd_wallet.positions["AAPL"].quantity == 10.0
    assert len(mgr2.usd_wallet.orders) == 1
    assert mgr2.usd_wallet.orders[0].symbol == "AAPL"


def test_insufficient_cash_prevents_purchase(
    sim: PaperExchangeSimulator, temp_wallet_mgr: WalletManager
) -> None:
    """Verify orders requiring more than available virtual cash are rejected."""
    # Attempt to buy $50,000 of SPY when USD wallet only has $10,000
    order = sim.submit_order(
        symbol="SPY",
        action="BUY",
        order_type="MARKET",
        quantity=100.0,
        current_market_price=500.0,
    )
    assert order.status == OrderStatus.REJECTED
    assert "Insufficient virtual cash" in (order.rejection_reason or "")
    # Cash must remain exactly unspent
    assert temp_wallet_mgr.usd_wallet.available_cash == 10_000.0
    assert len(temp_wallet_mgr.usd_wallet.positions) == 0


def test_insufficient_position_prevents_sale(
    sim: PaperExchangeSimulator, temp_wallet_mgr: WalletManager
) -> None:
    """Verify selling unowned or excess instruments is rejected."""
    order = sim.submit_order(
        symbol="MSFT",
        action="SELL",
        order_type="MARKET",
        quantity=5.0,
        current_market_price=400.0,
    )
    assert order.status == OrderStatus.REJECTED
    assert "Insufficient position quantity" in (order.rejection_reason or "")


def test_fees_and_slippage_calculation(sim: PaperExchangeSimulator) -> None:
    """Verify transparent fee and slippage math."""
    # 1. USD Order of $2,000
    # Slippage: 5 bps = 0.05% -> buy price increases by 0.05%
    # Commission: 10 bps = 0.10% ($2.00) + $1 fixed fee = $3.00
    est = sim.estimate_order(
        symbol="AAPL",
        action="BUY",
        quantity=10.0,
        price=200.0,
    )
    assert est["estimated_fill_price"] == pytest.approx(200.0 * 1.0005)
    assert est["estimated_fees"] == pytest.approx(3.0)
    assert est["total_capital_required"] == pytest.approx((10.0 * 200.0 * 1.0005) + 3.0)


def test_realized_and_unrealized_pnl_math(
    sim: PaperExchangeSimulator, temp_wallet_mgr: WalletManager
) -> None:
    """Verify realized and unrealized P&L accounting during round-trip trades."""
    # 1. Buy 10 units at 100.0
    order_buy = sim.submit_order(
        symbol="XYZ",
        action="BUY",
        order_type="MARKET",
        quantity=10.0,
        current_market_price=100.0,
    )
    assert order_buy.status == OrderStatus.FILLED
    pos = temp_wallet_mgr.usd_wallet.positions["XYZ"]
    assert pos.quantity == 10.0

    # 2. Price rises to 120.0
    temp_wallet_mgr.update_market_prices({"XYZ": 120.0})
    assert pos.current_price == 120.0
    assert pos.unrealized_pnl > 0.0

    # 3. Sell 5 units at 120.0
    order_sell = sim.submit_order(
        symbol="XYZ",
        action="SELL",
        order_type="MARKET",
        quantity=5.0,
        current_market_price=120.0,
    )
    assert order_sell.status == OrderStatus.FILLED
    assert temp_wallet_mgr.usd_wallet.realized_pnl > 0.0
    assert pos.quantity == 5.0


def test_limit_order_cash_reservation_and_cancellation(
    sim: PaperExchangeSimulator, temp_wallet_mgr: WalletManager
) -> None:
    """Verify limit orders reserve cash and cancellation refunds it completely."""
    usd_w = temp_wallet_mgr.usd_wallet
    initial_cash = usd_w.available_cash

    # Place Buy Limit Order at $150 (market is $160)
    order = sim.submit_order(
        symbol="NVDA",
        action="BUY",
        order_type="LIMIT",
        quantity=10.0,
        current_market_price=160.0,
        limit_price=150.0,
    )
    assert order.status == OrderStatus.PENDING
    assert order.reserved_cash > 0.0
    assert usd_w.reserved_cash == order.reserved_cash
    assert usd_w.available_cash == pytest.approx(initial_cash - order.reserved_cash)

    # Cancel order
    cancelled = sim.cancel_order(order.order_id)
    assert cancelled.status == OrderStatus.CANCELLED
    assert usd_w.reserved_cash == 0.0
    assert usd_w.available_cash == pytest.approx(initial_cash)


def test_limit_order_execution_when_price_hits(
    sim: PaperExchangeSimulator, temp_wallet_mgr: WalletManager
) -> None:
    """Verify pending limit orders execute when market reaches or beats limit price."""
    usd_w = temp_wallet_mgr.usd_wallet

    order = sim.submit_order(
        symbol="TSLA",
        action="BUY",
        order_type="LIMIT",
        quantity=5.0,
        current_market_price=220.0,
        limit_price=200.0,
    )
    assert order.status == OrderStatus.PENDING

    # Price stays above limit -> no fill
    filled_none = sim.process_pending_limit_orders({"TSLA": 210.0})
    assert len(filled_none) == 0
    assert order.status == OrderStatus.PENDING

    # Price drops to 198.0 -> hits limit -> fills!
    filled_orders = sim.process_pending_limit_orders({"TSLA": 198.0})
    assert len(filled_orders) == 1
    assert filled_orders[0].order_id == order.order_id
    assert order.status == OrderStatus.FILLED
    assert order.filled_price == 200.0
    assert "TSLA" in usd_w.positions


def test_cross_currency_consolidated_summary(temp_wallet_mgr: WalletManager) -> None:
    """Verify explicit exchange-rate reporting and safe fallback when rate
    is missing.
    """
    # 1. Missing rate
    summary_no_rate = temp_wallet_mgr.get_consolidated_summary()
    assert summary_no_rate["consolidated_valuation"]["status"] == "UNAVAILABLE"

    # 2. Valid rate (1 USD = 86.50 INR)
    summary_with_rate = temp_wallet_mgr.get_consolidated_summary(
        usd_inr_rate=86.50,
        rate_timestamp="2026-10-15T10:00:00Z",
    )
    consol = summary_with_rate["consolidated_valuation"]
    assert consol["status"] == "CALCULATED"
    assert consol["usd_inr_conversion_rate"] == 86.50
    assert consol["total_combined_equity_usd"] > 10_000.0
    assert consol["total_combined_equity_inr"] > 100_000.0


def test_explicit_portfolio_reset_requires_confirmation(
    temp_wallet_mgr: WalletManager, sim: PaperExchangeSimulator
) -> None:
    """Safety check: reset requires confirmation=True."""
    sim.submit_order(
        symbol="SPY",
        action="BUY",
        order_type="MARKET",
        quantity=5.0,
        current_market_price=500.0,
    )
    assert len(temp_wallet_mgr.usd_wallet.positions) == 1

    # Attempt reset without confirmation -> raises ValueError
    with pytest.raises(ValueError, match="explicit confirmation=True required"):
        temp_wallet_mgr.reset_portfolio(confirmation=False)

    # Valid reset with confirmation
    res = temp_wallet_mgr.reset_portfolio(confirmation=True)
    assert res["status"] == "success"
    assert len(temp_wallet_mgr.usd_wallet.positions) == 0
    assert temp_wallet_mgr.usd_wallet.available_cash == 10_000.0


def test_immutable_forecast_metadata_preserved_on_order(
    sim: PaperExchangeSimulator,
) -> None:
    """Verify original forecast and model version are permanently attached to order."""
    forecast = {
        "model_version": "logistic_regression_v1.0",
        "horizon_days": 5,
        "probability_up": 0.685,
        "forecast_timestamp": "2026-10-15T09:30:00Z",
        "risk_rating": "MODERATE",
    }
    order = sim.submit_order(
        symbol="AAPL",
        action="BUY",
        order_type="MARKET",
        quantity=5.0,
        current_market_price=200.0,
        forecast_metadata=forecast,
    )
    assert order.status == OrderStatus.FILLED
    assert order.forecast_metadata is not None
    assert order.forecast_metadata["model_version"] == "logistic_regression_v1.0"
    assert order.forecast_metadata["probability_up"] == 0.685
    assert order.forecast_metadata["risk_rating"] == "MODERATE"
