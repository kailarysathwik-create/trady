"""Unit tests strictly enforcing safety firewalls and real-money trading prohibition."""

import os
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from trady.backtesting import SimulatedOrder
from trady.config.loader import load_config
from trady.config.settings import SafetyConfig


def test_safety_firewall_prohibits_live_trading() -> None:
    """Verifies that live_trading_enabled cannot be set to True."""
    with pytest.raises(ValueError, match="SAFETY VIOLATION"):
        SafetyConfig(live_trading_enabled=True)


def test_safety_firewall_prohibits_non_paper_mode() -> None:
    """Verifies that paper_trading_only cannot be set to False."""
    with pytest.raises(ValueError, match="SAFETY VIOLATION"):
        SafetyConfig(paper_trading_only=False)


def test_env_override_live_trading_triggers_fatal_error() -> None:
    """Verifies TRADY_LIVE_TRADING_ENABLED=true causes configuration failure."""
    with patch.dict(os.environ, {"TRADY_LIVE_TRADING_ENABLED": "true"}):
        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            load_config()


def test_simulated_order_enforces_simulation_invariant() -> None:
    """Verifies SimulatedOrder rejects non-simulated orders."""
    with pytest.raises(ValueError, match="SAFETY VIOLATION"):
        SimulatedOrder(
            order_id="ORD-001",
            symbol="TEST",
            action="BUY",
            quantity=10.0,
            timestamp=datetime.now(UTC),
            simulated=False,
        )


def test_simulated_order_enforces_positive_quantity() -> None:
    """Verifies SimulatedOrder requires positive quantity."""
    with pytest.raises(ValueError, match="positive"):
        SimulatedOrder(
            order_id="ORD-002",
            symbol="TEST",
            action="BUY",
            quantity=-5.0,
            timestamp=datetime.now(UTC),
            simulated=True,
        )
