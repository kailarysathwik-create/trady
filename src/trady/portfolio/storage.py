"""Atomic JSON storage and persistence for TRADE9 Virtual Wallets."""

import json
import os
from pathlib import Path
from typing import Any

from trady.portfolio.models import Currency, VirtualWallet
from trady.utils.logging import get_logger

logger = get_logger("portfolio.storage")

DEFAULT_STORAGE_PATH = Path("data/paper/portfolio.json")
DEFAULT_INR_INITIAL = 100_000.0  # ₹100,000 fictional starting balance
DEFAULT_USD_INITIAL = 10_000.0  # $10,000 fictional starting balance


class PortfolioStorage:
    """Manages persistent serialization of virtual currency portfolios."""

    def __init__(self, file_path: Path | str | None = None) -> None:
        self.file_path = Path(file_path or DEFAULT_STORAGE_PATH).resolve()

    def load(
        self,
        default_inr_balance: float = DEFAULT_INR_INITIAL,
        default_usd_balance: float = DEFAULT_USD_INITIAL,
    ) -> tuple[VirtualWallet, VirtualWallet]:
        """Load INR and USD virtual wallets from disk, or initialize defaults."""
        if not self.file_path.exists():
            logger.info(
                "No existing portfolio found at %s. Creating default wallets.",
                self.file_path,
            )
            inr_wallet = VirtualWallet(
                currency=Currency.INR,
                initial_balance=default_inr_balance,
                available_cash=default_inr_balance,
            )
            usd_wallet = VirtualWallet(
                currency=Currency.USD,
                initial_balance=default_usd_balance,
                available_cash=default_usd_balance,
            )
            self.save(inr_wallet, usd_wallet)
            return inr_wallet, usd_wallet

        try:
            with open(self.file_path, encoding="utf-8") as f:
                data = json.load(f)

            inr_data = data.get("INR")
            usd_data = data.get("USD")

            inr_wallet = (
                VirtualWallet.from_dict(inr_data)
                if inr_data
                else VirtualWallet(
                    currency=Currency.INR,
                    initial_balance=default_inr_balance,
                    available_cash=default_inr_balance,
                )
            )

            usd_wallet = (
                VirtualWallet.from_dict(usd_data)
                if usd_data
                else VirtualWallet(
                    currency=Currency.USD,
                    initial_balance=default_usd_balance,
                    available_cash=default_usd_balance,
                )
            )

            logger.info("Successfully loaded virtual wallets from %s", self.file_path)
            return inr_wallet, usd_wallet
        except Exception as exc:
            logger.error(
                "Error reading portfolio from %s: %s. Initializing fresh wallets.",
                self.file_path,
                exc,
            )
            inr_wallet = VirtualWallet(
                currency=Currency.INR,
                initial_balance=default_inr_balance,
                available_cash=default_inr_balance,
            )
            usd_wallet = VirtualWallet(
                currency=Currency.USD,
                initial_balance=default_usd_balance,
                available_cash=default_usd_balance,
            )
            return inr_wallet, usd_wallet

    def save(self, inr_wallet: VirtualWallet, usd_wallet: VirtualWallet) -> None:
        """Atomically persist INR and USD wallets to disk using a temporary file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(".tmp")

        payload: dict[str, Any] = {
            "version": "1.0",
            "updated_at": Path(self.file_path).name,
            "disclaimer": (
                "VIRTUAL PAPER TRADING ONLY. These are simulated fictional "
                "balances with no real monetary value. No broker connection."
            ),
            "INR": inr_wallet.to_dict(),
            "USD": usd_wallet.to_dict(),
        }

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Atomic replacement
        os.replace(tmp_path, self.file_path)
        logger.debug("Atomically persisted virtual wallets to %s", self.file_path)

    def reset(
        self,
        currency: Currency | None = None,
        inr_balance: float = DEFAULT_INR_INITIAL,
        usd_balance: float = DEFAULT_USD_INITIAL,
    ) -> tuple[VirtualWallet, VirtualWallet]:
        """Reset specified wallet or all wallets to initial starting balances."""
        inr_w, usd_w = self.load(
            default_inr_balance=inr_balance,
            default_usd_balance=usd_balance,
        )

        if currency is None or currency == Currency.INR:
            inr_w = VirtualWallet(
                currency=Currency.INR,
                initial_balance=inr_balance,
                available_cash=inr_balance,
            )
        if currency is None or currency == Currency.USD:
            usd_w = VirtualWallet(
                currency=Currency.USD,
                initial_balance=usd_balance,
                available_cash=usd_balance,
            )

        self.save(inr_w, usd_w)
        logger.info("Reset virtual portfolio balances (currency=%s)", currency)
        return inr_w, usd_w
