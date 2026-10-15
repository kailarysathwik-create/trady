"""Wallet Manager for isolated dual-currency virtual portfolios."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trady.portfolio.models import Currency, VirtualWallet
from trady.portfolio.storage import (
    DEFAULT_INR_INITIAL,
    DEFAULT_STORAGE_PATH,
    DEFAULT_USD_INITIAL,
    PortfolioStorage,
)
from trady.utils.logging import get_logger

logger = get_logger("portfolio.wallet")


def get_instrument_currency(symbol: str) -> Currency:
    """Determine trading currency strictly from instrument ticker rules.

    Rules:
    - Instruments with suffix '.NS' (NSE) or '.BO' (BSE) -> INR
    - Indices starting with '^NSE' or '^BSE' -> INR
    - US tickers, ETFs, and major US indices (e.g. SPY, AAPL, QQQ, ^GSPC) -> USD
    """
    clean = symbol.strip().upper()
    if (
        clean.endswith(".NS")
        or clean.endswith(".BO")
        or clean.startswith("^NSE")
        or clean.startswith("^BSE")
        or clean in ("NIFTY50", "BANKNIFTY", "SENSEX")
    ):
        return Currency.INR
    return Currency.USD


class WalletManager:
    """Coordinates isolated INR and USD virtual portfolios with atomic persistence."""

    def __init__(
        self,
        storage_path: Path | str | None = None,
        storage: PortfolioStorage | None = None,
        initial_inr: float = DEFAULT_INR_INITIAL,
        initial_usd: float = DEFAULT_USD_INITIAL,
    ) -> None:
        self.storage = storage or PortfolioStorage(storage_path or DEFAULT_STORAGE_PATH)
        self.inr_wallet, self.usd_wallet = self.storage.load(
            default_inr_balance=initial_inr,
            default_usd_balance=initial_usd,
        )

    def get_wallet(self, currency: Currency | str) -> VirtualWallet:
        """Retrieve the isolated wallet for the specified currency."""
        curr = Currency(currency) if isinstance(currency, str) else currency
        return self.inr_wallet if curr == Currency.INR else self.usd_wallet

    def get_wallet_for_symbol(self, symbol: str) -> VirtualWallet:
        """Retrieve appropriate wallet strictly determined by symbol exchange rules."""
        currency = get_instrument_currency(symbol)
        return self.get_wallet(currency)

    def update_market_prices(
        self,
        price_map: dict[str, float],
    ) -> None:
        """Update current market prices for open positions and
        snapshot equity curves.
        """
        updated_any = False
        for wallet in (self.inr_wallet, self.usd_wallet):
            wallet_updated = False
            for sym, pos in wallet.positions.items():
                if sym in price_map and price_map[sym] > 0:
                    pos.current_price = float(price_map[sym])
                    pos.updated_at = datetime.now(UTC).isoformat()
                    wallet_updated = True
            if wallet_updated:
                wallet.record_equity_snapshot()
                updated_any = True

        if updated_any:
            self.persist()

    def persist(self) -> None:
        """Atomically persist current wallet states to disk."""
        self.storage.save(self.inr_wallet, self.usd_wallet)

    def reset_portfolio(
        self,
        confirmation: bool,
        currency: Currency | str | None = None,
        new_inr_balance: float = DEFAULT_INR_INITIAL,
        new_usd_balance: float = DEFAULT_USD_INITIAL,
    ) -> dict[str, Any]:
        """Reset virtual portfolio balances after explicit user confirmation.

        Safety:
        - Requires confirmation=True.
        - Refuses automatic or accidental resets.
        """
        if not confirmation:
            raise ValueError(
                "Reset operation rejected: explicit confirmation=True required."
            )

        curr_enum = (
            Currency(currency)
            if (currency is not None and isinstance(currency, str))
            else currency
        )

        self.inr_wallet, self.usd_wallet = self.storage.reset(
            currency=curr_enum,
            inr_balance=new_inr_balance,
            usd_balance=new_usd_balance,
        )

        return {
            "status": "success",
            "message": (
                f"Virtual portfolio reset complete for "
                f"{curr_enum.value if curr_enum else 'ALL'} wallets."
            ),
            "INR": self.inr_wallet.to_dict(),
            "USD": self.usd_wallet.to_dict(),
        }

    def get_consolidated_summary(
        self,
        usd_inr_rate: float | None = None,
        rate_timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Generate dual-wallet summary with optional cross-currency valuation.

        Strict Currency Invariant:
        - INR and USD figures are ALWAYS presented in their native currencies.
        - Cross-currency consolidation is ONLY calculated when an explicit
          valid exchange rate is supplied, with rate provenance and friction notes.
        """
        inr_dict = self.inr_wallet.to_dict()
        usd_dict = self.usd_wallet.to_dict()

        summary: dict[str, Any] = {
            "disclaimer": (
                "SIMULATED BALANCES ONLY. These fictional wallets are for educational "
                "paper trading and cannot be converted to real funds or transferred."
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "wallets": {
                "INR": inr_dict,
                "USD": usd_dict,
            },
        }

        # Cross-currency consolidation if rate provided
        if usd_inr_rate is not None and usd_inr_rate > 0:
            ts = rate_timestamp or datetime.now(UTC).isoformat()
            # Valuate total combined equity in USD and in INR
            total_equity_usd = usd_dict["total_equity"] + (
                inr_dict["total_equity"] / usd_inr_rate
            )
            total_equity_inr = inr_dict["total_equity"] + (
                usd_dict["total_equity"] * usd_inr_rate
            )

            summary["consolidated_valuation"] = {
                "status": "CALCULATED",
                "usd_inr_conversion_rate": usd_inr_rate,
                "rate_timestamp": ts,
                "total_combined_equity_usd": round(total_equity_usd, 2),
                "total_combined_equity_inr": round(total_equity_inr, 2),
                "notes": (
                    "Valuation uses indicative exchange rate without currency "
                    "conversion transaction spread or FX transfer friction."
                ),
            }
        else:
            summary["consolidated_valuation"] = {
                "status": "UNAVAILABLE",
                "reason": (
                    "No verified USD/INR exchange rate provided. Consolidated "
                    "cross-currency valuation cannot be calculated reliably. "
                    "Wallets are tracked separately."
                ),
            }

        return summary
