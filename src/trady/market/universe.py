"""Searchable Market Universe for US and Indian Instruments."""

from dataclasses import dataclass
from typing import Any

from trady.portfolio.models import Currency


@dataclass(frozen=True)
class InstrumentInfo:
    """Canonical metadata for a supported market instrument."""

    symbol: str
    name: str
    exchange: str
    currency: Currency
    asset_type: str  # EQUITY, ETF, INDEX
    country: str  # US, IN
    sector: str = "General"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "currency": self.currency.value,
            "currency_symbol": self.currency.symbol,
            "asset_type": self.asset_type,
            "country": self.country,
            "sector": self.sector,
        }


# Curated, searchable multi-market universe covering US and India
CANONICAL_UNIVERSE: list[InstrumentInfo] = [
    # US Major Indices & ETFs
    InstrumentInfo(
        "SPY",
        "SPDR S&P 500 ETF Trust",
        "NYSE Arca",
        Currency.USD,
        "ETF",
        "US",
        "Broad Market",
    ),
    InstrumentInfo(
        "QQQ",
        "Invesco QQQ Trust (Nasdaq 100)",
        "NASDAQ",
        Currency.USD,
        "ETF",
        "US",
        "Technology",
    ),
    InstrumentInfo(
        "IWM",
        "iShares Russell 2000 ETF",
        "NYSE Arca",
        Currency.USD,
        "ETF",
        "US",
        "Small Cap",
    ),
    InstrumentInfo(
        "DIA",
        "SPDR Dow Jones Industrial Average ETF",
        "NYSE Arca",
        Currency.USD,
        "ETF",
        "US",
        "Large Cap",
    ),
    # US Large-Cap Equities
    InstrumentInfo(
        "AAPL",
        "Apple Inc.",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "Consumer Electronics",
    ),
    InstrumentInfo(
        "MSFT",
        "Microsoft Corporation",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "Enterprise Software",
    ),
    InstrumentInfo(
        "NVDA",
        "NVIDIA Corporation",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "Semiconductors / AI",
    ),
    InstrumentInfo(
        "AMZN",
        "Amazon.com Inc.",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "E-Commerce / Cloud",
    ),
    InstrumentInfo(
        "GOOGL",
        "Alphabet Inc. (Class A)",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "Internet / Software",
    ),
    InstrumentInfo(
        "META",
        "Meta Platforms, Inc.",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "Social Media",
    ),
    InstrumentInfo(
        "TSLA",
        "Tesla, Inc.",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "Electric Vehicles / Clean Energy",
    ),
    InstrumentInfo(
        "JPM",
        "JPMorgan Chase & Co.",
        "NYSE",
        Currency.USD,
        "EQUITY",
        "US",
        "Financial Services",
    ),
    # US High-Growth Tech & Semiconductor Equities
    InstrumentInfo(
        "AMD",
        "Advanced Micro Devices, Inc.",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "Semiconductors",
    ),
    InstrumentInfo(
        "PLTR",
        "Palantir Technologies Inc.",
        "NYSE",
        Currency.USD,
        "EQUITY",
        "US",
        "Enterprise Software / AI",
    ),
    InstrumentInfo(
        "COIN",
        "Coinbase Global, Inc.",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "Crypto Exchange / FinTech",
    ),
    InstrumentInfo(
        "NFLX",
        "Netflix, Inc.",
        "NASDAQ",
        Currency.USD,
        "EQUITY",
        "US",
        "Entertainment / Streaming",
    ),
    # Indian Major Indices
    InstrumentInfo(
        "^NSEI",
        "Nifty 50 Index",
        "NSE",
        Currency.INR,
        "INDEX",
        "IN",
        "Broad Market Benchmark",
    ),
    InstrumentInfo(
        "^NSEBANK",
        "Nifty Bank Index",
        "NSE",
        Currency.INR,
        "INDEX",
        "IN",
        "Banking Benchmark",
    ),
    InstrumentInfo(
        "^BSESN",
        "S&P BSE SENSEX",
        "BSE",
        Currency.INR,
        "INDEX",
        "IN",
        "Broad Market Benchmark",
    ),
    # Indian Large-Cap Equities (NSE)
    InstrumentInfo(
        "RELIANCE.NS",
        "Reliance Industries Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Energy & Telecom",
    ),
    InstrumentInfo(
        "TCS.NS",
        "Tata Consultancy Services Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "IT Services",
    ),
    InstrumentInfo(
        "HDFCBANK.NS",
        "HDFC Bank Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Private Banking",
    ),
    InstrumentInfo(
        "INFY.NS", "Infosys Limited", "NSE", Currency.INR, "EQUITY", "IN", "IT Services"
    ),
    InstrumentInfo(
        "ICICIBANK.NS",
        "ICICI Bank Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Private Banking",
    ),
    InstrumentInfo(
        "SBIN.NS",
        "State Bank of India",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Public Banking",
    ),
    InstrumentInfo(
        "BHARTIARTL.NS",
        "Bharti Airtel Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Telecommunications",
    ),
    InstrumentInfo(
        "ITC.NS",
        "ITC Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Consumer Goods / FMCG",
    ),
    InstrumentInfo(
        "LT.NS",
        "Larsen & Toubro Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Engineering & Construction",
    ),
    InstrumentInfo(
        "TATAMOTORS.NS",
        "Tata Motors Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Automotive",
    ),
    InstrumentInfo(
        "TATASTEEL.NS",
        "Tata Steel Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Metals & Mining",
    ),
    InstrumentInfo(
        "BAJFINANCE.NS",
        "Bajaj Finance Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Non-Banking Financial",
    ),
    InstrumentInfo(
        "MARUTI.NS",
        "Maruti Suzuki India Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Automotive",
    ),
    InstrumentInfo(
        "SUNPHARMA.NS",
        "Sun Pharmaceutical Industries Ltd",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Pharmaceuticals",
    ),
    InstrumentInfo(
        "WIPRO.NS",
        "Wipro Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "IT Services",
    ),
    InstrumentInfo(
        "HINDUNILVR.NS",
        "Hindustan Unilever Limited",
        "NSE",
        Currency.INR,
        "EQUITY",
        "IN",
        "Consumer FMCG",
    ),
]

_UNIVERSE_INDEX = {inst.symbol.upper(): inst for inst in CANONICAL_UNIVERSE}


def get_instrument_universe() -> list[InstrumentInfo]:
    """Return all instruments in the supported universe."""
    return list(CANONICAL_UNIVERSE)


def get_instrument_info(symbol: str) -> InstrumentInfo:
    """Retrieve instrument metadata or synthesize canonical fallback."""
    clean = symbol.strip().upper()
    if clean in _UNIVERSE_INDEX:
        return _UNIVERSE_INDEX[clean]

    # Synthesize fallback for unlisted but valid symbol
    is_inr = (
        clean.endswith(".NS")
        or clean.endswith(".BO")
        or clean.startswith("^NSE")
        or clean.startswith("^BSE")
    )
    exch = (
        "NSE" if clean.endswith(".NS") else ("BSE" if clean.endswith(".BO") else "US")
    )
    return InstrumentInfo(
        symbol=clean,
        name=clean,
        exchange=exch,
        currency=Currency.INR if is_inr else Currency.USD,
        asset_type="EQUITY",
        country="IN" if is_inr else "US",
        sector="Unspecified",
    )


def search_universe(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search instrument universe by ticker symbol, company name, or sector."""
    q = query.strip().upper()
    if not q:
        return [inst.to_dict() for inst in CANONICAL_UNIVERSE[:limit]]

    matches: list[InstrumentInfo] = []
    # Priority 1: Exact symbol match
    for inst in CANONICAL_UNIVERSE:
        if inst.symbol.upper() == q:
            matches.append(inst)

    # Priority 2: Symbol prefix match
    for inst in CANONICAL_UNIVERSE:
        if inst not in matches and inst.symbol.upper().startswith(q):
            matches.append(inst)

    # Priority 3: Name or sector substring match
    for inst in CANONICAL_UNIVERSE:
        if inst not in matches and (
            q in inst.name.upper()
            or q in inst.sector.upper()
            or q in inst.exchange.upper()
        ):
            matches.append(inst)

    return [inst.to_dict() for inst in matches[:limit]]
