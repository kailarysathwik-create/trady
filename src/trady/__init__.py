"""TRADY: Educational Quantitative Research and Paper-Trading Platform.

Purpose:
    Research whether statistical and machine-learning models can identify
    repeatable patterns in historical financial-market data.

Disclaimers:
    - TRADY does NOT claim or guarantee profits.
    - TRADY does NOT execute real-money trading or autonomous financial execution.
    - TRADY is strictly an educational and simulation research framework.
"""

from typing import Final

__version__: Final[str] = "0.1.0"
__platform__: Final[str] = "TRADY"

DISCLAIMER: Final[str] = (
    "TRADY is an educational quantitative research and paper-trading platform. "
    "TRADY does not provide investment advice, does not claim guaranteed profits, "
    "and does not execute real-money transactions."
)

__all__ = ["__version__", "__platform__", "DISCLAIMER"]
