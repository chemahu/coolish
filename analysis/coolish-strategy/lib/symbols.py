"""lib/symbols.py — Symbol classification helpers.

All symbols are from BitMEX; classifications follow the naming conventions
used in the bwjoke/BTC-Trading-Since-2020 dataset.
"""

from __future__ import annotations

import re
from typing import Literal

SymbolClass = Literal[
    "xbt_perp", "eth_perp", "alt_perp", "quarterly", "spot", "conversion", "other"
]

# Quarterly suffix pattern: one of H/M/U/Z followed by exactly two digits.
# e.g. XBTH21, ETHU24, LTCZ20
_QUARTERLY_RE = re.compile(r"[HMUZ]\d{2}$")

# Known XBT perpetuals
_XBT_PERP = {"XBTUSD", "XBTUSDT"}

# Known ETH perpetuals
_ETH_PERP = {"ETHUSD", "ETHUSDT"}

# Known spot / conversion symbols (currency conversion rows in wallet history)
_CONVERSION = {"XBTUSD_conv", "USDXBT", "XBTEUR", "EURXBT", "XBTUSDT_conv"}


def classify_symbol(sym: str) -> SymbolClass:
    """Classify a BitMEX symbol into one of seven categories.

    Parameters
    ----------
    sym:
        Raw symbol string from the BitMEX API (e.g. ``"XBTUSD"``, ``"ETHU24"``).

    Returns
    -------
    SymbolClass
        One of: ``xbt_perp``, ``eth_perp``, ``alt_perp``, ``quarterly``,
        ``spot``, ``conversion``, ``other``.

    Examples
    --------
    >>> classify_symbol("XBTUSD")
    'xbt_perp'
    >>> classify_symbol("ETHU24")
    'quarterly'
    >>> classify_symbol("DOTUSDT")
    'alt_perp'
    """
    if not sym or not isinstance(sym, str):
        return "other"

    upper = sym.upper()

    if upper in _XBT_PERP:
        return "xbt_perp"
    if upper in _ETH_PERP:
        return "eth_perp"
    if upper in _CONVERSION:
        return "conversion"

    # Quarterly contracts: symbol ends with H/M/U/Z + two-digit year
    if _QUARTERLY_RE.search(upper):
        return "quarterly"

    # Perpetual altcoins: ends with "USD", "USDT", "XBT" but not already matched
    if upper.endswith(("USD", "USDT", "XBT", "ETH")):
        return "alt_perp"

    # Spot-like or misc
    return "other"
