"""lib/fifo.py — FIFO round-trip matching for BitMEX execution fills.

Pairs every buy fill against sell fills (and vice-versa) in strict FIFO order,
producing one row per completed round-trip.

Units
-----
- ``qty``            : number of contracts (integer, same as ``lastQty`` in tradeHistory)
- ``avg_open_px``    : price in USD (or quote currency) per contract
- ``avg_close_px``   : price in USD (or quote currency) per contract
- ``gross_pnl_xbt``  : gross profit/loss in XBT (float)
- ``fees_xbt``       : total fees paid in XBT (positive = paid, derived from execComm)
- ``net_pnl_xbt``    : gross_pnl_xbt - fees_xbt
- ``hold_seconds``   : (close_ts - open_ts).total_seconds()
- ``n_open_fills``   : number of individual fills that built the long/short leg
- ``n_close_fills``  : number of individual fills that closed the position
- ``max_open_qty``   : peak open quantity during the round-trip

For **inverse contracts** (``isInverse=True``), PnL is calculated as:

    pnl_xbt = qty * (1/open_px - 1/close_px) * sign

where ``sign = +1`` for a long round-trip (bought then sold) and ``-1`` for a
short round-trip (sold then bought).

For **linear / quanto contracts** (``isInverse=False``), PnL is:

    pnl_xbt = qty * (close_px - open_px) * sign * multiplier

``multiplier`` is taken from ``instrument.all.csv`` (units: XBt per contract
per unit of price movement; divide by 1e8 to get XBT).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class _FifoLot:
    """One open lot on the FIFO queue.

    Attributes
    ----------
    qty:
        Remaining open quantity (contracts, positive).
    px:
        Fill price (USD or quote currency).
    ts:
        Fill timestamp (tz-aware UTC).
    fee_xbt:
        Fee for this lot in XBT (proportionally allocated on partial match).
    """
    qty: int
    px: float
    ts: pd.Timestamp
    fee_xbt: float


@dataclass
class RoundTrip:
    """One completed FIFO round-trip.

    See module docstring for unit definitions.
    """
    symbol: str
    open_ts: pd.Timestamp
    close_ts: pd.Timestamp
    side: str                  # "Long" or "Short"
    qty: int
    avg_open_px: float
    avg_close_px: float
    gross_pnl_xbt: float
    fees_xbt: float
    net_pnl_xbt: float
    hold_seconds: float
    n_open_fills: int
    n_close_fills: int
    max_open_qty: int


# ── per-symbol FIFO matcher ───────────────────────────────────────────────────

class _FifoMatcher:
    """Incremental FIFO round-trip matcher for a single symbol.

    Parameters
    ----------
    symbol:
        BitMEX symbol string.
    is_inverse:
        Whether the contract is inverse (XBT-settled, BTC-margined).
    multiplier:
        Instrument multiplier from ``instrument.all.csv``.  For inverse
        contracts this is typically ``-1`` (XBt per contract per USD, negative
        because inverse).  Divide by 1e8 to convert satoshis → XBT.
    """

    def __init__(self, symbol: str, is_inverse: bool, multiplier: float) -> None:
        self.symbol = symbol
        self.is_inverse = is_inverse
        self.multiplier = multiplier  # satoshis per contract per unit price move

        self._long_queue:  list[_FifoLot] = []  # open long lots
        self._short_queue: list[_FifoLot] = []  # open short lots
        self._completed:   list[RoundTrip] = []

    # internal helpers --------------------------------------------------------

    def _calc_pnl(self, qty: int, open_px: float, close_px: float, side: str) -> float:
        """Calculate gross PnL in XBT for a matched lot.

        Parameters
        ----------
        qty:
            Matched quantity (contracts, positive).
        open_px:
            Average open price.
        close_px:
            Average close price.
        side:
            ``"Long"`` or ``"Short"``.

        Returns
        -------
        float
            Gross PnL in XBT.
        """
        sign = 1.0 if side == "Long" else -1.0
        if self.is_inverse and open_px > 0 and close_px > 0:
            # Inverse: 1 contract = 1 USD; settled in XBT
            # PnL (XBT) = qty × (1/open - 1/close) × sign
            # multiplier is not used for inverse contracts
            return float(qty) * (1.0 / open_px - 1.0 / close_px) * sign
        else:
            # Linear / quanto
            # multiplier is in XBt per contract per unit price move; convert → XBT
            # Do NOT use abs(): sign of multiplier determines direction of PnL
            mult_xbt = self.multiplier / 1e8 if self.multiplier != 0 else 1e-8
            return float(qty) * (close_px - open_px) * sign * mult_xbt

    def _drain_queue(
        self,
        open_queue: list[_FifoLot],
        close_lots: list[tuple[int, float, pd.Timestamp, float]],
        side: str,
    ) -> None:
        """Match close lots against the open FIFO queue and emit round-trips.

        Parameters
        ----------
        open_queue:
            The FIFO lot queue for the opening direction.
        close_lots:
            List of ``(qty, px, ts, fee_xbt)`` tuples representing closing fills.
        side:
            ``"Long"`` or ``"Short"``.
        """
        for close_qty, close_px, close_ts, close_fee in close_lots:
            remaining_close = close_qty

            while remaining_close > 0 and open_queue:
                lot = open_queue[0]
                matched = min(lot.qty, remaining_close)
                frac = matched / lot.qty
                open_fee_alloc   = lot.fee_xbt * frac
                close_fee_alloc  = close_fee * (matched / close_qty)

                gross = self._calc_pnl(matched, lot.px, close_px, side)
                total_fees = open_fee_alloc + close_fee_alloc

                rt = RoundTrip(
                    symbol        = self.symbol,
                    open_ts       = lot.ts,
                    close_ts      = close_ts,
                    side          = side,
                    qty           = matched,
                    avg_open_px   = lot.px,
                    avg_close_px  = close_px,
                    gross_pnl_xbt = gross,
                    fees_xbt      = total_fees,
                    net_pnl_xbt   = gross - total_fees,
                    hold_seconds  = (close_ts - lot.ts).total_seconds(),
                    n_open_fills  = 1,
                    n_close_fills = 1,
                    max_open_qty  = lot.qty,
                )
                self._completed.append(rt)

                lot.qty -= matched
                lot.fee_xbt -= open_fee_alloc
                remaining_close -= matched

                if lot.qty == 0:
                    open_queue.pop(0)

            # If more close than open, the remainder is a new short/long open
            if remaining_close > 0:
                # This shouldn't happen in a correctly sequenced book, but
                # we handle it gracefully by opening a reverse lot.
                close_fee_leftover = close_fee * (remaining_close / close_qty)
                opposite_queue = self._short_queue if side == "Long" else self._long_queue
                opposite_queue.append(
                    _FifoLot(
                        qty=remaining_close,
                        px=close_px,
                        ts=close_ts,
                        fee_xbt=close_fee_leftover,
                    )
                )

    # public interface --------------------------------------------------------

    def process_fill(
        self,
        side: str,
        qty: int,
        px: float,
        ts: pd.Timestamp,
        fee_xbt: float,
    ) -> None:
        """Process a single execution fill.

        Parameters
        ----------
        side:
            ``"Buy"`` or ``"Sell"`` (BitMEX convention).
        qty:
            Fill quantity in contracts (positive integer).
        px:
            Fill price.
        ts:
            Fill timestamp (tz-aware UTC).
        fee_xbt:
            Fee in XBT (positive = paid).  Derived from ``execComm / 1e8``.
        """
        if side == "Buy":
            # If there are open short positions, this closes them (Short round-trips)
            if self._short_queue:
                self._drain_queue(
                    self._short_queue,
                    [(qty, px, ts, fee_xbt)],
                    side="Short",
                )
                # Check if we have leftover that opened a new long
            else:
                self._long_queue.append(_FifoLot(qty=qty, px=px, ts=ts, fee_xbt=fee_xbt))
        else:  # Sell
            if self._long_queue:
                self._drain_queue(
                    self._long_queue,
                    [(qty, px, ts, fee_xbt)],
                    side="Long",
                )
            else:
                self._short_queue.append(_FifoLot(qty=qty, px=px, ts=ts, fee_xbt=fee_xbt))

    @property
    def completed(self) -> list[RoundTrip]:
        """List of completed round-trips."""
        return self._completed

    @property
    def open_long_qty(self) -> int:
        """Current open long quantity (contracts)."""
        return sum(lot.qty for lot in self._long_queue)

    @property
    def open_short_qty(self) -> int:
        """Current open short quantity (contracts)."""
        return sum(lot.qty for lot in self._short_queue)


# ── public API ───────────────────────────────────────────────────────────────

def run_fifo(
    trades: pd.DataFrame,
    instruments: pd.DataFrame,
    symbol_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run FIFO round-trip matching on a trade history DataFrame.

    Parameters
    ----------
    trades:
        Trade history as returned by ``lib.io.load_trades()``.  Required columns:
        ``symbol``, ``side``, ``lastQty``, ``lastPx``, ``timestamp``,
        ``execComm``.
    instruments:
        Instrument metadata as returned by ``lib.io.load_instruments()``.
        Required columns: ``symbol``, ``isInverse``, ``multiplier``.
    symbol_filter:
        If provided, only process fills for these symbols.

    Returns
    -------
    pd.DataFrame
        One row per completed round-trip with columns matching :class:`RoundTrip`.
    """
    required_trade_cols = {"symbol", "side", "lastQty", "lastPx", "timestamp", "execComm"}
    missing = required_trade_cols - set(trades.columns)
    if missing:
        raise ValueError(f"trades DataFrame is missing columns: {missing}")

    # Build instrument lookup: symbol → (is_inverse, multiplier)
    inst_idx: dict[str, tuple[bool, float]] = {}
    if instruments is not None and len(instruments) > 0:
        for _, row in instruments.iterrows():
            sym = row.get("symbol", "")
            is_inv = bool(row.get("isInverse", False))
            mult   = float(row.get("multiplier", 0) or 0)
            inst_idx[sym] = (is_inv, mult)

    # Filter to Trade and Settlement execTypes (Settlement closes quarterly futures)
    if "execType" in trades.columns:
        fills = trades[trades["execType"].isin(["Trade", "Settlement"])].copy()
    else:
        fills = trades.copy()

    if symbol_filter:
        fills = fills[fills["symbol"].isin(symbol_filter)]

    fills = fills.dropna(subset=["lastQty", "lastPx", "side"])
    fills = fills[fills["lastQty"] > 0]
    fills = fills.sort_values("timestamp").reset_index(drop=True)

    matchers: dict[str, _FifoMatcher] = {}

    for _, row in fills.iterrows():
        sym = str(row["symbol"])
        if sym not in matchers:
            is_inv, mult = inst_idx.get(sym, (True, -1))  # default: inverse
            matchers[sym] = _FifoMatcher(sym, is_inv, mult)

        qty  = int(row["lastQty"])
        px   = float(row["lastPx"])
        ts   = row["timestamp"]
        side = str(row["side"])

        exec_comm = row.get("execComm", 0)
        # execComm: positive = user paid fees, negative = maker rebate (user received).
        # Preserve the sign so that rebates reduce net cost instead of inflating it.
        fee_xbt   = float(exec_comm or 0) / 1e8  # satoshis → XBT; sign preserved

        matchers[sym].process_fill(side=side, qty=qty, px=px, ts=ts, fee_xbt=fee_xbt)

    all_trips: list[dict] = []
    for sym, matcher in matchers.items():
        for rt in matcher.completed:
            all_trips.append({
                "symbol":        rt.symbol,
                "open_ts":       rt.open_ts,
                "close_ts":      rt.close_ts,
                "side":          rt.side,
                "qty":           rt.qty,
                "avg_open_px":   rt.avg_open_px,
                "avg_close_px":  rt.avg_close_px,
                "gross_pnl_xbt": rt.gross_pnl_xbt,
                "fees_xbt":      rt.fees_xbt,
                "net_pnl_xbt":   rt.net_pnl_xbt,
                "hold_seconds":  rt.hold_seconds,
                "n_open_fills":  rt.n_open_fills,
                "n_close_fills": rt.n_close_fills,
                "max_open_qty":  rt.max_open_qty,
            })

    if not all_trips:
        return pd.DataFrame(columns=[
            "symbol", "open_ts", "close_ts", "side", "qty",
            "avg_open_px", "avg_close_px", "gross_pnl_xbt", "fees_xbt",
            "net_pnl_xbt", "hold_seconds", "n_open_fills", "n_close_fills",
            "max_open_qty",
        ])

    df_out = pd.DataFrame(all_trips)
    df_out = df_out.sort_values(["symbol", "open_ts"]).reset_index(drop=True)

    logger.info(
        "FIFO complete: %d round-trips across %d symbols",
        len(df_out), df_out["symbol"].nunique(),
    )
    return df_out
