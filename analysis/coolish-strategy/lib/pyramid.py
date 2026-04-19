"""lib/pyramid.py — Pyramid (scaling-in / scaling-out) episode identification.

A "pyramid episode" is a sequence of same-direction fills on the same symbol
where consecutive fills are separated by less than ``gap_minutes`` minutes.
When the gap between two same-direction fills exceeds the threshold, a new
episode begins.

Output columns per episode
--------------------------
- ``symbol``           : instrument symbol
- ``episode_id``       : monotonically increasing integer per symbol
- ``direction``        : ``"Long"`` (net buy) or ``"Short"`` (net sell)
- ``start_ts``         : timestamp of first fill in episode (tz-aware UTC)
- ``end_ts``           : timestamp of last fill in episode (tz-aware UTC)
- ``duration_seconds`` : (end_ts - start_ts).total_seconds()
- ``n_layers``         : number of distinct fills in the episode
- ``total_qty``        : sum of fill quantities (contracts)
- ``first_px``         : price of first fill
- ``last_px``          : price of last fill
- ``price_range_pct``  : abs(last_px - first_px) / first_px * 100
- ``layer_spacing_median_pct`` : median inter-fill price spacing as % of price
- ``avg_layer_qty``    : total_qty / n_layers
- ``notional_usd``     : total_qty * avg_price  (approximate USD notional)

ATR multiples are **not** computed here because ATR requires OHLCV data that
is not in the dataset; callers should enrich the output separately if needed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_GAP_MINUTES = 30


def identify_episodes(
    trades: pd.DataFrame,
    symbol_filter: Optional[list[str]] = None,
    gap_minutes: int = _DEFAULT_GAP_MINUTES,
) -> pd.DataFrame:
    """Identify pyramid (scaling) episodes from trade history.

    Two consecutive fills on the same symbol in the same direction are
    considered part of the same episode if they are within ``gap_minutes``
    of each other.  A direction change always starts a new episode.

    Parameters
    ----------
    trades:
        Trade history as returned by ``lib.io.load_trades()``.  Required
        columns: ``symbol``, ``side``, ``lastQty``, ``lastPx``, ``timestamp``.
    symbol_filter:
        If provided, only process fills for these symbols.
    gap_minutes:
        Inter-fill time gap (in minutes) above which a new episode begins.
        Default: 30 minutes.

    Returns
    -------
    pd.DataFrame
        One row per pyramid episode.
    """
    if "execType" in trades.columns:
        fills = trades[trades["execType"] == "Trade"].copy()
    else:
        fills = trades.copy()

    if symbol_filter:
        fills = fills[fills["symbol"].isin(symbol_filter)]

    fills = fills.dropna(subset=["lastQty", "lastPx", "side", "timestamp"])
    fills = fills[fills["lastQty"] > 0].copy()
    fills = fills.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    gap_td = pd.Timedelta(minutes=gap_minutes)
    records: list[dict] = []
    episode_id_counter: dict[str, int] = {}

    for sym, grp in fills.groupby("symbol", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        ep_counter = 0
        ep_rows: list[pd.Series] = []
        prev_ts: Optional[pd.Timestamp] = None
        prev_dir: Optional[str] = None

        def _flush_episode(rows: list[pd.Series], sym: str, ep_id: int) -> Optional[dict]:
            if not rows:
                return None
            qtys  = [int(r["lastQty"]) for r in rows]
            pxs   = [float(r["lastPx"]) for r in rows]
            tss   = [r["timestamp"] for r in rows]
            total_qty = sum(qtys)
            avg_px    = float(np.average(pxs, weights=qtys)) if total_qty > 0 else float(pxs[0])
            n_layers  = len(rows)
            first_px  = pxs[0]
            last_px   = pxs[-1]
            price_range_pct = (
                abs(last_px - first_px) / first_px * 100.0
                if first_px != 0 else 0.0
            )
            # Inter-fill price spacings
            if n_layers > 1:
                spacings = [
                    abs(pxs[i] - pxs[i - 1]) / pxs[i - 1] * 100.0
                    for i in range(1, n_layers)
                    if pxs[i - 1] != 0
                ]
                layer_spacing_median_pct = float(np.median(spacings)) if spacings else 0.0
            else:
                layer_spacing_median_pct = 0.0

            direction = "Long" if rows[0]["side"] == "Buy" else "Short"

            return {
                "symbol":                    sym,
                "episode_id":                ep_id,
                "direction":                 direction,
                "start_ts":                  tss[0],
                "end_ts":                    tss[-1],
                "duration_seconds":          (tss[-1] - tss[0]).total_seconds(),
                "n_layers":                  n_layers,
                "total_qty":                 total_qty,
                "first_px":                  first_px,
                "last_px":                   last_px,
                "price_range_pct":           price_range_pct,
                "layer_spacing_median_pct":  layer_spacing_median_pct,
                "avg_layer_qty":             total_qty / n_layers,
                "notional_usd":              total_qty * avg_px,
            }

        for _, row in grp.iterrows():
            cur_dir = "Long" if row["side"] == "Buy" else "Short"
            cur_ts  = row["timestamp"]

            new_ep = False
            if prev_dir is None:
                new_ep = True
            elif cur_dir != prev_dir:
                new_ep = True
            elif prev_ts is not None and (cur_ts - prev_ts) > gap_td:
                new_ep = True

            if new_ep and ep_rows:
                rec = _flush_episode(ep_rows, str(sym), ep_counter)
                if rec:
                    records.append(rec)
                ep_counter += 1
                ep_rows = []

            ep_rows.append(row)
            prev_dir = cur_dir
            prev_ts  = cur_ts

        # flush last episode
        if ep_rows:
            rec = _flush_episode(ep_rows, str(sym), ep_counter)
            if rec:
                records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["symbol", "start_ts"]).reset_index(drop=True)
    logger.info("Identified %d pyramid episodes across %d symbols", len(df), df["symbol"].nunique())
    return df
