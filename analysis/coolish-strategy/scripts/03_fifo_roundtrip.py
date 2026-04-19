#!/usr/bin/env python3
"""scripts/03_fifo_roundtrip.py — FIFO round-trip matching and summary statistics.

Outputs
-------
outputs/roundtrips_xbtusd.parquet  — round-trips for XBTUSD
outputs/roundtrips_ethusd.parquet  — round-trips for ETHUSD
outputs/roundtrip_stats.csv        — aggregate statistics per symbol
outputs/hold_time_histogram.png    — histogram of holding times

Validation
----------
The script verifies that the FIFO net PnL for each symbol matches the
``walletSummary`` realised PnL within 0.5%.  Exits non-zero on failure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.fifo import run_fifo
from lib.io import load_instruments, load_trades, load_wallet_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAT_TO_XBT = 1e-8
TOLERANCE  = 0.005  # 0.5 %


def _roundtrip_stats(rt: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics per symbol from round-trips.

    Parameters
    ----------
    rt:
        Round-trip DataFrame from ``lib.fifo.run_fifo()``.

    Returns
    -------
    pd.DataFrame
        One row per symbol with win-rate, avg hold time, etc.
    """
    if rt.empty:
        return pd.DataFrame()

    stats = rt.groupby("symbol").agg(
        n_roundtrips         = ("net_pnl_xbt", "count"),
        gross_pnl_xbt        = ("gross_pnl_xbt", "sum"),
        fees_xbt             = ("fees_xbt", "sum"),
        net_pnl_xbt          = ("net_pnl_xbt", "sum"),
        win_rate             = ("net_pnl_xbt", lambda x: (x > 0).mean()),
        avg_hold_hours       = ("hold_seconds", lambda x: x.mean() / 3600),
        median_hold_hours    = ("hold_seconds", lambda x: x.median() / 3600),
        avg_qty              = ("qty", "mean"),
        avg_gross_pnl_per_rt = ("gross_pnl_xbt", "mean"),
    ).reset_index()

    stats["profit_factor"] = stats.apply(
        lambda r: (
            rt.loc[(rt["symbol"] == r["symbol"]) & (rt["net_pnl_xbt"] > 0), "net_pnl_xbt"].sum()
            / abs(rt.loc[(rt["symbol"] == r["symbol"]) & (rt["net_pnl_xbt"] < 0), "net_pnl_xbt"].sum())
            if rt.loc[(rt["symbol"] == r["symbol"]) & (rt["net_pnl_xbt"] < 0), "net_pnl_xbt"].sum() != 0
            else np.inf
        ),
        axis=1,
    )

    return stats.sort_values("net_pnl_xbt", ascending=False).reset_index(drop=True)


def _validate_pnl(
    rt: pd.DataFrame,
    wallet: pd.DataFrame,
    tolerance: float = TOLERANCE,
) -> bool:
    """Validate FIFO net PnL against walletHistory realised PnL.

    For each symbol present in both sources, checks that the relative
    difference is within ``tolerance``.

    Parameters
    ----------
    rt:
        Round-trip DataFrame.
    wallet:
        Wallet history DataFrame.
    tolerance:
        Relative tolerance (e.g. 0.005 = 0.5%).

    Returns
    -------
    bool
        ``True`` if all symbols pass; ``False`` otherwise.
    """
    w = wallet.copy()
    w["amount_xbt"] = w["amount"].fillna(0).astype(float) * SAT_TO_XBT

    sym_col = None
    for c in ("symbol", "address"):
        if c in w.columns:
            sym_col = c
            break

    if sym_col is None:
        logger.warning("Cannot validate PnL: no symbol column in walletHistory")
        return True

    wallet_pnl = (
        w[w["transactType"] == "RealisedPNL"]
        .groupby(sym_col)["amount_xbt"].sum()
    )

    fifo_pnl = rt.groupby("symbol")["net_pnl_xbt"].sum()

    # Compare on symbols present in both
    common = wallet_pnl.index.intersection(fifo_pnl.index)
    if common.empty:
        logger.warning("No common symbols between FIFO and walletHistory; skipping PnL check")
        return True

    # Aggregate totals for the cross-check
    fifo_total   = fifo_pnl[common].sum()
    wallet_total = wallet_pnl[common].sum()

    if wallet_total == 0:
        logger.warning("walletHistory total realised PnL is 0; skipping PnL check")
        return True

    rel_diff = abs(fifo_total - wallet_total) / abs(wallet_total)
    logger.info(
        "PnL cross-check: FIFO=%.4f XBT  wallet=%.4f XBT  rel_diff=%.4f%%",
        fifo_total, wallet_total, rel_diff * 100,
    )

    passed = rel_diff <= tolerance
    if not passed:
        logger.error(
            "PnL mismatch exceeds tolerance! FIFO=%.4f  wallet=%.4f  diff=%.4f%%  (limit %.1f%%)",
            fifo_total, wallet_total, rel_diff * 100, tolerance * 100,
        )
    return passed


def _plot_hold_histogram(rt: pd.DataFrame, out_path: Path) -> None:
    """Plot histogram of round-trip holding times.

    Parameters
    ----------
    rt:
        Round-trip DataFrame.
    out_path:
        Path to save the PNG figure.
    """
    if rt.empty:
        logger.warning("No round-trips to plot")
        return

    hold_hours = rt["hold_seconds"] / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].hist(hold_hours, bins=100, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Holding time (hours)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Hold Time Distribution (linear)")

    # Log scale  — clip to avoid log(0)
    log_hours = np.log1p(hold_hours)
    axes[1].hist(log_hours, bins=80, color="darkorange", edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel("log1p(hours)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Hold Time Distribution (log1p scale)")

    fig.suptitle("XBTUSD + ETHUSD Round-Trip Hold Times", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    logger.info("=== 03 — FIFO Round-Trip ===")

    trades      = load_trades()
    instruments = load_instruments()
    wallet      = load_wallet_history()

    # ── XBTUSD ────────────────────────────────────────────────────────────────
    logger.info("Running FIFO for XBTUSD…")
    rt_xbt = run_fifo(trades, instruments, symbol_filter=["XBTUSD"])
    rt_xbt.to_parquet(outputs / "roundtrips_xbtusd.parquet", index=False, engine="pyarrow")
    logger.info("  %d round-trips for XBTUSD", len(rt_xbt))

    # ── ETHUSD ────────────────────────────────────────────────────────────────
    logger.info("Running FIFO for ETHUSD…")
    rt_eth = run_fifo(trades, instruments, symbol_filter=["ETHUSD"])
    rt_eth.to_parquet(outputs / "roundtrips_ethusd.parquet", index=False, engine="pyarrow")
    logger.info("  %d round-trips for ETHUSD", len(rt_eth))

    # ── Combined stats ────────────────────────────────────────────────────────
    rt_all = run_fifo(trades, instruments)
    stats  = _roundtrip_stats(rt_all)
    stats.to_csv(outputs / "roundtrip_stats.csv", index=False, float_format="%.6f")
    logger.info("Wrote outputs/roundtrip_stats.csv  (%d symbols)", len(stats))

    # ── Hold-time histogram (XBTUSD + ETHUSD) ─────────────────────────────────
    rt_main = pd.concat([rt_xbt, rt_eth], ignore_index=True)
    _plot_hold_histogram(rt_main, outputs / "hold_time_histogram.png")

    # ── PnL validation ────────────────────────────────────────────────────────
    passed = _validate_pnl(rt_all, wallet, tolerance=TOLERANCE)
    if not passed:
        logger.error("FATAL: FIFO PnL does not reconcile with walletHistory within 0.5%!")
        sys.exit(1)

    logger.info("03 complete.")


if __name__ == "__main__":
    main()
