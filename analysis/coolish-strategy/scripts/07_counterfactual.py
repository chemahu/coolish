#!/usr/bin/env python3
"""scripts/07_counterfactual.py — Four counterfactual scenario analyses.

Outputs
-------
outputs/counterfactuals.md   — Markdown report with 4 counterfactual results

Scenarios
---------
1. Remove net-negative symbols: PnL delta vs actual
2. Remove quarterly contracts entirely: PnL delta vs actual
3. All taker fills converted to maker (fee savings from instrument.all.csv)
4. "Stop-loss by symbol": move a symbol to exclusion list when its cumulative
   realised loss hits 2% of current account equity; measure PnL impact.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from lib.fifo import run_fifo
from lib.io import load_equity_curve, load_instruments, load_trades, load_wallet_history
from lib.symbols import classify_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAT_TO_XBT = 1e-8


# ── Scenario 1: remove net-negative symbols ───────────────────────────────────

def _scenario1(rt_all: pd.DataFrame) -> dict:
    """Counterfactual 1: delete all symbols with net negative PnL.

    Parameters
    ----------
    rt_all:
        All round-trips from ``lib.fifo.run_fifo()``.

    Returns
    -------
    dict
        Keys: actual_pnl_xbt, cf_pnl_xbt, delta_xbt, removed_symbols.
    """
    sym_pnl = rt_all.groupby("symbol")["net_pnl_xbt"].sum()
    positive_syms = sym_pnl[sym_pnl >= 0].index.tolist()
    negative_syms = sym_pnl[sym_pnl < 0].index.tolist()

    actual_pnl = rt_all["net_pnl_xbt"].sum()
    cf_pnl     = rt_all[rt_all["symbol"].isin(positive_syms)]["net_pnl_xbt"].sum()

    return {
        "actual_pnl_xbt":  actual_pnl,
        "cf_pnl_xbt":      cf_pnl,
        "delta_xbt":       cf_pnl - actual_pnl,
        "removed_symbols": negative_syms,
    }


# ── Scenario 2: remove quarterly contracts ────────────────────────────────────

def _scenario2(rt_all: pd.DataFrame) -> dict:
    """Counterfactual 2: remove all quarterly contract round-trips.

    Parameters
    ----------
    rt_all:
        All round-trips.

    Returns
    -------
    dict
        Keys: actual_pnl_xbt, cf_pnl_xbt, delta_xbt, quarterly_pnl_xbt.
    """
    rt_all = rt_all.copy()
    rt_all["symbol_class"] = rt_all["symbol"].apply(classify_symbol)
    quarterly_mask = rt_all["symbol_class"] == "quarterly"

    actual_pnl   = rt_all["net_pnl_xbt"].sum()
    quarterly_pnl = rt_all[quarterly_mask]["net_pnl_xbt"].sum()
    cf_pnl       = rt_all[~quarterly_mask]["net_pnl_xbt"].sum()

    return {
        "actual_pnl_xbt":    actual_pnl,
        "cf_pnl_xbt":        cf_pnl,
        "delta_xbt":         cf_pnl - actual_pnl,
        "quarterly_pnl_xbt": quarterly_pnl,
    }


# ── Scenario 3: all taker → maker fee savings ─────────────────────────────────

def _scenario3(trades: pd.DataFrame, instruments: pd.DataFrame) -> dict:
    """Counterfactual 3: replace all taker fees with maker fees.

    Parameters
    ----------
    trades:
        Trade history from ``load_trades()``.
    instruments:
        Instrument metadata from ``load_instruments()``.

    Returns
    -------
    dict
        Keys: actual_fees_xbt, cf_fees_xbt, savings_xbt, taker_fill_count.
    """
    if "execType" in trades.columns:
        fills = trades[trades["execType"] == "Trade"].copy()
    else:
        fills = trades.copy()

    if "lastLiquidityInd" not in fills.columns:
        return {"actual_fees_xbt": 0.0, "cf_fees_xbt": 0.0, "savings_xbt": 0.0,
                "taker_fill_count": 0, "note": "lastLiquidityInd column not found"}

    # Build fee rate lookup from instruments
    inst_fees: dict[str, tuple[float, float]] = {}  # symbol → (makerFee, takerFee)
    if instruments is not None:
        for _, row in instruments.iterrows():
            sym      = str(row.get("symbol", ""))
            maker_f  = float(row.get("makerFee", 0.0) or 0.0)
            taker_f  = float(row.get("takerFee", 0.0) or 0.0)
            inst_fees[sym] = (maker_f, taker_f)

    # Classify taker fills
    taker_mask = fills["lastLiquidityInd"].str.upper().isin(
        ["REMOVEDLIQUIDITY", "T", "TAKER"]
    )
    taker_fills = fills[taker_mask].copy()

    actual_fees_xbt = fills["execComm"].fillna(0).astype(float).sum() * SAT_TO_XBT

    # Counterfactual: if all taker fills had been maker fills
    # Use execCost (position cost in XBt satoshi) × makerFee to compute the
    # hypothetical maker commission.  This avoids needing to invert prices
    # (which fails for XBt-quoted contracts like ADAM20 where px ≈ 5e-6).
    cf_savings = 0.0
    for _, row in taker_fills.iterrows():
        sym       = str(row["symbol"])
        exec_cost = abs(float(row.get("execCost", 0) or 0))   # position cost in XBt sat

        maker_f, taker_f = inst_fees.get(sym, (-0.00025, 0.00075))
        actual_comm  = float(row.get("execComm", 0) or 0)     # sat
        cf_comm      = exec_cost * maker_f                     # sat (maker_f can be negative)
        cf_savings  += (actual_comm - cf_comm) * SAT_TO_XBT   # XBT saved

    return {
        "actual_fees_xbt":  actual_fees_xbt,
        "cf_fees_xbt":      actual_fees_xbt - cf_savings,
        "savings_xbt":      cf_savings,
        "taker_fill_count": int(taker_mask.sum()),
    }


# ── Scenario 4: stop-loss-by-symbol rule ─────────────────────────────────────

def _scenario4(
    rt_all: pd.DataFrame,
    equity_df: pd.DataFrame,
    threshold_pct: float = 2.0,
    exclusion_months: int = 12,
) -> dict:
    """Counterfactual 4: exclude symbol when cumulative loss >= 2% of equity.

    Walk round-trips chronologically.  When a non-core symbol's cumulative
    net PnL (since it was last admitted or since inception) dips below
    ``-threshold_pct`` of the current account equity, exclude it for
    ``exclusion_months`` months.

    Parameters
    ----------
    rt_all:
        All round-trips sorted by open_ts.
    equity_df:
        Equity curve for current account equity lookup.
    threshold_pct:
        Loss threshold as percentage of equity.
    exclusion_months:
        Months to exclude the symbol after triggering.

    Returns
    -------
    dict
        Keys: actual_pnl_xbt, cf_pnl_xbt, delta_xbt, n_exclusions.
    """
    CORE_SYMBOLS = {"XBTUSD", "XBTUSDT", "ETHUSD", "ETHUSDT"}

    eq_ts_col  = "timestamp"
    eq_val_col = next(
        (c for c in ("equityXBT", "equity", "walletBalance") if c in equity_df.columns),
        None,
    )

    if eq_val_col is None or rt_all.empty:
        return {"actual_pnl_xbt": 0.0, "cf_pnl_xbt": 0.0, "delta_xbt": 0.0, "n_exclusions": 0}

    equity_df = equity_df.dropna(subset=[eq_ts_col, eq_val_col]).sort_values(eq_ts_col)
    eq_ts_s   = equity_df[eq_ts_col].reset_index(drop=True)
    eq_val_s  = equity_df[eq_val_col].astype(float).reset_index(drop=True)
    if eq_val_col == "walletBalance":
        eq_val_s = eq_val_s * SAT_TO_XBT

    rt_sorted = rt_all.sort_values("open_ts").reset_index(drop=True)
    actual_pnl = rt_sorted["net_pnl_xbt"].sum()

    # Track cumulative PnL per symbol and exclusion windows
    sym_cum_pnl:   dict[str, float] = {}
    exclusions:    dict[str, pd.Timestamp] = {}  # symbol → exclusion end ts
    cf_pnl        = 0.0
    n_exclusions  = 0

    def _get_equity(ts: pd.Timestamp) -> float:
        mask = eq_ts_s <= ts
        if not mask.any():
            return 1.0
        return float(eq_val_s.iloc[mask[mask].index[-1]])

    for _, row in rt_sorted.iterrows():
        sym    = row["symbol"]
        ts     = row["open_ts"]
        pnl    = row["net_pnl_xbt"]

        # Skip core symbols
        if sym in CORE_SYMBOLS:
            cf_pnl += pnl
            continue

        # Check if still in exclusion window
        if sym in exclusions and ts < exclusions[sym]:
            continue  # skipped — trade would not have been taken

        # Track running PnL
        sym_cum_pnl[sym] = sym_cum_pnl.get(sym, 0.0) + pnl
        cf_pnl += pnl

        # Check trigger
        cur_equity = _get_equity(ts)
        threshold_xbt = cur_equity * threshold_pct / 100.0

        if sym_cum_pnl[sym] < -threshold_xbt:
            # Trigger exclusion
            excl_end = ts + pd.DateOffset(months=exclusion_months)
            exclusions[sym] = excl_end
            sym_cum_pnl[sym] = 0.0  # reset cumulative PnL counter
            n_exclusions += 1
            logger.debug("Excluded %s until %s (trigger at %s)", sym, excl_end, ts)

    return {
        "actual_pnl_xbt": actual_pnl,
        "cf_pnl_xbt":     cf_pnl,
        "delta_xbt":      cf_pnl - actual_pnl,
        "n_exclusions":   n_exclusions,
    }


# ── Report generator ──────────────────────────────────────────────────────────

def _build_report(s1: dict, s2: dict, s3: dict, s4: dict) -> str:
    """Build the Markdown counterfactuals report.

    Parameters
    ----------
    s1, s2, s3, s4:
        Result dicts from each scenario function.

    Returns
    -------
    str
        Formatted Markdown.
    """
    def _fmt(v: float) -> str:
        return f"{v:+.4f} XBT"

    removed_syms = ", ".join(s1.get("removed_symbols", [])[:20])

    lines = [
        "# Counterfactual Analysis",
        "",
        "All PnL figures in XBT.  Positive delta = counterfactual is better than actual.",
        "",
        "---",
        "",
        "## Scenario 1 — Remove Net-Negative Symbols",
        "",
        "Exclude any symbol whose total FIFO net PnL is negative over the full history.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Actual net PnL | {_fmt(s1['actual_pnl_xbt'])} |",
        f"| Counterfactual net PnL | {_fmt(s1['cf_pnl_xbt'])} |",
        f"| Delta | **{_fmt(s1['delta_xbt'])}** |",
        f"| Symbols removed | {removed_syms or '—'} |",
        "",
        "---",
        "",
        "## Scenario 2 — Remove Quarterly Contracts",
        "",
        "Exclude all trades on quarterly / futures contracts (H/M/U/Z suffix).",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Actual net PnL | {_fmt(s2['actual_pnl_xbt'])} |",
        f"| Counterfactual net PnL | {_fmt(s2['cf_pnl_xbt'])} |",
        f"| Delta | **{_fmt(s2['delta_xbt'])}** |",
        f"| Quarterly contracts PnL (removed) | {_fmt(s2['quarterly_pnl_xbt'])} |",
        "",
        "---",
        "",
        "## Scenario 3 — Convert All Taker Fills to Maker",
        "",
        "Re-price all taker executions at the instrument's maker fee rate.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Actual total fees | {_fmt(s3.get('actual_fees_xbt', 0.0))} |",
        f"| Counterfactual fees | {_fmt(s3.get('cf_fees_xbt', 0.0))} |",
        f"| Estimated savings | **{_fmt(s3.get('savings_xbt', 0.0))}** |",
        f"| Taker fills converted | {s3.get('taker_fill_count', 0):,} |",
        "",
        "---",
        "",
        "## Scenario 4 — Symbol Stop-Loss Rule (2% equity, 12-month exclusion)",
        "",
        "When any non-core symbol's cumulative net loss exceeds 2% of current account",
        "equity, exclude it from trading for 12 months and reset the counter.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Actual net PnL | {_fmt(s4['actual_pnl_xbt'])} |",
        f"| Counterfactual net PnL | {_fmt(s4['cf_pnl_xbt'])} |",
        f"| Delta | **{_fmt(s4['delta_xbt'])}** |",
        f"| Exclusion events triggered | {s4['n_exclusions']} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    logger.info("=== 07 — Counterfactuals ===")

    trades      = load_trades()
    instruments = load_instruments()
    wallet      = load_wallet_history()
    equity_df   = load_equity_curve()

    logger.info("Running FIFO for all symbols…")
    rt_all = run_fifo(trades, instruments)

    logger.info("Scenario 1: net-negative symbols…")
    s1 = _scenario1(rt_all)
    logger.info("  delta=%.4f XBT", s1["delta_xbt"])

    logger.info("Scenario 2: quarterly contracts…")
    s2 = _scenario2(rt_all)
    logger.info("  delta=%.4f XBT", s2["delta_xbt"])

    logger.info("Scenario 3: taker→maker fee savings…")
    s3 = _scenario3(trades, instruments)
    logger.info("  savings=%.4f XBT", s3.get("savings_xbt", 0.0))

    logger.info("Scenario 4: symbol stop-loss rule…")
    s4 = _scenario4(rt_all, equity_df)
    logger.info("  delta=%.4f XBT  exclusions=%d", s4["delta_xbt"], s4["n_exclusions"])

    report = _build_report(s1, s2, s3, s4)
    out_path = outputs / "counterfactuals.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote %s", out_path)

    logger.info("07 complete.")


if __name__ == "__main__":
    main()
