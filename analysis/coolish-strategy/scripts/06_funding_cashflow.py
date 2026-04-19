#!/usr/bin/env python3
"""scripts/06_funding_cashflow.py — Funding income and withdrawal timing analysis.

Outputs
-------
outputs/funding_yearly.csv              — yearly funding net (amount - fee) in XBT
outputs/withdrawal_vs_high.png          — scatter: days since equity high vs withdrawal size
outputs/deposit_vs_drawdown.png         — scatter: drawdown depth at deposit vs deposit size
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

from lib.io import load_equity_curve, load_wallet_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAT_TO_XBT = 1e-8


def _funding_yearly(wallet: pd.DataFrame) -> pd.DataFrame:
    """Aggregate funding-related wallet events by year.

    Parameters
    ----------
    wallet:
        Wallet history from ``load_wallet_history()``.

    Returns
    -------
    pd.DataFrame
        Columns: year, funding_amount_xbt, funding_fee_xbt, funding_net_xbt.
    """
    ts_col = "transactTime" if "transactTime" in wallet.columns else "timestamp"
    w = wallet.copy()
    w["year"]       = w[ts_col].dt.year
    w["amount_xbt"] = w["amount"].fillna(0).astype(float) * SAT_TO_XBT
    w["fee_xbt"]    = w["fee"].fillna(0).astype(float) * SAT_TO_XBT

    fund = w[w["transactType"] == "Funding"]
    agg  = fund.groupby("year").agg(
        funding_amount_xbt = ("amount_xbt", "sum"),
        funding_fee_xbt    = ("fee_xbt", lambda s: s.abs().sum()),
    ).reset_index()
    agg["funding_net_xbt"] = agg["funding_amount_xbt"] - agg["funding_fee_xbt"]
    return agg


def _days_since_high(equity_ts: pd.Series, equity_val: pd.Series, event_ts: pd.Timestamp) -> float:
    """Return number of days between the last equity all-time high before event_ts and event_ts.

    Parameters
    ----------
    equity_ts:
        Timestamp series for equity curve.
    equity_val:
        Equity value series (same index).
    event_ts:
        Timestamp of the event.

    Returns
    -------
    float
        Days since last all-time high, or NaN if not determinable.
    """
    mask = equity_ts <= event_ts
    if not mask.any():
        return np.nan

    eq_before = equity_val[mask]
    ts_before  = equity_ts[mask]
    running_max = eq_before.expanding().max()
    # Last time equity was at or above current running max
    ath_idx = running_max.idxmax()
    ath_ts  = ts_before.loc[ath_idx]
    return (event_ts - ath_ts).total_seconds() / 86400.0


def _withdrawal_vs_high(wallet: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    """Build dataset of withdrawals annotated with days since equity high.

    Parameters
    ----------
    wallet:
        Wallet history from ``load_wallet_history()``.
    equity_df:
        Equity curve from ``load_equity_curve()``.

    Returns
    -------
    pd.DataFrame
        Columns: ts, withdrawal_xbt, days_since_high.
    """
    ts_col = "transactTime" if "transactTime" in wallet.columns else "timestamp"
    w = wallet.copy()
    w["amount_xbt"] = w["amount"].fillna(0).astype(float) * SAT_TO_XBT

    withdrawals = w[w["transactType"] == "Withdrawal"].copy()
    withdrawals["withdrawal_xbt"] = withdrawals["amount_xbt"].abs()

    # Resolve equity
    eq_ts_col  = "timestamp"
    eq_val_col = next(
        (c for c in ("equityXBT", "equity", "walletBalance") if c in equity_df.columns),
        None,
    )
    if eq_val_col is None or eq_ts_col not in equity_df.columns:
        return pd.DataFrame()

    equity_df  = equity_df.dropna(subset=[eq_ts_col, eq_val_col]).sort_values(eq_ts_col)
    eq_ts_s    = equity_df[eq_ts_col].reset_index(drop=True)
    eq_val_s   = equity_df[eq_val_col].astype(float).reset_index(drop=True)
    if eq_val_col == "walletBalance":
        eq_val_s = eq_val_s * SAT_TO_XBT

    records = []
    for _, row in withdrawals.iterrows():
        evt_ts = row[ts_col]
        days   = _days_since_high(eq_ts_s, eq_val_s, evt_ts)
        records.append({
            "ts":               evt_ts,
            "withdrawal_xbt":   row["withdrawal_xbt"],
            "days_since_high":  days,
        })

    return pd.DataFrame(records)


def _deposit_drawdown(wallet: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    """Build dataset of deposits annotated with drawdown at time of deposit.

    Parameters
    ----------
    wallet:
        Wallet history from ``load_wallet_history()``.
    equity_df:
        Equity curve from ``load_equity_curve()``.

    Returns
    -------
    pd.DataFrame
        Columns: ts, deposit_xbt, drawdown_pct.
    """
    ts_col = "transactTime" if "transactTime" in wallet.columns else "timestamp"
    w = wallet.copy()
    w["amount_xbt"] = w["amount"].fillna(0).astype(float) * SAT_TO_XBT

    deposits = w[w["transactType"] == "Deposit"].copy()
    deposits["deposit_xbt"] = deposits["amount_xbt"].abs()

    eq_ts_col  = "timestamp"
    eq_val_col = next(
        (c for c in ("equityXBT", "equity", "walletBalance") if c in equity_df.columns),
        None,
    )
    if eq_val_col is None or eq_ts_col not in equity_df.columns:
        return pd.DataFrame()

    equity_df = equity_df.dropna(subset=[eq_ts_col, eq_val_col]).sort_values(eq_ts_col)
    eq_ts_s   = equity_df[eq_ts_col].reset_index(drop=True)
    eq_val_s  = equity_df[eq_val_col].astype(float).reset_index(drop=True)
    if eq_val_col == "walletBalance":
        eq_val_s = eq_val_s * SAT_TO_XBT

    running_max = eq_val_s.expanding().max()

    records = []
    for _, row in deposits.iterrows():
        evt_ts  = row[ts_col]
        mask    = eq_ts_s <= evt_ts
        if not mask.any():
            continue
        last_i  = mask[mask].index[-1]
        cur_eq  = eq_val_s.iloc[last_i]
        cur_max = running_max.iloc[last_i]
        dd_pct  = (cur_eq - cur_max) / cur_max * 100.0 if cur_max != 0 else 0.0
        records.append({
            "ts":           evt_ts,
            "deposit_xbt":  row["deposit_xbt"],
            "drawdown_pct": dd_pct,
        })

    return pd.DataFrame(records)


def _plot_withdrawal_vs_high(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter: days since equity high vs withdrawal size (XBT)."""
    if df.empty:
        logger.warning("No withdrawal data for plot; skipping")
        return
    df = df.dropna(subset=["days_since_high", "withdrawal_xbt"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["days_since_high"], df["withdrawal_xbt"],
               alpha=0.6, s=40, color="darkorange")
    ax.set_xlabel("Days since last equity ATH")
    ax.set_ylabel("Withdrawal size (XBT)")
    ax.set_title("Withdrawal vs Days Since Equity All-Time High")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_deposit_vs_drawdown(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter: drawdown depth at deposit time vs deposit size (XBT)."""
    if df.empty:
        logger.warning("No deposit data for plot; skipping")
        return
    df = df.dropna(subset=["drawdown_pct", "deposit_xbt"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["drawdown_pct"], df["deposit_xbt"],
               alpha=0.6, s=40, color="steelblue")
    ax.set_xlabel("Drawdown at deposit time (%)")
    ax.set_ylabel("Deposit size (XBT)")
    ax.set_title("Deposit Size vs Drawdown Depth")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    logger.info("=== 06 — Funding Cashflow ===")

    wallet    = load_wallet_history()
    equity_df = load_equity_curve()

    # ── Funding yearly ────────────────────────────────────────────────────────
    fund = _funding_yearly(wallet)
    fund.to_csv(outputs / "funding_yearly.csv", index=False, float_format="%.6f")
    logger.info("Wrote funding_yearly.csv")

    # ── Withdrawal vs equity high ─────────────────────────────────────────────
    wd_df = _withdrawal_vs_high(wallet, equity_df)
    _plot_withdrawal_vs_high(wd_df, outputs / "withdrawal_vs_high.png")

    # ── Deposit vs drawdown ───────────────────────────────────────────────────
    dep_df = _deposit_drawdown(wallet, equity_df)
    _plot_deposit_vs_drawdown(dep_df, outputs / "deposit_vs_drawdown.png")

    logger.info("06 complete.")


if __name__ == "__main__":
    main()
