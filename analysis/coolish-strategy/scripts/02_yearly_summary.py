#!/usr/bin/env python3
"""scripts/02_yearly_summary.py — Compute yearly performance summary tables.

Outputs
-------
outputs/yearly_summary.csv         — per-year key metrics
outputs/btc_share_quarterly.csv    — BTC % of notional per quarter
outputs/symbol_pnl_ranking.csv     — per-symbol PnL ranked, with symbol class
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from lib.io import load_equity_curve, load_trades, load_wallet_history
from lib.symbols import classify_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAT_TO_XBT = 1e-8  # 1 satoshi = 1e-8 XBT


def _yearly_trade_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade-level stats by year.

    Parameters
    ----------
    trades:
        Full trade history DataFrame from ``load_trades()``.

    Returns
    -------
    pd.DataFrame
        Columns: year, n_fills, total_notional_xbt, gross_pnl_xbt, fees_xbt,
        net_pnl_xbt, btc_share_pct.
    """
    ts_col = "timestamp" if "timestamp" in trades.columns else "transactTime"

    # execType == "Trade" only
    if "execType" in trades.columns:
        fills = trades[trades["execType"] == "Trade"].copy()
    else:
        fills = trades.copy()

    fills["year"] = fills[ts_col].dt.year

    # homeNotional is in XBT for inverse contracts
    fills["home_notional_xbt"] = fills["homeNotional"].fillna(0.0).abs()
    fills["exec_comm_xbt"] = fills["execComm"].fillna(0).astype(float) * SAT_TO_XBT

    # Mark which fills are BTC-related
    fills["is_btc"] = fills["symbol"].str.upper().str.contains("XBT", na=False)

    agg = fills.groupby("year").agg(
        n_fills              = ("execID", "count"),
        total_notional_xbt   = ("home_notional_xbt", "sum"),
        fees_xbt             = ("exec_comm_xbt", "sum"),
        btc_notional_xbt     = ("home_notional_xbt", lambda s: s[fills.loc[s.index, "is_btc"]].sum()),
    ).reset_index()

    # fees are stored as negative in execComm (cost); make positive for display
    agg["fees_xbt"] = agg["fees_xbt"].abs()

    agg["btc_share_pct"] = (
        agg["btc_notional_xbt"] / agg["total_notional_xbt"].replace(0, np.nan) * 100.0
    ).round(2)

    return agg[["year", "n_fills", "total_notional_xbt", "fees_xbt", "btc_share_pct"]]


def _yearly_order_stats(orders: pd.DataFrame) -> pd.DataFrame:
    """Count orders per year.

    Parameters
    ----------
    orders:
        Order DataFrame from ``load_orders()``.

    Returns
    -------
    pd.DataFrame
        Columns: year, n_orders.
    """
    ts_col = "timestamp" if "timestamp" in orders.columns else "transactTime"
    orders = orders.copy()
    orders["year"] = orders[ts_col].dt.year
    return orders.groupby("year").agg(n_orders=("orderID", "count")).reset_index()


def _yearly_wallet_stats(wallet: pd.DataFrame) -> pd.DataFrame:
    """Extract deposits, withdrawals, and year-end equity from wallet history.

    Parameters
    ----------
    wallet:
        Wallet history DataFrame from ``load_wallet_history()``.

    Returns
    -------
    pd.DataFrame
        Columns: year, deposits_xbt, withdrawals_xbt, gross_pnl_xbt,
        year_end_equity_xbt.
    """
    ts_col = "transactTime" if "transactTime" in wallet.columns else "timestamp"
    w = wallet.copy()
    w["year"] = w[ts_col].dt.year
    w["amount_xbt"] = w["amount"].fillna(0).astype(float) * SAT_TO_XBT

    # Deposits: transactType == "Deposit"
    deps = (
        w[w["transactType"] == "Deposit"]
        .groupby("year")["amount_xbt"].sum()
        .rename("deposits_xbt")
    )

    # Withdrawals: transactType == "Withdrawal"
    withs = (
        w[w["transactType"] == "Withdrawal"]
        .groupby("year")["amount_xbt"].sum()
        .abs()
        .rename("withdrawals_xbt")
    )

    # Gross PnL: transactType == "RealisedPNL"
    pnl = (
        w[w["transactType"] == "RealisedPNL"]
        .groupby("year")["amount_xbt"].sum()
        .rename("gross_pnl_xbt")
    )

    # Year-end equity: take the last walletBalance per year
    w["wallet_balance_xbt"] = w["walletBalance"].fillna(0).astype(float) * SAT_TO_XBT
    ye_equity = (
        w.sort_values(ts_col)
        .groupby("year")["wallet_balance_xbt"].last()
        .rename("year_end_equity_xbt")
    )

    out = (
        pd.DataFrame({"year": sorted(w["year"].dropna().unique().astype(int))})
        .set_index("year")
        .join(deps)
        .join(withs)
        .join(pnl)
        .join(ye_equity)
        .fillna(0.0)
        .reset_index()
    )
    return out


def _btc_share_quarterly(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute BTC share of notional per calendar quarter.

    Parameters
    ----------
    trades:
        Full trade history from ``load_trades()``.

    Returns
    -------
    pd.DataFrame
        Columns: quarter (YYYY-Qn), total_notional_xbt, btc_notional_xbt,
        btc_share_pct.
    """
    ts_col = "timestamp" if "timestamp" in trades.columns else "transactTime"
    if "execType" in trades.columns:
        fills = trades[trades["execType"] == "Trade"].copy()
    else:
        fills = trades.copy()

    fills["quarter"] = fills[ts_col].dt.to_period("Q").astype(str)
    fills["home_notional_xbt"] = fills["homeNotional"].fillna(0.0).abs()
    fills["is_btc"] = fills["symbol"].str.upper().str.contains("XBT", na=False)

    agg = fills.groupby("quarter").agg(
        total_notional_xbt = ("home_notional_xbt", "sum"),
        btc_notional_xbt   = ("home_notional_xbt",
                               lambda s: s[fills.loc[s.index, "is_btc"]].sum()),
    ).reset_index()

    agg["btc_share_pct"] = (
        agg["btc_notional_xbt"] / agg["total_notional_xbt"].replace(0, np.nan) * 100.0
    ).round(2)

    return agg


def _symbol_pnl_ranking(wallet: pd.DataFrame) -> pd.DataFrame:
    """Rank symbols by realised PnL using walletHistory.

    Parameters
    ----------
    wallet:
        Wallet history from ``load_wallet_history()``.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, gross_pnl_xbt, fees_xbt, net_pnl_xbt, symbol_class,
        sorted descending by net_pnl_xbt.
    """
    w = wallet.copy()
    w["amount_xbt"] = w["amount"].fillna(0).astype(float) * SAT_TO_XBT
    w["fee_xbt"]    = w["fee"].fillna(0).astype(float) * SAT_TO_XBT

    # Only rows with a symbol-like address/text may carry symbol info.
    # walletSummary has a "symbol" column; walletHistory may not.
    sym_col = None
    for c in ("symbol", "address", "text"):
        if c in w.columns:
            sym_col = c
            break

    if sym_col is None:
        logger.warning("No symbol column found in walletHistory; skipping symbol PnL ranking")
        return pd.DataFrame()

    pnl_rows = w[w["transactType"] == "RealisedPNL"]
    fee_rows  = w[w["transactType"] == "ExchangeFee"]

    pnl_by_sym = (
        pnl_rows.groupby(sym_col)["amount_xbt"].sum().rename("gross_pnl_xbt")
    )
    fee_by_sym = (
        fee_rows.groupby(sym_col)["fee_xbt"].sum().abs().rename("fees_xbt")
    )

    ranking = (
        pnl_by_sym.to_frame()
        .join(fee_by_sym, how="outer")
        .fillna(0.0)
        .reset_index()
        .rename(columns={sym_col: "symbol"})
    )
    ranking["net_pnl_xbt"] = ranking["gross_pnl_xbt"] - ranking["fees_xbt"]
    ranking["symbol_class"] = ranking["symbol"].apply(classify_symbol)
    ranking = ranking.sort_values("net_pnl_xbt", ascending=False).reset_index(drop=True)
    return ranking


def main() -> None:
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    logger.info("=== 02 — Yearly Summary ===")

    trades = load_trades()
    wallet = load_wallet_history()

    # Need orders for order count per year
    try:
        from lib.io import load_orders
        orders = load_orders()
        order_stats = _yearly_order_stats(orders)
    except Exception as exc:
        logger.warning("Could not load orders: %s", exc)
        order_stats = pd.DataFrame(columns=["year", "n_orders"])

    trade_stats  = _yearly_trade_stats(trades)
    wallet_stats = _yearly_wallet_stats(wallet)

    # Merge everything on year
    summary = (
        pd.DataFrame({"year": sorted(
            set(trade_stats["year"].tolist())
            | set(wallet_stats["year"].tolist())
        )})
        .merge(order_stats, on="year", how="left")
        .merge(trade_stats, on="year", how="left")
        .merge(wallet_stats, on="year", how="left")
        .fillna(0.0)
    )
    summary["net_pnl_xbt"] = summary["gross_pnl_xbt"] - summary["fees_xbt"]
    summary["year"] = summary["year"].astype(int)

    col_order = [
        "year", "n_orders", "n_fills", "total_notional_xbt",
        "gross_pnl_xbt", "fees_xbt", "net_pnl_xbt",
        "deposits_xbt", "withdrawals_xbt", "year_end_equity_xbt",
        "btc_share_pct",
    ]
    summary = summary[[c for c in col_order if c in summary.columns]]

    out1 = outputs / "yearly_summary.csv"
    summary.to_csv(out1, index=False, float_format="%.6f")
    logger.info("Wrote %s", out1)

    # BTC share quarterly
    btc_q = _btc_share_quarterly(trades)
    out2 = outputs / "btc_share_quarterly.csv"
    btc_q.to_csv(out2, index=False, float_format="%.2f")
    logger.info("Wrote %s", out2)

    # Symbol PnL ranking
    sym_pnl = _symbol_pnl_ranking(wallet)
    if not sym_pnl.empty:
        out3 = outputs / "symbol_pnl_ranking.csv"
        sym_pnl.to_csv(out3, index=False, float_format="%.6f")
        logger.info("Wrote %s", out3)
    else:
        logger.warning("symbol_pnl_ranking.csv not written (no data)")

    logger.info("02 complete.")


if __name__ == "__main__":
    main()
