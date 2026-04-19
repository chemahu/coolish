#!/usr/bin/env python3
"""scripts/04_pyramid_fingerprint.py — Pyramid episode analysis.

Outputs
-------
outputs/pyramid_episodes.parquet   — all identified pyramid episodes
outputs/pyramid_stats.md           — summary statistics in Markdown
outputs/maker_taker_share.csv      — maker vs taker breakdown per year
outputs/cancel_rate.csv            — order cancellation rate per year
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from lib.io import load_orders, load_trades
from lib.pyramid import identify_episodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAT_TO_XBT = 1e-8


def _maker_taker_share(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute maker vs taker share of fill volume per year.

    Parameters
    ----------
    trades:
        Trade history from ``load_trades()``.

    Returns
    -------
    pd.DataFrame
        Columns: year, maker_fills, taker_fills, total_fills, maker_pct, taker_pct.
    """
    ts_col = "timestamp" if "timestamp" in trades.columns else "transactTime"
    if "execType" in trades.columns:
        fills = trades[trades["execType"] == "Trade"].copy()
    else:
        fills = trades.copy()

    if "lastLiquidityInd" not in fills.columns:
        logger.warning("lastLiquidityInd column not found; skipping maker/taker analysis")
        return pd.DataFrame()

    fills["year"] = fills[ts_col].dt.year
    fills["is_maker"] = fills["lastLiquidityInd"].str.upper().isin(["ADDEDLIQUIDITY", "M", "MAKER"])
    fills["is_taker"] = ~fills["is_maker"]

    agg = fills.groupby("year").agg(
        maker_fills = ("is_maker", "sum"),
        taker_fills = ("is_taker", "sum"),
        total_fills = ("execID", "count"),
    ).reset_index()

    agg["maker_pct"] = (agg["maker_fills"] / agg["total_fills"] * 100).round(2)
    agg["taker_pct"] = (agg["taker_fills"] / agg["total_fills"] * 100).round(2)
    return agg


def _cancel_rate(orders: pd.DataFrame) -> pd.DataFrame:
    """Compute order cancellation rate per year.

    Parameters
    ----------
    orders:
        Order DataFrame from ``load_orders()``.

    Returns
    -------
    pd.DataFrame
        Columns: year, total_orders, canceled_orders, cancel_rate_pct.
    """
    if "ordStatus" not in orders.columns:
        logger.warning("ordStatus column not found; skipping cancel rate")
        return pd.DataFrame()

    ts_col = "timestamp" if "timestamp" in orders.columns else "transactTime"
    orders = orders.copy()
    orders["year"] = orders[ts_col].dt.year

    agg = orders.groupby("year").agg(
        total_orders    = ("orderID", "count"),
        canceled_orders = ("ordStatus", lambda s: (s == "Canceled").sum()),
    ).reset_index()

    agg["cancel_rate_pct"] = (
        agg["canceled_orders"] / agg["total_orders"] * 100
    ).round(2)
    return agg


def _pyramid_stats_md(episodes: pd.DataFrame) -> str:
    """Generate a Markdown summary of pyramid episode statistics.

    Parameters
    ----------
    episodes:
        Episode DataFrame from ``lib.pyramid.identify_episodes()``.

    Returns
    -------
    str
        Formatted Markdown text.
    """
    if episodes.empty:
        return "# Pyramid Statistics\n\nNo episodes found.\n"

    total_eps = len(episodes)
    avg_layers = episodes["n_layers"].mean()
    med_layers = episodes["n_layers"].median()
    avg_spacing = episodes["layer_spacing_median_pct"].mean()
    med_spacing = episodes["layer_spacing_median_pct"].median()
    avg_notional = episodes["notional_usd"].mean()
    med_notional = episodes["notional_usd"].median()

    layer_dist = (
        episodes["n_layers"]
        .value_counts()
        .sort_index()
        .head(15)
        .to_frame()
        .rename(columns={"n_layers": "count"})
    )

    top_syms = (
        episodes.groupby("symbol")["n_layers"]
        .agg(["count", "mean", "median"])
        .rename(columns={"count": "n_episodes", "mean": "avg_layers", "median": "med_layers"})
        .sort_values("n_episodes", ascending=False)
        .head(10)
        .round(2)
    )

    long_short = episodes.groupby("direction").size().rename("count")

    lines = [
        "# Pyramid Episode Statistics",
        "",
        f"Total episodes: **{total_eps:,}**",
        "",
        "## Layer Counts",
        f"- Average layers per episode: **{avg_layers:.2f}**",
        f"- Median layers per episode: **{med_layers:.1f}**",
        "",
        "## Price Spacing (inter-fill)",
        f"- Average median spacing: **{avg_spacing:.4f}%**",
        f"- Median of medians: **{med_spacing:.4f}%**",
        "",
        "## Episode Notional (USD)",
        f"- Average: **{avg_notional:,.0f}**",
        f"- Median: **{med_notional:,.0f}**",
        "",
        "## Direction Split",
        long_short.to_markdown(),
        "",
        "## Layer Count Distribution (top 15)",
        layer_dist.to_markdown(),
        "",
        "## Top Symbols by Episode Count",
        top_syms.to_markdown(),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    logger.info("=== 04 — Pyramid Fingerprint ===")

    trades = load_trades()

    # Identify episodes (all symbols)
    logger.info("Identifying pyramid episodes…")
    episodes = identify_episodes(trades, gap_minutes=30)

    if not episodes.empty:
        episodes.to_parquet(
            outputs / "pyramid_episodes.parquet", index=False, engine="pyarrow"
        )
        logger.info("Wrote pyramid_episodes.parquet  (%d rows)", len(episodes))
    else:
        logger.warning("No episodes identified")

    # Stats Markdown
    md = _pyramid_stats_md(episodes)
    (outputs / "pyramid_stats.md").write_text(md, encoding="utf-8")
    logger.info("Wrote pyramid_stats.md")

    # Maker / taker share
    mt = _maker_taker_share(trades)
    if not mt.empty:
        mt.to_csv(outputs / "maker_taker_share.csv", index=False, float_format="%.2f")
        logger.info("Wrote maker_taker_share.csv")

    # Cancel rate
    try:
        orders = load_orders()
        cr = _cancel_rate(orders)
        if not cr.empty:
            cr.to_csv(outputs / "cancel_rate.csv", index=False, float_format="%.2f")
            logger.info("Wrote cancel_rate.csv")
    except Exception as exc:
        logger.warning("Could not compute cancel rate: %s", exc)

    logger.info("04 complete.")


if __name__ == "__main__":
    main()
