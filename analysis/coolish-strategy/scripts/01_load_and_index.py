#!/usr/bin/env python3
"""scripts/01_load_and_index.py — Load all CSV files, print diagnostics, and cache by year.

Outputs
-------
outputs/cache/trades_<year>.parquet   — trade history split by year
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lib.io import (
    load_equity_curve,
    load_instruments,
    load_orders,
    load_trades,
    load_wallet_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _describe(name: str, df: pd.DataFrame) -> None:
    """Print shape, memory usage, and time range for a DataFrame.

    Parameters
    ----------
    name:
        Human-readable name for display.
    df:
        DataFrame to describe.
    """
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    ts_col = next((c for c in ("timestamp", "transactTime") if c in df.columns), None)
    if ts_col:
        t_min = df[ts_col].dropna().min()
        t_max = df[ts_col].dropna().max()
        time_range = f"{t_min}  →  {t_max}"
    else:
        time_range = "N/A"
    logger.info(
        "%-35s  shape=%s  mem=%.1f MB  time=%s",
        name,
        str(df.shape),
        mem_mb,
        time_range,
    )


def main() -> None:
    output_cache = PROJECT_ROOT / "outputs" / "cache"
    output_cache.mkdir(parents=True, exist_ok=True)

    logger.info("=== 01 — Load & Index ===")

    # ── load ──────────────────────────────────────────────────────────────────
    logger.info("Loading tradeHistory…")
    trades = load_trades()
    _describe("tradeHistory", trades)

    logger.info("Loading orders…")
    orders = load_orders()
    _describe("orders", orders)

    logger.info("Loading walletHistory…")
    wallet = load_wallet_history()
    _describe("walletHistory", wallet)

    logger.info("Loading equity curve…")
    equity = load_equity_curve()
    _describe("equity_curve", equity)

    logger.info("Loading instruments…")
    instruments = load_instruments()
    _describe("instruments", instruments)

    # ── split tradeHistory by year → parquet ─────────────────────────────────
    ts_col = "timestamp" if "timestamp" in trades.columns else "transactTime"
    trades["_year"] = trades[ts_col].dt.year

    years = sorted(trades["_year"].dropna().unique().astype(int))
    logger.info("Splitting tradeHistory by year: %s", years)

    for year in years:
        chunk = trades[trades["_year"] == year].drop(columns=["_year"])
        out_path = output_cache / f"trades_{year}.parquet"
        chunk.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            "  Wrote %s  (%d rows, %.1f MB)",
            out_path.name,
            len(chunk),
            out_path.stat().st_size / 1024 / 1024,
        )

    logger.info("01 complete.")


if __name__ == "__main__":
    main()
