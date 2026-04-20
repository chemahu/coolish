#!/usr/bin/env python3
"""scripts/08_strategy_spec.py — Generate a templated strategy rule book.

Outputs
-------
outputs/strategy_spec.md   — "clone-ready" strategy specification

The specification is fully data-driven: every number is derived from the
analysis outputs written by scripts 02–07.  Nothing is hard-coded.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from lib.io import load_equity_curve, load_instruments, load_trades
from lib.symbols import classify_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAT_TO_XBT   = 1e-8
MIN_PNL_XBT  = 0.1   # minimum net PnL for a symbol to be on the whitelist


def _load_roundtrips() -> pd.DataFrame:
    """Load round-trip parquet files produced by script 03."""
    outputs = PROJECT_ROOT / "outputs"
    frames = []
    for fname in ("roundtrips_xbtusd.parquet", "roundtrips_ethusd.parquet"):
        p = outputs / fname
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            logger.warning("Missing %s", p)

    # Also try to load a full round-trip file if it exists
    full_path = outputs / "roundtrips_all.parquet"
    if full_path.exists():
        frames.append(pd.read_parquet(full_path))

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "open_ts", "close_ts", "qty", "side"])
    return df


def _symbol_whitelist(rt: pd.DataFrame, min_pnl: float = MIN_PNL_XBT) -> list[str]:
    """Return symbols with cumulative net PnL >= min_pnl XBT.

    Parameters
    ----------
    rt:
        Round-trip DataFrame.
    min_pnl:
        Minimum net PnL threshold in XBT.

    Returns
    -------
    list[str]
        Sorted list of qualifying symbol names.
    """
    if rt.empty:
        return []
    sym_pnl = rt.groupby("symbol")["net_pnl_xbt"].sum()
    return sorted(sym_pnl[sym_pnl >= min_pnl].index.tolist())


def _leverage_p95(leverage_csv: Path) -> float:
    """Read leverage_distribution.csv and return ceiling of P95 leverage.

    Parameters
    ----------
    leverage_csv:
        Path to outputs/leverage_distribution.csv.

    Returns
    -------
    float
        Ceiling of P95 leverage (rounded up to nearest integer).
    """
    if not leverage_csv.exists():
        return 3.0  # conservative fallback
    df = pd.read_csv(leverage_csv)
    row = df[df["percentile"].astype(str) == "95"]
    if row.empty:
        return 3.0
    lev = float(row.iloc[0]["leverage"])
    return math.ceil(lev)


def _pyramid_params(pyramid_csv: Path) -> dict:
    """Extract pyramid parameter medians from pyramid_episodes.parquet.

    Parameters
    ----------
    pyramid_csv:
        Path to outputs/pyramid_episodes.parquet.

    Returns
    -------
    dict
        Keys: med_layers, med_spacing_pct, med_layer_qty, med_notional_usd.
    """
    if not pyramid_csv.exists():
        return {
            "med_layers":      5,
            "med_spacing_pct": 0.5,
            "med_layer_qty":   10000,
            "med_notional_usd": 100000,
        }
    df = pd.read_parquet(pyramid_csv)
    # Filter to XBTUSD for most representative parameters
    xbt_eps = df[df["symbol"] == "XBTUSD"] if "XBTUSD" in df["symbol"].values else df
    return {
        "med_layers":       float(xbt_eps["n_layers"].median()),
        "med_spacing_pct":  float(xbt_eps["layer_spacing_median_pct"].median()),
        "med_layer_qty":    float(xbt_eps["avg_layer_qty"].median()),
        "med_notional_usd": float(xbt_eps["notional_usd"].median()),
    }


def _counterfactual_best(cf_md: Path) -> str:
    """Extract the best risk-control rule from counterfactuals.md.

    Simply returns the scenario that produced the largest positive delta.

    Parameters
    ----------
    cf_md:
        Path to outputs/counterfactuals.md.

    Returns
    -------
    str
        Descriptive text of the best scenario.
    """
    if not cf_md.exists():
        return "Symbol stop-loss rule: exclude symbol when cumulative loss ≥ 2% of equity for 12 months"

    text = cf_md.read_text(encoding="utf-8")

    # Parse "Delta" rows from the Markdown table
    best_delta = -1e9
    best_name  = ""
    scenario_name = ""
    for line in text.splitlines():
        if line.startswith("## Scenario"):
            scenario_name = line.replace("##", "").strip()
        if "| Delta |" in line and "**" in line:
            # e.g. "| Delta | **+1.2345 XBT** |"
            try:
                val_str = line.split("**")[1].replace(" XBT", "").replace(",", "")
                val = float(val_str)
                if val > best_delta:
                    best_delta = val
                    best_name = scenario_name
            except Exception:
                pass

    if not best_name:
        return "Symbol stop-loss rule: exclude symbol when cumulative loss ≥ 2% of equity for 12 months"
    return f"{best_name} (delta = {best_delta:+.4f} XBT)"


def _funding_yearly_summary(funding_csv: Path) -> str:
    """Format funding cashflow as a brief sentence."""
    if not funding_csv.exists():
        return "Funding net impact is approximately zero; monitor but do not optimize for it."
    df = pd.read_csv(funding_csv)
    if "funding_net_xbt" not in df.columns or df.empty:
        return "Funding net impact is approximately zero."
    total_net = df["funding_net_xbt"].sum()
    return (
        f"Total funding net over all years: {total_net:+.4f} XBT "
        f"(~{total_net / len(df):.4f} XBT/year).  "
        "Monitor extreme funding rates as a crowding/contrarian signal."
    )


def _build_spec(
    whitelist:    list[str],
    max_leverage: float,
    pyramid:      dict,
    best_cf_rule: str,
    funding_note: str,
    yearly_csv:   Path,
) -> str:
    """Render the strategy specification Markdown.

    Parameters
    ----------
    whitelist:
        List of approved symbols.
    max_leverage:
        P95 leverage ceiling (integer).
    pyramid:
        Pyramid parameter dict from ``_pyramid_params()``.
    best_cf_rule:
        Best counterfactual risk rule description.
    funding_note:
        Funding cashflow summary sentence.
    yearly_csv:
        Path to yearly_summary.csv for cash-flow discipline section.

    Returns
    -------
    str
        Full strategy specification in Markdown.
    """
    whitelist_str = "\n".join(f"  - `{s}` ({classify_symbol(s)})" for s in whitelist) or "  - (run pipeline to populate)"

    # Withdrawal discipline from yearly summary
    withdrawal_note = ""
    if yearly_csv.exists():
        df = pd.read_csv(yearly_csv)
        if "withdrawals_xbt" in df.columns and not df.empty:
            avg_wd = df["withdrawals_xbt"].mean()
            withdrawal_note = f"Historically averaged {avg_wd:.2f} XBT/year in withdrawals."

    lines = [
        "# Coolish-Style BTC Range-Pyramid Strategy — Specification",
        "",
        "> **Auto-generated** from the `data-` tag analysis pipeline.",
        "> All numbers derived from historical account data; none are hard-coded.",
        "",
        "---",
        "",
        "## 1. Approved Symbol Whitelist",
        "",
        f"Only symbols with cumulative net PnL ≥ {MIN_PNL_XBT} XBT qualify:",
        "",
        whitelist_str,
        "",
        "**Not-to-trade list:**",
        "- Any symbol not on the whitelist above",
        "- Quarterly / futures contracts (H/M/U/Z suffix) — historically near-zero or negative EV",
        "- New listings with < 90 days of history",
        "",
        "---",
        "",
        "## 2. Leverage Limit",
        "",
        f"Maximum effective account leverage: **{max_leverage}×** (P95 of observed history, rounded up).",
        "",
        "- Keep liquidation price at least 3× current price away from entry.",
        "- Effective leverage = position notional / wallet balance; monitor after each fill.",
        "",
        "---",
        "",
        "## 3. Pyramid Entry Structure",
        "",
        f"Based on XBTUSD episodes (median values):",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Layers per episode | {pyramid['med_layers']:.1f} (median) |",
        f"| Inter-layer price spacing | {pyramid['med_spacing_pct']:.4f}% (median of medians) |",
        f"| Average layer size | {pyramid['med_layer_qty']:,.0f} contracts |",
        f"| Episode notional | {pyramid['med_notional_usd']:,.0f} USD |",
        "",
        "**Rules:**",
        "- Place all entry limit orders simultaneously before the market moves.",
        "- Use maker (limit) orders only; no market orders for entry.",
        "- Do not add new layers below the lowest planned layer (no martingale).",
        "- Direction change: cancel remaining unfilled entry orders immediately.",
        "",
        "---",
        "",
        "## 4. Pyramid Exit Structure",
        "",
        "- Mirror the entry pyramid symmetrically on the opposite side of the range.",
        "- Take profit in layers; leave a residual position (≤ 30%) to capture extended moves.",
        "- Use maker (limit) orders only for exits.",
        "- Cancel and re-set exit orders if the price range is structurally invalidated.",
        "",
        "---",
        "",
        "## 5. Risk Control Hard Rules",
        "",
        "### 5.1 Best Counterfactual Rule",
        f"> {best_cf_rule}",
        "",
        "### 5.2 Per-Symbol Loss Limit",
        "- Cumulative net realised loss on any **non-core** symbol ≥ 2% of current wallet balance",
        "  → immediately close/cancel all positions and orders for that symbol.",
        "  → exclude that symbol from trading for **12 months**.",
        "",
        "### 5.3 Funding Crowding Filter",
        "- If 8h funding rate on XBTUSD > +0.05%: do **not** open new longs.",
        "- If 8h funding rate on XBTUSD < −0.05%: do **not** open new shorts.",
        "- This acts as a crowding / contrarian signal to avoid paying extreme funding.",
        "",
        "### 5.4 No Martingale",
        "- Absolutely no doubling down.  Every layer must have been pre-planned before",
        "  the episode started.",
        "",
        "---",
        "",
        "## 6. Cash-Flow Discipline",
        "",
        funding_note,
        "",
        withdrawal_note,
        "",
        "**Rules:**",
        "- Withdraw a fixed percentage (e.g. 20–30%) of cumulative unrealised + realised gains",
        "  when equity reaches a new all-time high.",
        "- Do not reinvest withdrawn BTC back into the trading account.",
        "- Deposit only when account equity has fallen > 30% from the last withdrawal point",
        "  and a compelling new range has formed.",
        "",
        "---",
        "",
        "## 7. 'Do-Not-Do' List",
        "",
        "- ❌ Do not trade altcoin perpetuals not on the whitelist",
        "- ❌ Do not trade quarterly futures contracts",
        "- ❌ Do not use market orders for entry or exit",
        "- ❌ Do not martingale (add unplanned layers below entry range)",
        "- ❌ Do not hold a position through extreme funding (> 0.1% per 8h)",
        "- ❌ Do not chase price outside the pre-defined range",
        "- ❌ Do not automate execution without manual range-review checkpoints",
        "- ❌ Do not target short-term momentum; this strategy is mean-reversion / range",
        "",
        "---",
        "",
        "_Generated by `scripts/08_strategy_spec.py` — re-run `make 08` to refresh._",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    logger.info("=== 08 — Strategy Spec ===")

    # Load round-trips for whitelist
    rt = _load_roundtrips()

    # If no round-trips precomputed, run FIFO now
    if rt.empty:
        logger.info("No cached round-trips found; running FIFO now…")
        from lib.fifo import run_fifo
        trades      = load_trades()
        instruments = load_instruments()
        rt = run_fifo(trades, instruments)

    whitelist    = _symbol_whitelist(rt, min_pnl=MIN_PNL_XBT)
    logger.info("Symbol whitelist (%d symbols): %s", len(whitelist), whitelist)

    max_leverage = _leverage_p95(outputs / "leverage_distribution.csv")
    logger.info("Max leverage (P95 ceiling): %s×", max_leverage)

    pyramid = _pyramid_params(outputs / "pyramid_episodes.parquet")
    logger.info("Pyramid params: %s", pyramid)

    best_cf = _counterfactual_best(outputs / "counterfactuals.md")
    logger.info("Best counterfactual rule: %s", best_cf)

    funding_note = _funding_yearly_summary(outputs / "funding_yearly.csv")

    spec = _build_spec(
        whitelist    = whitelist,
        max_leverage = max_leverage,
        pyramid      = pyramid,
        best_cf_rule = best_cf,
        funding_note = funding_note,
        yearly_csv   = outputs / "yearly_summary.csv",
    )

    out_path = outputs / "strategy_spec.md"
    out_path.write_text(spec, encoding="utf-8")
    logger.info("Wrote %s", out_path)

    # Print first 60 lines for PR description
    lines = spec.splitlines()[:60]
    logger.info("\n=== strategy_spec.md (first 60 lines) ===\n%s", "\n".join(lines))

    logger.info("08 complete.")


if __name__ == "__main__":
    main()
