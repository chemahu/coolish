#!/usr/bin/env python3
"""scripts/05_leverage_drawdown.py — Leverage distribution and drawdown analysis.

Outputs
-------
outputs/leverage_distribution.csv  — historical leverage distribution statistics
outputs/drawdown_top10.csv         — top-10 drawdown episodes by depth
outputs/equity_with_drawdowns.png  — equity curve with drawdown shading
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

from lib.io import load_equity_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAT_TO_XBT = 1e-8


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute the drawdown series from a running peak.

    Parameters
    ----------
    equity:
        Series of equity values (any unit, e.g. XBT).

    Returns
    -------
    pd.Series
        Drawdown as a negative fraction of the running peak
        (e.g. -0.25 means -25% from the peak).
    """
    peak = equity.expanding().max()
    dd   = (equity - peak) / peak.replace(0, np.nan)
    return dd


def _find_drawdown_episodes(
    ts: pd.Series,
    equity: pd.Series,
    dd: pd.Series,
) -> pd.DataFrame:
    """Find distinct drawdown episodes (start → trough → recovery).

    Parameters
    ----------
    ts:
        Timestamp series (tz-aware UTC).
    equity:
        Equity series in XBT.
    dd:
        Drawdown fraction series (negative values).

    Returns
    -------
    pd.DataFrame
        Columns: start_ts, trough_ts, end_ts, peak_xbt, trough_xbt,
        drawdown_pct, duration_days, recovery_days.
    """
    in_dd = dd < 0
    episodes: list[dict] = []
    start_idx: int | None = None

    for i, v in enumerate(in_dd):
        if v and start_idx is None:
            start_idx = i
        elif not v and start_idx is not None:
            segment_dd  = dd.iloc[start_idx:i]
            trough_idx  = segment_dd.idxmin()
            t_i         = segment_dd.index.get_loc(trough_idx)
            episodes.append({
                "start_ts":      ts.iloc[start_idx],
                "trough_ts":     ts.iloc[start_idx + t_i],
                "end_ts":        ts.iloc[i - 1],
                "peak_xbt":      equity.iloc[start_idx],
                "trough_xbt":    equity.iloc[start_idx + t_i],
                "drawdown_pct":  float(segment_dd.min() * 100),
                "duration_days": (ts.iloc[i - 1] - ts.iloc[start_idx]).total_seconds() / 86400,
                "recovery_days": (ts.iloc[i - 1] - ts.iloc[start_idx + t_i]).total_seconds() / 86400,
            })
            start_idx = None

    # Handle ongoing drawdown at end of series
    if start_idx is not None:
        segment_dd  = dd.iloc[start_idx:]
        trough_idx  = segment_dd.idxmin()
        t_i         = segment_dd.index.get_loc(trough_idx)
        episodes.append({
            "start_ts":      ts.iloc[start_idx],
            "trough_ts":     ts.iloc[start_idx + t_i],
            "end_ts":        ts.iloc[-1],
            "peak_xbt":      equity.iloc[start_idx],
            "trough_xbt":    equity.iloc[start_idx + t_i],
            "drawdown_pct":  float(segment_dd.min() * 100),
            "duration_days": (ts.iloc[-1] - ts.iloc[start_idx]).total_seconds() / 86400,
            "recovery_days": None,
        })

    df = pd.DataFrame(episodes)
    if not df.empty:
        df = df.sort_values("drawdown_pct").reset_index(drop=True)
    return df


def _leverage_distribution(equity_df: pd.DataFrame) -> pd.DataFrame:
    """Compute leverage distribution from equity curve.

    Leverage is approximated as ``marginBalance / walletBalance``.
    If that column is unavailable, returns an empty frame.

    Parameters
    ----------
    equity_df:
        Equity curve DataFrame from ``load_equity_curve()``.

    Returns
    -------
    pd.DataFrame
        Percentile table of effective leverage.
    """
    if "marginBalance" not in equity_df.columns or "walletBalance" not in equity_df.columns:
        logger.warning("marginBalance or walletBalance not in equity curve; skipping leverage")
        return pd.DataFrame()

    lev = (
        equity_df["marginBalance"].astype(float)
        / equity_df["walletBalance"].replace(0, np.nan).astype(float)
    ).dropna()

    pcts = [5, 10, 25, 50, 75, 90, 95, 99]
    rows = [{"percentile": p, "leverage": float(np.percentile(lev, p))} for p in pcts]
    rows += [
        {"percentile": "mean", "leverage": float(lev.mean())},
        {"percentile": "max",  "leverage": float(lev.max())},
    ]
    return pd.DataFrame(rows)


def _plot_equity_drawdown(
    ts: pd.Series,
    equity: pd.Series,
    dd: pd.Series,
    top_episodes: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot the equity curve with top drawdown periods shaded.

    Parameters
    ----------
    ts:
        Timestamp series.
    equity:
        Equity in XBT.
    dd:
        Drawdown fraction series.
    top_episodes:
        Top-N drawdown episodes to shade.
    out_path:
        Output PNG path.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(ts, equity, color="steelblue", linewidth=1.0, label="Equity (XBT)")
    ax1.set_ylabel("Equity (XBT)")
    ax1.set_title("Equity Curve")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Shade top drawdown episodes
    for _, ep in top_episodes.iterrows():
        ax1.axvspan(ep["start_ts"], ep["end_ts"], alpha=0.12, color="red")

    ax2.fill_between(ts, dd * 100, 0, where=(dd < 0), color="crimson", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown from Peak")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    logger.info("=== 05 — Leverage & Drawdown ===")

    equity_df = load_equity_curve()

    # Resolve equity column
    equity_col = None
    for c in ("equityXBT", "equity", "walletBalance"):
        if c in equity_df.columns:
            equity_col = c
            break

    if equity_col is None:
        logger.error("No equity column found in equity curve; exiting")
        sys.exit(1)

    ts_col = "timestamp" if "timestamp" in equity_df.columns else "transactTime"

    equity_df = equity_df.dropna(subset=[ts_col, equity_col]).copy()
    equity_df = equity_df.sort_values(ts_col).reset_index(drop=True)

    equity = equity_df[equity_col].astype(float)
    if equity_col == "walletBalance":
        equity = equity * SAT_TO_XBT

    ts = equity_df[ts_col]

    # ── Leverage distribution ─────────────────────────────────────────────────
    lev_dist = _leverage_distribution(equity_df)
    if not lev_dist.empty:
        lev_dist.to_csv(outputs / "leverage_distribution.csv", index=False, float_format="%.4f")
        logger.info("Wrote leverage_distribution.csv")
    else:
        # Write empty file so downstream scripts don't fail
        pd.DataFrame(columns=["percentile", "leverage"]).to_csv(
            outputs / "leverage_distribution.csv", index=False
        )

    # ── Drawdown analysis ─────────────────────────────────────────────────────
    dd = _drawdown_series(equity)
    dd = dd.reset_index(drop=True)

    episodes = _find_drawdown_episodes(ts.reset_index(drop=True), equity.reset_index(drop=True), dd)
    top10    = episodes.head(10)
    top10.to_csv(outputs / "drawdown_top10.csv", index=False, float_format="%.4f")
    logger.info("Wrote drawdown_top10.csv  (%d episodes total)", len(episodes))

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_equity_drawdown(ts, equity, dd, top10, outputs / "equity_with_drawdowns.png")

    logger.info("05 complete.")


if __name__ == "__main__":
    main()
