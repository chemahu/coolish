"""tests/test_reconciliation.py — walletSummary hard-check reconciliation tests.

Verifies that the walletSummary ground-truth numbers match expected values
from the BitMEX account:

  - All-account RealisedPNL (XBt currency) ≈ +93.198 XBT
  - XBTUSD RealisedPNL ≈ +38.348 XBT
  - ETHUSD RealisedPNL ≈ +37.853 XBT

Tolerance: ±0.01 XBT for each assertion.

These tests require the data directory to be populated (make data).
They are skipped if the walletSummary CSV is not available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.io import load_wallet_summary

SAT_TO_XBT = 1e-8
TOLERANCE_XBT = 0.01  # ±0.01 XBT


def _get_wallet_summary():
    """Load walletSummary, skip test if unavailable."""
    try:
        df = load_wallet_summary()
    except FileNotFoundError:
        pytest.skip("walletSummary file not found; run `make data` first")
    if df.empty:
        pytest.skip("walletSummary is empty; run `make data` first")
    return df


def test_total_realised_pnl():
    """All-account XBt RealisedPNL should total ≈ +93.198 XBT."""
    ws = _get_wallet_summary()
    pnl_rows = ws[(ws["transactType"] == "RealisedPNL") & (ws["currency"] == "XBt")]
    if pnl_rows.empty:
        pytest.skip("No XBt RealisedPNL rows in walletSummary")

    total_xbt = pnl_rows["amount"].fillna(0).astype(float).sum() * SAT_TO_XBT
    expected = 93.198
    assert abs(total_xbt - expected) <= TOLERANCE_XBT, (
        f"Total XBt RealisedPNL = {total_xbt:.4f} XBT, "
        f"expected {expected:.3f} ± {TOLERANCE_XBT} XBT"
    )


def test_xbtusd_realised_pnl():
    """XBTUSD RealisedPNL should total ≈ +38.348 XBT."""
    ws = _get_wallet_summary()
    pnl_rows = ws[
        (ws["transactType"] == "RealisedPNL")
        & (ws["currency"] == "XBt")
        & (ws["symbol"] == "XBTUSD")
    ]
    if pnl_rows.empty:
        pytest.skip("No XBTUSD RealisedPNL rows in walletSummary")

    total_xbt = pnl_rows["amount"].fillna(0).astype(float).sum() * SAT_TO_XBT
    expected = 38.348
    assert abs(total_xbt - expected) <= TOLERANCE_XBT, (
        f"XBTUSD RealisedPNL = {total_xbt:.4f} XBT, "
        f"expected {expected:.3f} ± {TOLERANCE_XBT} XBT"
    )


def test_ethusd_realised_pnl():
    """ETHUSD RealisedPNL should total ≈ +37.853 XBT."""
    ws = _get_wallet_summary()
    pnl_rows = ws[
        (ws["transactType"] == "RealisedPNL")
        & (ws["currency"] == "XBt")
        & (ws["symbol"] == "ETHUSD")
    ]
    if pnl_rows.empty:
        pytest.skip("No ETHUSD RealisedPNL rows in walletSummary")

    total_xbt = pnl_rows["amount"].fillna(0).astype(float).sum() * SAT_TO_XBT
    expected = 37.853
    assert abs(total_xbt - expected) <= TOLERANCE_XBT, (
        f"ETHUSD RealisedPNL = {total_xbt:.4f} XBT, "
        f"expected {expected:.3f} ± {TOLERANCE_XBT} XBT"
    )


def test_fifo_basics():
    """Basic FIFO unit test: a Long round-trip on inverse contract gives correct PnL."""
    import numpy as np
    from lib.fifo import run_fifo

    import pandas as pd

    # Simulate a simple inverse round-trip: buy 1000 at 10000, sell 1000 at 11000
    # Expected PnL = 1000 * (1/10000 - 1/11000) = 1000 * 9.09e-6 ≈ 0.00909 XBT
    trades = pd.DataFrame({
        "execID":    ["a1", "a2"],
        "symbol":    ["XBTUSD", "XBTUSD"],
        "side":      ["Buy", "Sell"],
        "lastQty":   [1000, 1000],
        "lastPx":    [10000.0, 11000.0],
        "timestamp": pd.to_datetime(["2021-01-01 00:00:00", "2021-01-01 01:00:00"], utc=True),
        "execComm":  [0, 0],
        "execType":  ["Trade", "Trade"],
    })
    instruments = pd.DataFrame({
        "symbol":    ["XBTUSD"],
        "isInverse": [True],
        "multiplier": [-1.0],
    })

    rt = run_fifo(trades, instruments)
    assert len(rt) == 1
    rt_row = rt.iloc[0]
    assert rt_row["symbol"] == "XBTUSD"
    assert rt_row["side"] == "Long"
    expected_pnl = 1000.0 * (1.0 / 10000.0 - 1.0 / 11000.0)
    assert abs(rt_row["gross_pnl_xbt"] - expected_pnl) < 1e-9


def test_maker_rebate_preserved():
    """Maker rebate (negative execComm) should reduce fees, not inflate them."""
    import pandas as pd
    from lib.fifo import run_fifo

    # execComm = -500 sat (maker rebate = user receives 5e-6 XBT)
    trades = pd.DataFrame({
        "execID":    ["b1", "b2"],
        "symbol":    ["XBTUSD", "XBTUSD"],
        "side":      ["Buy", "Sell"],
        "lastQty":   [100, 100],
        "lastPx":    [50000.0, 50000.0],
        "timestamp": pd.to_datetime(["2021-06-01 00:00:00", "2021-06-01 01:00:00"], utc=True),
        "execComm":  [-500, -500],  # negative = rebate
        "execType":  ["Trade", "Trade"],
    })
    instruments = pd.DataFrame({
        "symbol":    ["XBTUSD"],
        "isInverse": [True],
        "multiplier": [-1.0],
    })

    rt = run_fifo(trades, instruments)
    assert len(rt) == 1
    # fees_xbt should be negative (net rebate collected)
    assert rt.iloc[0]["fees_xbt"] < 0, "Maker rebate should yield negative fees_xbt"
    # net_pnl should be gross + |rebate|
    assert rt.iloc[0]["net_pnl_xbt"] > rt.iloc[0]["gross_pnl_xbt"], (
        "With maker rebate, net PnL should exceed gross PnL"
    )
