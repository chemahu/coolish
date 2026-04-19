"""lib/io.py — Streaming CSV loaders with explicit dtype schemas.

All timestamp fields are returned as timezone-aware UTC ``pandas.Timestamp``.
Monetary amounts use BitMEX native units (satoshis / XBt) stored as int64 where
possible; callers should divide by 1e8 to obtain XBT.

Loader functions accept a ``data_dir`` path that points to the extracted tag
directory (e.g. ``data/data-2026-04-17/``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

logger = logging.getLogger(__name__)

# ── dtype schemas (pyarrow) ───────────────────────────────────────────────────

_TRADE_HISTORY_SCHEMA = pa.schema([
    pa.field("execID",            pa.string()),
    pa.field("orderID",           pa.string()),
    pa.field("clOrdID",           pa.string()),
    pa.field("clOrdLinkID",       pa.string()),
    pa.field("account",           pa.int64()),
    pa.field("symbol",            pa.string()),
    pa.field("side",              pa.string()),
    pa.field("lastQty",           pa.int64()),
    pa.field("lastPx",            pa.float64()),
    pa.field("lastLiquidityInd",  pa.string()),
    pa.field("orderQty",          pa.int64()),
    pa.field("leavesQty",         pa.int64()),
    pa.field("cumQty",            pa.int64()),
    pa.field("avgPx",             pa.float64()),
    pa.field("commission",        pa.float64()),
    pa.field("tradePublishIndicator", pa.string()),
    pa.field("taxRate",           pa.float64()),
    pa.field("taxAmount",         pa.int64()),
    pa.field("text",              pa.string()),
    pa.field("execType",          pa.string()),
    pa.field("ordType",           pa.string()),
    pa.field("stopPx",            pa.float64()),
    pa.field("pegOffsetValue",    pa.float64()),
    pa.field("pegPriceType",      pa.string()),
    pa.field("currency",          pa.string()),
    pa.field("settlCurrency",     pa.string()),
    pa.field("execComm",          pa.int64()),
    pa.field("homeNotional",      pa.float64()),
    pa.field("foreignNotional",   pa.float64()),
    pa.field("transactTime",      pa.string()),
    pa.field("timestamp",         pa.string()),
])

_ORDER_SCHEMA = pa.schema([
    pa.field("orderID",           pa.string()),
    pa.field("clOrdID",           pa.string()),
    pa.field("clOrdLinkID",       pa.string()),
    pa.field("account",           pa.int64()),
    pa.field("symbol",            pa.string()),
    pa.field("side",              pa.string()),
    pa.field("simpleOrderQty",    pa.float64()),
    pa.field("orderQty",          pa.int64()),
    pa.field("price",             pa.float64()),
    pa.field("displayQty",        pa.float64()),
    pa.field("stopPx",            pa.float64()),
    pa.field("pegOffsetValue",    pa.float64()),
    pa.field("pegPriceType",      pa.string()),
    pa.field("currency",          pa.string()),
    pa.field("settlCurrency",     pa.string()),
    pa.field("ordType",           pa.string()),
    pa.field("timeInForce",       pa.string()),
    pa.field("execInst",          pa.string()),
    pa.field("contingencyType",   pa.string()),
    pa.field("exDestination",     pa.string()),
    pa.field("ordStatus",         pa.string()),
    pa.field("triggered",         pa.string()),
    pa.field("workingIndicator",  pa.bool_()),
    pa.field("ordRejReason",      pa.string()),
    pa.field("simpleLeavesQty",   pa.float64()),
    pa.field("leavesQty",         pa.int64()),
    pa.field("simpleCumQty",      pa.float64()),
    pa.field("cumQty",            pa.int64()),
    pa.field("avgPx",             pa.float64()),
    pa.field("multiLegReportingType", pa.string()),
    pa.field("text",              pa.string()),
    pa.field("transactTime",      pa.string()),
    pa.field("timestamp",         pa.string()),
])

_WALLET_HISTORY_SCHEMA = pa.schema([
    pa.field("transactID",        pa.string()),
    pa.field("account",           pa.int64()),
    pa.field("currency",          pa.string()),
    pa.field("marginBalance",     pa.int64()),
    pa.field("amount",            pa.int64()),
    pa.field("fee",               pa.int64()),
    pa.field("transactType",      pa.string()),
    pa.field("address",           pa.string()),
    pa.field("tx",                pa.string()),
    pa.field("text",              pa.string()),
    pa.field("transactTime",      pa.string()),
    pa.field("walletBalance",     pa.int64()),
    pa.field("timestamp",         pa.string()),
])

_EQUITY_CURVE_SCHEMA = pa.schema([
    pa.field("timestamp",         pa.string()),
    pa.field("walletBalance",     pa.int64()),
    pa.field("marginBalance",     pa.int64()),
    pa.field("realisedPnl",       pa.int64()),
    pa.field("unrealisedPnl",     pa.int64()),
    pa.field("withdrawnXBT",      pa.float64()),
    pa.field("depositedXBT",      pa.float64()),
    pa.field("equity",            pa.float64()),
    pa.field("equityXBT",         pa.float64()),
])

_INSTRUMENT_SCHEMA = pa.schema([
    pa.field("symbol",            pa.string()),
    pa.field("rootSymbol",        pa.string()),
    pa.field("state",             pa.string()),
    pa.field("typ",               pa.string()),
    pa.field("listing",           pa.string()),
    pa.field("front",             pa.string()),
    pa.field("expiry",            pa.string()),
    pa.field("settle",            pa.string()),
    pa.field("listedSettle",      pa.string()),
    pa.field("positionCurrency",  pa.string()),
    pa.field("underlying",        pa.string()),
    pa.field("quoteCurrency",     pa.string()),
    pa.field("underlyingSymbol",  pa.string()),
    pa.field("reference",         pa.string()),
    pa.field("referenceSymbol",   pa.string()),
    pa.field("calcInterval",      pa.string()),
    pa.field("publishInterval",   pa.string()),
    pa.field("publishTime",       pa.string()),
    pa.field("maxOrderQty",       pa.int64()),
    pa.field("maxPrice",          pa.float64()),
    pa.field("lotSize",           pa.int64()),
    pa.field("tickSize",          pa.float64()),
    pa.field("multiplier",        pa.int64()),
    pa.field("settlCurrency",     pa.string()),
    pa.field("underlyingToPositionMultiplier", pa.float64()),
    pa.field("underlyingToSettleMultiplier",   pa.float64()),
    pa.field("quoteToSettleMultiplier",        pa.float64()),
    pa.field("isQuanto",          pa.bool_()),
    pa.field("isInverse",         pa.bool_()),
    pa.field("initMargin",        pa.float64()),
    pa.field("maintMargin",       pa.float64()),
    pa.field("riskLimit",         pa.int64()),
    pa.field("riskStep",          pa.int64()),
    pa.field("limit",             pa.float64()),
    pa.field("capped",            pa.bool_()),
    pa.field("taxed",             pa.bool_()),
    pa.field("deleverage",        pa.bool_()),
    pa.field("makerFee",          pa.float64()),
    pa.field("takerFee",          pa.float64()),
    pa.field("settlementFee",     pa.float64()),
    pa.field("insuranceFee",      pa.float64()),
    pa.field("fundingBaseSymbol", pa.string()),
    pa.field("fundingQuoteSymbol",pa.string()),
    pa.field("fundingPremiumSymbol", pa.string()),
    pa.field("fundingTimestamp",  pa.string()),
    pa.field("fundingInterval",   pa.string()),
    pa.field("fundingRate",       pa.float64()),
    pa.field("indicativeFundingRate", pa.float64()),
    pa.field("rebalanceTimestamp",pa.string()),
    pa.field("rebalanceInterval", pa.string()),
    pa.field("openingTimestamp",  pa.string()),
    pa.field("closingTimestamp",  pa.string()),
    pa.field("sessionInterval",   pa.string()),
    pa.field("prevClosePrice",    pa.float64()),
    pa.field("limitDownPrice",    pa.float64()),
    pa.field("limitUpPrice",      pa.float64()),
    pa.field("bankruptLimitDownPrice", pa.float64()),
    pa.field("bankruptLimitUpPrice",   pa.float64()),
    pa.field("prevTotalVolume",   pa.int64()),
    pa.field("totalVolume",       pa.int64()),
    pa.field("volume",            pa.int64()),
    pa.field("volume24h",         pa.int64()),
    pa.field("prevTotalTurnover", pa.int64()),
    pa.field("totalTurnover",     pa.int64()),
    pa.field("turnover",          pa.int64()),
    pa.field("turnover24h",       pa.int64()),
    pa.field("homeNotional24h",   pa.float64()),
    pa.field("foreignNotional24h",pa.float64()),
    pa.field("prevPrice24h",      pa.float64()),
    pa.field("vwap",              pa.float64()),
    pa.field("highPrice",         pa.float64()),
    pa.field("lowPrice",          pa.float64()),
    pa.field("lastPrice",         pa.float64()),
    pa.field("lastPriceProtected",pa.float64()),
    pa.field("lastTickDirection", pa.string()),
    pa.field("lastChangePcnt",    pa.float64()),
    pa.field("bidPrice",          pa.float64()),
    pa.field("midPrice",          pa.float64()),
    pa.field("askPrice",          pa.float64()),
    pa.field("impactBidPrice",    pa.float64()),
    pa.field("impactMidPrice",    pa.float64()),
    pa.field("impactAskPrice",    pa.float64()),
    pa.field("hasLiquidity",      pa.bool_()),
    pa.field("openInterest",      pa.int64()),
    pa.field("openValue",         pa.int64()),
    pa.field("fairMethod",        pa.string()),
    pa.field("fairBasisRate",     pa.float64()),
    pa.field("fairBasis",         pa.float64()),
    pa.field("fairPrice",         pa.float64()),
    pa.field("markMethod",        pa.string()),
    pa.field("markPrice",         pa.float64()),
    pa.field("indicativeTaxRate", pa.float64()),
    pa.field("indicativeSettlePrice", pa.float64()),
    pa.field("optionUnderlyingPrice", pa.float64()),
    pa.field("settledPriceAdjustmentRate", pa.float64()),
    pa.field("settledPrice",      pa.float64()),
    pa.field("timestamp",         pa.string()),
])


# ── internal helpers ──────────────────────────────────────────────────────────

def _find_data_dir(data_dir: Optional[str | Path]) -> Path:
    """Resolve the data directory, searching for a tag subdirectory if needed.

    Parameters
    ----------
    data_dir:
        Explicit path to the data directory, or ``None`` to auto-discover
        from the ``data/`` folder relative to the project root.

    Returns
    -------
    Path
        Resolved directory that contains the CSV files.
    """
    if data_dir is not None:
        return Path(data_dir)

    # Auto-discover: look for a single subdirectory under <project>/data/
    project_root = Path(__file__).parent.parent
    base = project_root / "data"
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")

    subdirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("data-")]
    if not subdirs:
        raise FileNotFoundError(
            f"No 'data-*' tag directory found under {base}. "
            "Run `make data` first."
        )
    # Use the most recent tag (lexicographic sort works because tags are YYYY-MM-DD)
    return sorted(subdirs)[-1]


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the first available timestamp column to tz-aware UTC.

    Priority: ``timestamp`` > ``transactTime``.  The column is parsed with
    ``utc=True`` so the result is always a tz-aware ``DatetimeTZDtype[UTC]``.

    Parameters
    ----------
    df:
        DataFrame that may contain ``timestamp`` and/or ``transactTime`` columns
        as raw strings.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with the timestamp column converted in-place.
    """
    for col in ("timestamp", "transactTime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            logger.debug("Parsed %s as UTC datetime (%d rows)", col, len(df))
            break
    return df


def _read_csv_lenient(path: Path, schema: pa.Schema) -> pd.DataFrame:
    """Read a CSV file using pyarrow with lenient column coercion.

    Unknown columns are included as strings; schema columns that are absent
    from the file are silently skipped.  This tolerates API version drift.

    Parameters
    ----------
    path:
        Absolute path to the CSV file.
    schema:
        Expected pyarrow schema used for type coercion of known columns.

    Returns
    -------
    pd.DataFrame
        Loaded data with pyarrow-backed dtypes converted to pandas-native types.
    """
    logger.info("Loading %s", path)

    read_opts = pa_csv.ReadOptions(block_size=64 * 1024 * 1024)  # 64 MB chunks
    parse_opts = pa_csv.ParseOptions(invalid_row_handler="skip")

    # Build per-column type map from schema for known columns only
    schema_dict = {field.name: field.type for field in schema}

    convert_opts = pa_csv.ConvertOptions(
        column_types=schema_dict,
        null_values=["", "null", "NULL", "None", "NaN"],
        strings_can_be_null=True,
    )

    table = pa_csv.read_csv(
        str(path),
        read_options=read_opts,
        parse_options=parse_opts,
        convert_options=convert_opts,
    )

    df = table.to_pandas(timestamp_as_object=True, safe=False)
    logger.debug("Loaded %d rows × %d cols from %s", len(df), len(df.columns), path.name)
    return df


# ── public loader functions ───────────────────────────────────────────────────

def load_trades(data_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Load ``api-v1-execution-tradeHistory.csv``.

    Returns ~173 k rows with execution-level data.
    Timestamps are tz-aware UTC; ``execComm`` is in satoshis (XBt).

    Parameters
    ----------
    data_dir:
        Path to the extracted tag directory.  Auto-discovered if ``None``.

    Returns
    -------
    pd.DataFrame
        Trade history sorted by ``timestamp`` ascending.
    """
    path = _find_data_dir(data_dir) / "api-v1-execution-tradeHistory.csv"
    df = _read_csv_lenient(path, _TRADE_HISTORY_SCHEMA)
    df = _parse_timestamps(df)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_orders(data_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Load ``api-v1-order.csv``.

    Returns ~43 k rows with order-level data.
    Timestamps are tz-aware UTC.

    Parameters
    ----------
    data_dir:
        Path to the extracted tag directory.  Auto-discovered if ``None``.

    Returns
    -------
    pd.DataFrame
        Orders sorted by ``timestamp`` ascending.
    """
    path = _find_data_dir(data_dir) / "api-v1-order.csv"
    df = _read_csv_lenient(path, _ORDER_SCHEMA)
    df = _parse_timestamps(df)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_wallet_history(data_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Load ``api-v1-user-walletHistory.csv``.

    Returns ~17 k rows with wallet events (deposits, withdrawals, funding,
    realised PnL, etc.).  ``amount`` and ``walletBalance`` are in satoshis (XBt).

    Parameters
    ----------
    data_dir:
        Path to the extracted tag directory.  Auto-discovered if ``None``.

    Returns
    -------
    pd.DataFrame
        Wallet history sorted by ``transactTime`` ascending.
    """
    path = _find_data_dir(data_dir) / "api-v1-user-walletHistory.csv"
    df = _read_csv_lenient(path, _WALLET_HISTORY_SCHEMA)
    # walletHistory uses transactTime as the primary timestamp
    for col in ("transactTime", "timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    sort_col = "transactTime" if "transactTime" in df.columns else "timestamp"
    if sort_col in df.columns:
        df = df.sort_values(sort_col).reset_index(drop=True)
    return df


def load_equity_curve(data_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Load ``derived-equity-curve.csv``.

    Returns ~17 k rows with daily/periodic equity snapshots.
    ``walletBalance`` is in satoshis (XBt); ``equityXBT`` is in XBT (float64).
    Timestamps are tz-aware UTC.

    Parameters
    ----------
    data_dir:
        Path to the extracted tag directory.  Auto-discovered if ``None``.

    Returns
    -------
    pd.DataFrame
        Equity curve sorted by ``timestamp`` ascending.
    """
    path = _find_data_dir(data_dir) / "derived-equity-curve.csv"
    df = _read_csv_lenient(path, _EQUITY_CURVE_SCHEMA)
    df = _parse_timestamps(df)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_instruments(data_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Load ``api-v1-instrument.all.csv``.

    Returns ~3 k rows with instrument metadata.
    Key columns: ``symbol``, ``isInverse``, ``makerFee``, ``takerFee``,
    ``multiplier`` (used for PnL calculation of inverse contracts).

    Parameters
    ----------
    data_dir:
        Path to the extracted tag directory.  Auto-discovered if ``None``.

    Returns
    -------
    pd.DataFrame
        Instrument reference data (one row per symbol).
    """
    path = _find_data_dir(data_dir) / "api-v1-instrument.all.csv"
    df = _read_csv_lenient(path, _INSTRUMENT_SCHEMA)
    return df
