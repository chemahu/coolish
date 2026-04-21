# coolish-strategy Analysis

This sub-project reverse-engineers the trading strategy of BitMEX trader **Paul Wei
(@coolish)** from the publicly available raw account ledger at
[bwjoke/BTC-Trading-Since-2020](https://github.com/bwjoke/BTC-Trading-Since-2020).

---

## Upstream Dataset — Copyright & Privacy Notice

The upstream repository (`bwjoke/BTC-Trading-Since-2020`) is a **publicly released**
dataset.  The data owner has chosen to publish 43 k+ orders, 173 k+ trade fills,
and 17 k+ wallet events under a stable, daily-tagged release cycle.

- **No personally identifiable information** beyond a pseudonymous BitMEX username is present.
- **No upstream CSV files are committed** to this repository.
- All analysis runs locally from data fetched at `make data` time.
- If the upstream data owner requests removal, delete `data/` and refrain from re-fetching.

---

## Quick Start (minimal reproduction)

```bash
# 0. Clone and enter the sub-project
cd analysis/coolish-strategy

# 1. Create and activate a Python 3.11 venv
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the default (latest) dataset tag
make data

# 4. Run the full pipeline
make all

# 5. Or run a specific step
make 03   # FIFO round-trip only

# 6. Use a specific data tag
make data DATA_TAG=data-2026-04-17
make all  DATA_TAG=data-2026-04-17
```

Expected wall-clock time on a modern laptop: **< 10 minutes**.
Peak memory: **< 4 GB** (pyarrow streaming reads).

---

## Directory Structure

```
analysis/coolish-strategy/
├── README.md              — this file
├── Makefile               — one-command pipeline
├── requirements.txt       — Python dependencies
├── .gitignore             — excludes data/ and outputs/
├── data/                  — gitignored; populated by `make data`
├── outputs/               — gitignored; all analysis results
│   └── cache/             — intermediate parquet files
├── scripts/               — numbered analysis scripts
└── lib/                   — shared Python library
```

---

## Pipeline Steps

| Target | Script | Description |
|--------|--------|-------------|
| `data` | `download_data.sh` | Download & verify tagged release from upstream |
| `01`   | `01_load_and_index.py` | Load all CSVs; cache trades by year as parquet |
| `02`   | `02_yearly_summary.py` | Yearly performance tables; BTC share; symbol PnL ranking |
| `03`   | `03_fifo_roundtrip.py` | FIFO round-trip matching; stats; hold-time histogram |
| `04`   | `04_pyramid_fingerprint.py` | Pyramid episode identification; maker/taker; cancel rate |
| `05`   | `05_leverage_drawdown.py` | Leverage distribution; top-10 drawdown episodes; equity plot |
| `06`   | `06_funding_cashflow.py` | Funding cashflow; withdrawal vs equity high; deposit vs drawdown |
| `07`   | `07_counterfactual.py` | Four counterfactual scenarios |
| `08`   | `08_strategy_spec.py` | Templated strategy rule book |
| `09`   | `09_backtest_replica.py` | Forward-test of the distilled rules (skeleton) |

---

## Outputs Reference

| File | Description |
|------|-------------|
| `outputs/yearly_summary.csv` | Per-year: orders, fills, notional, gross/net PnL, deposits, withdrawals, year-end equity, BTC % |
| `outputs/btc_share_quarterly.csv` | BTC share of trading notional per calendar quarter |
| `outputs/symbol_pnl_ranking.csv` | Per-symbol realised PnL ranked, with symbol class |
| `outputs/roundtrips_xbtusd.parquet` | FIFO round-trips for XBTUSD |
| `outputs/roundtrips_ethusd.parquet` | FIFO round-trips for ETHUSD |
| `outputs/roundtrip_stats.csv` | Aggregate stats per symbol: win-rate, avg hold, profit factor |
| `outputs/hold_time_histogram.png` | Histogram of round-trip holding times |
| `outputs/pyramid_episodes.parquet` | All identified pyramid (scaling) episodes |
| `outputs/pyramid_stats.md` | Markdown summary: avg layers, spacing, notional distribution |
| `outputs/maker_taker_share.csv` | Maker vs taker fill share per year |
| `outputs/cancel_rate.csv` | Order cancellation rate per year |
| `outputs/leverage_distribution.csv` | Historical leverage percentiles |
| `outputs/drawdown_top10.csv` | Top-10 drawdown episodes by depth |
| `outputs/equity_with_drawdowns.png` | Equity curve chart with drawdown shading |
| `outputs/funding_yearly.csv` | Annual funding income/cost net |
| `outputs/withdrawal_vs_high.png` | Scatter: days since equity ATH vs withdrawal size |
| `outputs/deposit_vs_drawdown.png` | Scatter: drawdown depth at deposit vs deposit size |
| `outputs/counterfactuals.md` | Four counterfactual scenario results |
| `outputs/strategy_spec.md` | Auto-generated clone-ready strategy rule book |
| `outputs/replica_equity_curve.csv` | Daily equity from `09_backtest_replica.py` |
| `outputs/replica_trades.csv` | Every simulated fill from the replica backtest |
| `outputs/replica_summary.md` | Replica backtest headline metrics + baseline comparison |
| `outputs/cache/trades_<year>.parquet` | Trade history split by year (intermediate cache) |

---

## Library Modules

| Module | Purpose |
|--------|---------|
| `lib/io.py` | Streaming CSV loaders with explicit pyarrow dtype schemas |
| `lib/symbols.py` | `classify_symbol()` → `xbt_perp / eth_perp / alt_perp / quarterly / ...` |
| `lib/fifo.py` | FIFO round-trip matching; inverse & linear PnL calculation |
| `lib/pyramid.py` | Pyramid episode identification (same-direction fill clustering) |

---

## Validation

Script `03_fifo_roundtrip.py` cross-checks the FIFO aggregate net PnL against
`walletHistory` realised PnL.  If the relative difference exceeds **0.5%**,
the script exits with a non-zero code, stopping `make all`.

---

## Replica Backtest (`make 09`)

`scripts/09_backtest_replica.py` is a forward-looking **skeleton backtester**
that encodes the rules distilled from steps 01–08 + the four counterfactuals:

- Whitelist of XBTUSD/ETHUSD core + LTCUSD/XRPUSD/DOGEUSD secondary;
  quarterly contracts allowed (counterfactual ② → +6.88 XBT); 20 net-negative
  symbols blacklisted (counterfactual ① → +8.15 XBT).
- Maker-only fills at the BitMEX rebate (counterfactual ③ → +7.04 XBT).
- **No stop-loss** — counterfactual ④ proved a 2 %-equity stop costs −32 XBT.
- 4-layer pyramid mean-reversion at ~0.5 % spacing, exit on revert to VWAP.
- Concurrency cap of 8 symbols; funding-rate filter at ±5 bps.

The complete spec lives in the `STRATEGY_SPEC` dict at the top of the script,
making it easy to sweep parameters.

### Usage

```bash
# Built-in synthetic self-test (no external data needed):
make 09

# Real backtest against your own OHLCV bars CSV:
make 09 BARS=/path/to/bars.csv
```

Bars CSV schema (UTF-8, header required):

```
timestamp,symbol,open,high,low,close,funding_rate
2020-05-01T00:00:00Z,XBTUSD,8800.0,8830.0,8770.0,8810.0,0.0001
```

`funding_rate` is optional; consulted only at 8h boundaries.

### Baseline reference

PR #5 reported a real cumulative net PnL of **+141.56 XBT** across 6 years.
Subtracting the unreproducible BMEX_USDT windfall (+19.76 XBT) leaves a
clone-able baseline of **~122 XBT**. A successful replica run on a comparable
data window should land in the same order of magnitude.

### Known skeleton limitations

- All symbols are treated as linear (USD-margined) — XBTUSD's inverse
  contract PnL formula is **not** yet wired in. Refining this is a TODO.
- The mean-reversion entry trigger uses a single-bar open/close compare;
  a real implementation should use a band (Bollinger / Keltner) or
  microstructure signal.
- Funding payments are filtered, not paid; add an 8h accrual loop to
  reflect them in equity.

---

## Units

- All monetary values in **XBT** unless noted otherwise.
- BitMEX API returns balances in **satoshis (XBt)** = 1e-8 XBT; `lib/io.py` keeps
  them as raw int64; callers convert with `× 1e-8`.
- All timestamps are **tz-aware UTC**.

---

## License

Analysis code in this repository is MIT-licensed.
The upstream dataset at `bwjoke/BTC-Trading-Since-2020` is subject to its own license;
consult that repository before redistributing the data.
