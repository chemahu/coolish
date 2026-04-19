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
