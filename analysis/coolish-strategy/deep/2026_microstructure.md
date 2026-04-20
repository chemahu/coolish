# Dimension 3: 2026 至今 92% maker 的微观结构分析

## 核心问题

> 2026年的maker%从历史35–50%暴增到92%，这是个突变点。
> 这份分析要回答：**这是策略升级（量化挂单），还是极端人工纪律？**

---

## 1. OrderType Distribution (订单类型分布)

| ordType | Count | % of Fills |
|---------|-------|-----------|
| Limit | 759 | 98.4% |
| Market | 12 | 1.6% |

**Interpretation**: BitMEX labels all resting limit orders as `Limit` — but both  
human-placed and algo-placed limits look the same here. The signal is in the  
*timing* and *volume*, not the type label.

---

## 2. Maker/Taker Breakdown (挂单/吃单分布)

| lastLiquidityInd | Count | % |
|-----------------|-------|---|
| AddedLiquidity | 712 | 92.3% |
| RemovedLiquidity | 59 | 7.7% |

**2026 overall maker%: 92.3%**  

Historical comparison:
- 2020: ~38%  (learning phase, mostly taker)
- 2021: ~52%  (bull market, balanced)
- 2022: ~40%  (bear market panic, more taker)
- 2023: ~47%  (recovering)
- 2024: ~36%  (another taker-heavy year)
- 2025: ~49%  (back to balanced)
- **2026: 92.3%  ← structural break**

This is not a gradual trend — it is a sudden jump indicating a **fundamental  
change in execution methodology**.

---

## 3. Daily Fill Count Distribution (每日成交数量直方图)

| Statistic | Value |
|-----------|-------|
| Total fills (2026 YTD) | 771 |
| Total trading days | 44 |
| Mean daily fills | 17.5 |
| Median daily fills | 10.5 |
| Max daily fills | 185 |
| Min daily fills | 1 |

**Daily fill count detail:**

| Date | Fills |
|------|-------|
| 2026-01-02 | 17 |
| 2026-01-05 | 31 |
| 2026-01-06 | 13 |
| 2026-01-07 | 8 |
| 2026-01-08 | 7 |
| 2026-01-12 | 2 |
| 2026-01-13 | 37 |
| 2026-01-14 | 1 |
| 2026-01-15 | 2 |
| 2026-01-18 | 18 |
| 2026-01-19 | 13 |
| 2026-01-20 | 33 |
| 2026-01-21 | 18 |
| 2026-01-23 | 7 |
| 2026-01-25 | 1 |
| 2026-01-26 | 32 |
| 2026-01-28 | 23 |
| 2026-01-29 | 31 |
| 2026-01-30 | 29 |
| 2026-01-31 | 20 |
| 2026-02-01 | 16 |
| 2026-02-03 | 14 |
| 2026-02-05 | 185 |
| 2026-02-06 | 38 |
| 2026-02-08 | 1 |
| 2026-02-14 | 13 |
| 2026-02-15 | 24 |
| 2026-02-21 | 6 |
| 2026-02-25 | 5 |
| 2026-03-01 | 8 |
| 2026-03-04 | 19 |
| 2026-03-05 | 43 |
| 2026-03-06 | 1 |
| 2026-03-08 | 7 |
| 2026-03-11 | 2 |
| 2026-03-12 | 4 |
| 2026-03-13 | 12 |
| 2026-03-16 | 5 |
| 2026-03-17 | 5 |
| 2026-03-21 | 3 |
| 2026-03-24 | 9 |
| 2026-03-27 | 1 |
| 2026-04-07 | 3 |
| 2026-04-14 | 4 |

**Human trading benchmark**: A disciplined manual trader executing  
4-layer pyramid entries on 2–3 symbols typically generates 8–30 fills/day  
on active days. An automated system can generate 100–1000+ fills/day.

**Observation**: Max daily fills = 185 — this **exceeds the realistic human ceiling** for careful 4-layer trading, suggesting algorithmic assistance at peak activity.

---

## 4. Inter-Fill Time Interval Analysis (成交时间间隔分析)

Algorithmic order systems tend to show more regular timing  
(lower coefficient of variation), while human trading is bursty and irregular.

| Statistic | Seconds | Interpretation |
|-----------|---------|----------------|
| Mean interval | 11463.8s | Average time between consecutive fills |
| Median interval | 0.0s | Typical gap |
| 25th percentile | 0.0s | Fast fills |
| 75th percentile | 0.0s | Slow fills |
| 99th percentile | 282031.1s | Very slow / multi-day gaps |
| Std / Mean (CV) | 5.40 | High = irregular (human/bursty) |

**CV interpretation**:  
- CV < 1.0: Highly regular, strongly algorithmic  
- CV 1.0–2.5: Somewhat regular, possibly semi-algorithmic  
- CV > 2.5: Highly irregular, consistent with human timing  
- **2026 CV = 5.40** → **Irregular pattern — not a simple fixed-interval scheduler.**  However, maker% alone is a strong signal.

---

## 5. Pyramid Episode Layer Analysis (加仓层数分析)

Found **94 pyramid episodes** in 2026 data.

| Statistic | 2026 Value | Historical Median (all years) |
|-----------|-----------|-------------------------------|
| Mean layers per episode | 8.2 | 9.2 |
| Median layers per episode | 4.0 | 4.0 |
| Max layers per episode | 123 | 443 |
| Mean maker% per episode | 0.9% | 0.6% |

**Human baseline**: A manual trader running a 4-layer pyramid strategy  
produces episodes with 1–8 layers (median ~4). An algorithmic system  
can have 20–100+ layers per episode.

**Observation**: 2026 max layers = 123, median = 4  
→ **Exceeds typical human range**. High layer counts suggest automated layering.

Top 10 largest 2026 episodes by layer count:

| Symbol | Start | Layers | Maker% | Notional USD |
|--------|-------|--------|--------|-------------|
| XBTUSD | 2026-02-05 | 123 | 100% | 9022160000 |
| XBTUSD | 2026-02-05 | 57 | 100% | 5942201000 |
| XBTUSD | 2026-03-05 | 38 | 100% | 20530678000 |
| XBTUSD | 2026-01-26 | 29 | 100% | 3525600000 |
| XBTUSD | 2026-01-30 | 29 | 100% | 19562966000 |
| XBTUSD | 2026-01-29 | 28 | 100% | 8501080000 |
| XBTUSD | 2026-02-15 | 24 | 100% | 8396700000 |
| XBTUSD | 2026-01-20 | 23 | 100% | 17681800000 |
| XBTUSD | 2026-02-06 | 19 | 100% | 4279578000 |
| XBTUSD | 2026-01-18 | 18 | 100% | 1885088000 |

---

## 6. Average Fill Notional (平均成交名义价值)

Fill size can distinguish a human (consistently small, risk-managed) from  
an algorithm (often very small micro-fills or very systematic sizing).

| Metric | 2026 | 2021 (Peak Year) |
|--------|------|-----------------|
| Mean |homeNotional| per fill | 0.1144 XBT | 2346.6551 XBT |
| Median |homeNotional| per fill | 0.0193 XBT | — |

Symbol breakdown for 2026 fills:

| Symbol | Fills | % |
|--------|-------|---|
| XBTUSD | 771 | 100.0% |

---

## 7. Final Verdict: 纯人工 / 半算法 / 纯算法

**Analysis window**: 2026-01-02 → 2026-04-14 (103 days)  
**Total fills**: 771  
**Overall maker%**: 92.3%  

### Evidence Summary

| Indicator | Value | Human Threshold | Algo Signal? |
|-----------|-------|----------------|-------------|
| maker_pct | 92.3% | <65% = likely human | ✅ YES |
| Max daily fills | 185 | <50 = clear human | ✅ YES |
| Inter-fill CV | 5.40 | >3 = irregular (human) | ❌ NOT REGULAR |
| Episode max layers | 123 | <15 = human | ✅ YES |
| ordType dominance | Limit (98%) | Limit-only ≠ conclusive | ⚠️ NEUTRAL |

### **⚙️ SEMI-ALGORITHMIC**

The elevated maker% and order patterns suggest automated execution assistance, but volume levels and episode structure are still compatible with a disciplined human trader using advanced order tools.

**Supporting evidence for semi-algorithmic classification**:
- maker_pct of 92.3% is significantly elevated vs history.
  This could be achieved by a very disciplined human using advanced order tools  
  (post-only flags, iceberg orders, reserve orders) OR by a simple algo layer.
- Daily fill counts are higher than a casual human trader but not extreme enough
  to definitively exclude a highly active manual trader.

**Most likely scenario**: A human trader using execution assistance tools  
(automated order routing, smart order types) that enforce limit-only fills.

---

## 结论

> **⚙️ SEMI-ALGORITHMIC**  
>  
> The 92% maker rate in 2026 is the result of a structural change in execution.  
> Based on the combined evidence (maker%, daily volume, timing patterns,  
> episode structure), the most probable explanation is:  
>  
> **The elevated maker% and order patterns suggest automated execution assistance, but volume levels and episode structure are still compatible with a disciplined human trader using advanced order tools.**

---

## 8. Deep Dive: The 2026-02-05 Anomaly (185 fills in One Day)

The single biggest day in the 2026 dataset was 2026-02-05 with 185 fills.  
For context, the median active day has only ~10.5 fills. This is a 17× spike.

### 2026-02-05: 185 fills breakdown

| Metric | Value |
|--------|-------|
| Total fills | 185 |
| Maker fills | 185 |
| Taker fills | 0 |
| Maker% | 100.0% |
| Unique symbols | 1 |
| Symbols | XBTUSD |
| First fill | 04:06:34 UTC |
| Last fill | 20:55:03 UTC |
| Duration | 16.8 hours |
| ordType distribution | Limit: 185 |

**Fill timing on 2026-02-05**:

| Stat | Seconds |
|------|---------|
| Min interval | 0.000s |
| Median interval | 0.001s |
| Mean interval | 328.9s |
| Max interval | 39650s |

**⚡ Sub-second fills detected**

Sub-second fill bursts are a strong indicator of algorithmic execution —  
a human physically cannot place and have confirmed multiple orders in under  
0.1 seconds. These are the 'fingerprints' of automated order execution.

---

## 9. Comparison: 2024 vs 2026 Execution Style

2024 was the second-worst year for maker% (36%) — let's compare it directly to 2026.

| Metric | 2024 | 2026 | Change |
|--------|------|------|--------|
| Total fills | 14,271 | 771 | -13,500 |
| maker_pct | 36.3% | 92.3% | +56.0pp |
| Max daily fills | 648 | 185 | -463 |
| Median daily fills | 26.0 | 10.5 | -15.5 |
| Mean |homeNotional| | 531.7370 XBT | 0.1144 XBT | -531.6226 XBT |

**Order type breakdown:**

| ordType | 2024 % | 2026 % |
|---------|--------|--------|
| Limit | 58.8% | 98.4% |
| Market | 40.0% | 1.6% |
| Stop | 1.1% | 0.0% |

---

## 10. Strategy Implications of the 2026 Execution Change

Regardless of *how* the 92% maker% was achieved (human vs algo), the  
*strategic implication* is the same and highly favorable:

### Fee Economics (BitMEX XBTUSD)

| Scenario | Maker fee | Taker fee | Net fee per round-trip |
|----------|-----------|-----------|----------------------|
| 2024 style (36% maker) | -0.0100% rebate | +0.0750% cost | ~+0.044% (net payer) |
| 2026 style (92% maker) | -0.0100% rebate | +0.0750% cost | ~-0.006% (net receiver) |

A trader who goes from 36% to 92% maker fills flips from **paying** fees to  
**receiving** fee rebates. On a high-volume account, this is worth **tens of  
basis points per round-trip** — compounding to significant alpha over time.

### Execution Alpha Estimate

At 771 fills/year (2026 YTD annualized ×3) and average notional  
0.1144 XBT per fill, the fee improvement over 2024-style execution  
represents approximately:  

```
  Annual fills (est.):      ~2,698
  Avg notional per fill:    0.1144 XBT
  Fee improvement per fill: ~0.05% of notional
  Annual execution alpha:   ~0.154 XBT/year
```

This is a **measurable and recurring edge** from execution alone — independent of  
any directional or timing skill.

---

## Final Verdict (Expanded)

**⚙️ SEMI-ALGORITHMIC**

The elevated maker% and order patterns suggest automated execution assistance, but volume levels and episode structure are still compatible with a disciplined human trader using advanced order tools.

### The Three-Factor Test Results:

1. **maker_pct = 92.3%** — far beyond human discipline ceiling (≤70%). ✅ Algo signal
2. **Max daily fills = 185** — exceeds comfortable human limit (≤50). ✅ Algo signal
3. **Inter-fill CV = 5.40** — high irregularity, NOT a fixed scheduler. ⚠️ Mixed signal
4. **Episode max layers = 123** — exceeds human norm (≤15). ✅ Algo signal

3 out of 4 signals point to algorithmic execution. The irregular CV is explained by  
**a human-directed algo**: the human sets targets and the algo executes — so the  
*decision* timing is human (irregular) but the *fill* timing within each decision  
burst is algorithmic (fast, systematic).

This is consistent with a **hybrid execution model**: human strategy signals + automated order placement.