# FinSentinel — Market Intelligence System

**Financial Sentiment & Market Dynamics Research**  
BTC · NIFTY 50 · GMSI · MFI · Shock Propagation

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-FinSentinel-00d4ff?style=flat-square&logo=streamlit)](https://financial-sentiment-market-analysis.streamlit.app/)
[![SSRN](https://img.shields.io/badge/Paper%201-In%20Progress-f59e0b?style=flat-square)](https://ssrn.com)
[![Python](https://img.shields.io/badge/Python-3.11-3b82f6?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=flat-square)](LICENSE)

---

## 🔗 Live Dashboard

**[financial-sentiment-market-analysis.streamlit.app](https://financial-sentiment-market-analysis.streamlit.app/)**

Real-time analysis of global stress, market fragility, and shock propagation across BTC and NIFTY 50.

---

## Overview

This project investigates **why and how markets move** — not price prediction.

The core research question: *How does global stress interact with market fragility and positioning to produce volatility?*

Built on three original contributions:

| Contribution | Description |
|---|---|
| **GMSI** | Global Market Stress Index — pure exogenous composite from GDELT events, FinBERT/VADER sentiment, and attention signals. Zero price-derived inputs. |
| **MFI** | Market Fragility Index — composite of volatility persistence (AC₁), vol-of-vol (CoV), and tail risk frequency. Interpretable, no-lookahead. |
| **Shock Propagation** | Forward volatility decay analysis after top-5% return events, conditioned on GMSI regime. Approximate half-life estimation. |

---

## Key Findings

### Finding 1 — The Complacency Effect *(Paper 1)*
Low GMSI (calm stress signal) predicts **higher** forward volatility, not lower.

```
E[7d Vol | GMSI Quintile]
Q1 (Low Stress):  BTC = 0.0325   NIFTY = 0.0105   ← Highest
Q3 (Medium):      BTC = 0.0278   NIFTY = 0.0080
Q5 (High Stress): BTC = 0.0294   NIFTY = 0.0076   ← Lowest
```

Mechanism: low measured stress → investor complacency → under-hedged positioning → larger shock impact when any event eventually arrives.

### Finding 2 — Statistically Significant via Placebo Test *(Paper 1)*
500-permutation placebo test: real Spearman correlation (BTC: −0.084, NIFTY: −0.058) falls in the bottom 2–5% of the null distribution. The relationship is not by chance.

### Finding 3 — The AC1 Paradox *(Paper 2)*
Volatility persistence (AC₁ of |returns|) **decreases** as stress increases:

```
NIFTY Vol Persistence by Regime:
  Low Stress:  AC1 = 0.148   ← Shocks linger longest
  Medium:      AC1 = 0.117
  High Stress: AC1 = 0.083   ← Shocks decay fastest
```

Markets in high-stress regimes are alert and mean-revert quickly. Markets in low-stress regimes are complacent — shocks find no prepared hedges and persist.

### Finding 4 — BTC Shock Secondary Wave *(Paper 2)*
BTC forward volatility peaks at **t+3** after a shock, not t+1. NIFTY peaks immediately at t+1 and decays monotonically. BTC's delayed peak reflects retail investor lag — narrative accumulates before trading execution.

---

## Dashboard Pages

| Page | Content |
|---|---|
| **Overview** | Key findings, asset prices, regime timeline |
| **GMSI & Conditional Vol** | Real conditional expectation charts by GMSI quintile |
| **Placebo & Robustness** | 500-permutation null distribution vs real correlations |
| **Market Fragility (MFI)** | MFI time series 2016–2024, component decomposition |
| **Shock Propagation** | Forward vol decay curves, regime-conditioned shock response |
| **Regime Analysis** | Volatility distributions, Wasserstein distances, AC1 paradox |
| **Methodology** | Full pipeline diagram, GMSI construction, statistical methods |

---

## Project Architecture

```
financial-sentiment-market-analysis/
│
├── dashboard/                    # Live Streamlit dashboard
│   ├── app.py                    # Main dashboard (1200 lines)
│   ├── requirements.txt          # Dashboard-only dependencies
│   ├── runtime.txt               # Python 3.11 pin
│   └── assets/                   # Real research figures
│       ├── cond_exp_BTC.png
│       ├── cond_exp_NIFTY.png
│       ├── fig1_mfi_btc.png
│       ├── fig3_regime_shock_nifty.png
│       ├── fig4_regime_stats_nifty.png
│       ├── fig5_mfi_components_btc.png
│       ├── fig6_shock_decay_comparison.png
│       ├── gmsi_sanity_checks.png
│       ├── placebo_test_BTC.png
│       └── placebo_test_NIFTY.png
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_time_alignment.ipynb
│   ├── 05_correlation_analysis.ipynb
│   ├── 06_volatility_forecasting.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_market_dynamics_analysis.ipynb  # MFI + Shock Propagation
│
├── data_pipeline/
│   ├── fetch_all.py              # Master pipeline runner
│   ├── sources/
│   │   ├── fetch_yfinance.py
│   │   ├── fetch_fred.py
│   │   ├── fetch_gdelt.py
│   │   ├── fetch_trends.py
│   │   ├── fetch_alphavantage.py
│   │   ├── fetch_reddit.py
│   │   ├── fetch_newsapi.py
│   │   └── fetch_quandl.py
│   └── utils/
│       ├── cache.py
│       ├── manifest.py
│       └── logger.py
│
├── data/
│   ├── raw/                      # Source CSVs (gitignored)
│   └── processed/                # Aligned master dataset (gitignored)
│
├── models/                       # Saved model artifacts
├── scripts/                      # Automation scripts
├── requirements.txt              # Full project dependencies
└── README.md
```

---

## Data Sources

| Source | Data | Notes |
|---|---|---|
| **yfinance** | BTC-USD, ^NSEI OHLCV | 2016–present, log returns, rolling vol |
| **FRED** | VIX, credit spreads, yield curve, TED spread | Macro fragility validation |
| **GDELT GKG** | Event counts, avg_tone, negative_share, conflict themes | BigQuery, 2015–present |
| **NewsAPI** | Financial headlines | FinBERT + VADER scoring pipeline |
| **Google Trends** | Search volume for 14 financial terms | Attention/fear proxy |
| **Alpha Vantage** | GDP, Fed Funds Rate, CPI | Macro gap-fill |

**GMSI construction uses zero price-derived inputs.** Previous version had mechanical coupling (correlation ~0.9 with volatility). The corrected exogenous GMSI uses only event intensity, sentiment, and attention signals.

---

## Methodology

### GMSI — Global Market Stress Index

```
GMSI = w₁·EventIntensity + w₂·NegativeShare + w₃·GoldsteinInv
     + w₄·FinBERT + w₅·VADER + w₆·SentimentSurprise + w₇·Attention

Weights = PCA loading scores on first principal component.
Normalization = expanding min-max (zero look-ahead).
```

### MFI — Market Fragility Index

```
MFI = (A_norm + B_norm + C_norm) / 3

A: AC₁(|rₜ|) rolling 30d    — Volatility Persistence
B: CoV(σ₇ᵈ) rolling 30d     — Vol-of-Vol
C: P(|rₜ| > 2σ₃₀) rolling 30d — Tail Risk Frequency

Each component: expanding min-max normalization. No look-ahead.
```

### Statistical Validation
- **Conditional expectation** E[Vol | GMSI quintile] — non-parametric, no distributional assumptions
- **Spearman rank correlation** — robust to fat tails
- **Placebo permutation test** (500 shuffles) — empirical null distribution
- **Wasserstein distance** between regime volatility distributions

---

## Research Papers (In Progress)

### Paper 1 — The Complacency Effect
*"When Calm Breeds Risk: Asymmetric Volatility Response to Global Stress Signals in Cryptocurrency and Equity Markets"*

**Status:** Analysis complete → Writing  
**Target:** Finance Research Letters / SSRN preprint  
**Core finding:** Q1 (Low GMSI) predicts highest forward volatility. Confirmed via placebo test.

### Paper 2 — Market Fragility as a Dynamical Property
*"Market Fragility as a Dynamical Property: Shock Propagation and Regime-Dependent Volatility Persistence"*

**Status:** Analysis complete → Writing  
**Target:** Quantitative Finance / Physica A  
**Core finding:** MFI construction, shock half-life estimation, AC1 paradox across regimes.

---

## Run Locally

```bash
# Clone
git clone https://github.com/VT69/financial-sentiment-market-analysis.git
cd financial-sentiment-market-analysis

# Dashboard only (no heavy dependencies)
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py

# Full pipeline
pip install -r requirements.txt
cp .env.example .env        # Add your API keys
python data_pipeline/fetch_all.py --source yfinance
python data_pipeline/fetch_all.py --source fred
```

---

## Environment Setup

```bash
cp .env.example .env
```

Required keys (all free tiers):

| Variable | Source |
|---|---|
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org/register) |
| `ALPHA_VANTAGE_KEY` | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| `QUANDL_API_KEY` | [data.nasdaq.com](https://data.nasdaq.com/sign-up) |
| `REDDIT_CLIENT_ID` | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account (for GDELT BigQuery) |

---

## Tech Stack

**Analysis:** Python · Pandas · NumPy · SciPy · Statsmodels  
**NLP:** HuggingFace Transformers · FinBERT · VADER  
**Visualization:** Plotly · Matplotlib · Seaborn  
**Dashboard:** Streamlit  
**Data:** yfinance · fredapi · google-cloud-bigquery · pytrends  
**Pipeline:** Modular fetch scripts · 24hr local caching · JSON manifest

---

## Current Status

```
✅ Data pipeline (yfinance, FRED, GDELT, NewsAPI, Google Trends)
✅ GMSI constructed — exogenous, validated, leakage-free
✅ MFI built and validated against VIX
✅ Shock propagation analysis complete
✅ Placebo tests run — findings statistically significant
✅ Live dashboard deployed
🔄 Paper 1 — writing in progress
🔄 Paper 2 — writing in progress
⏳ FRED integration (API key pending)
⏳ HMM regime detection (Paper 3)
⏳ Volatility surface analysis (BTC options via Deribit)
```

---

## Author

**Vaibhav Tiwari**  
B.Tech AI & ML, VIT Bhopal University  
📧 [vaibhavtiwari159@gmail.com](mailto:vaibhavtiwari159@gmail.com)  
🔗 [linkedin.com/in/vt004](https://www.linkedin.com/in/vt004)  
💻 [github.com/VT69](https://github.com/VT69)

---

*This project is part of an ongoing research program in quantitative market dynamics. The dashboard and pipeline are designed to be modular and extensible for future research phases.*
