# FinSentinel — Market Intelligence Dashboard

Financial Sentiment & Market Dynamics Research Platform
DSN-278 · VIT Bhopal University · 2026

---

## Quick Start

```bash
# 1. Install dependencies
pip install streamlit plotly scipy pandas numpy

# 2. Run the dashboard
streamlit run app.py
```

Opens at: http://localhost:8501

---

## Pages

| Page | What it shows |
|------|--------------|
| 🏠 Overview | Live metrics, asset prices, key findings summary |
| 📊 Sentiment & Correlation | FinBERT+VADER scores vs price, lag correlation, scatter |
| 🌡️ Market Fragility (MFI) | MFI components, VIX validation, fragility timeline |
| ⚡ Shock Propagation | Decay curves, half-life, regime-conditioned shock response |
| 🔬 Regime Analysis | GMSI regimes, Wasserstein distance, vol distributions |
| 📖 Methodology | Full pipeline diagram, NLP formulas, statistical methods |

---

## Data Note

Currently uses realistic synthetic data (GARCH-simulated returns, AR(1) GMSI).
To connect real data: replace the `generate_data()` function in `app.py`
with reads from your `data/processed/master_daily.csv`.

---

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io
3. Select repo → `dashboard/app.py`
4. Deploy (free, public URL)

---

## Stack
- Streamlit — UI framework
- Plotly — interactive charts
- SciPy — correlation statistics
- Pandas / NumPy — data processing
