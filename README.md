📈 Sentiment-Aware Market Forecasting System


BTC & NIFTY50 | AI × Finance × Quant Analytics
A full-stack machine learning system that integrates financial news sentiment with market data to forecast next-day market volatility, designed with both research rigor and industry-grade ML systems in mind.


🚀 Project Motivation

Financial markets are increasingly driven by information flow — news, narratives, and collective sentiment.
This project explores how textual sentiment signals interact with market dynamics, and whether they can improve volatility forecasting, a core problem in:
Quantitative trading
Risk management
Portfolio construction
Derivatives pricing
The goal is not just prediction, but interpretability, robustness, and real-world usability.


🧠 Key Objectives

•Extract financial sentiment from large-scale news data
•Align sentiment signals with market returns & volatility
•Build sentiment-aware volatility forecasting models
•Evaluate statistical & economic significance
•Deploy a live, end-to-end ML pipeline with a public dashboard


🗂️ Project Architecture
|
|
├── data/
│   ├── raw/                # Raw price & news data
│   ├── processed/          # Cleaned & aligned datasets
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_time_alignment.ipynb
│   ├── 05_correlation_analysis.ipynb
│   ├── 06_volatility_forecasting.ipynb
│   └── 07_model_evaluation.ipynb
│
├── scripts/                # Data download & automation scripts
├── api/                    # FastAPI inference service (planned)
├── dashboard/              # Web dashboard (planned)
├── models/                 # Saved trained models
├── README.md
└── requirements.txt


📊 Data Sources

Market Data
Bitcoin (BTC) — ~8 years of daily OHLCV data
NIFTY50 — ~12 years of daily OHLCV data

Derived features:
Log returns
Rolling volatility (5, 22, 60 days)
Absolute & squared returns

Text & Sentiment Data
GDELT Global News Database (2023–2024)
Financial-domain filtering applied

Sentiment models:
FinBERT (financial transformer)
VADER (rule-based baseline)

ℹ️ Sentiment data is intentionally shorter-term.

Academic justification: Sentiment effects are regime-dependent and most informative in recent market microstructures.


🔍 Feature Engineering

Key engineered signals include:
Rolling historical volatility
Sentiment momentum & smoothing
Volatility-weighted sentiment interaction
Lagged sentiment features (t+1, t+3, t+5)
These features form the bridge between NLP outputs and financial time series modeling.


📐 Modeling Approach

Prediction Target
Next-day log-volatility, not price direction
More stable
Quantitatively meaningful
Widely used in risk management
Models Implemented
Baselines
Historical mean volatility
EWMA volatility
GARCH (planned)
Machine Learning
Random Forest
XGBoost
Deep Learning
Temporal CNN (local temporal patterns)
LSTM (long-term dependencies)


📈 Evaluation Strategy

Models are evaluated using:
MAE / RMSE on volatility forecasts
Lag-wise sentiment impact analysis
Stability across market regimes

Comparative performance:
with sentiment vs without sentiment
The emphasis is on interpretability and robustness, not just headline accuracy.


🧪 Research Track

This project is structured to be research-ready, targeting platforms such as:
SSRN
Springer special issues (textual analysis in finance)

Planned analysis includes:
Sentiment–volatility causality tests
Regime-specific behavior
Limitations & failure modes
🌐 Live System & Dashboard (In Progress)

Planned features:
Real-time news ingestion
Live sentiment scoring
Volatility forecasting API

Interactive dashboard with:
Sentiment indices
Live forecasts
Methodology explanation
Demo & contact page


🛠️ Tech Stack

Python
Pandas, NumPy
Scikit-learn
TensorFlow / Keras
HuggingFace Transformers
FastAPI (planned)
Plotly / Streamlit / React (dashboard)


📌 Current Status

✅ Data pipeline complete
✅ Sentiment analysis implemented
✅ Time alignment & correlation analysis
✅ Forecasting model stabilization (ongoing)
🚧 Live system & dashboard (planned)


🎯 Why This Project Matters

This is not a toy notebook.
It demonstrates:
Quantitative reasoning
ML system design
NLP × Finance integration
Research-grade thinking
Production awareness


👤 Author

Vaibhav Tiwari
B.Tech CSE (AI & ML), VIT Bhopal University 
📧 Email: vaibhavtiwari159@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/vt004
💻 GitHub: https://github.com/VT69