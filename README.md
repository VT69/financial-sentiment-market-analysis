# Financial Sentiment Analysis for Stock Market Prediction

## Overview
This project explores the relationship between financial market sentiment derived from news and social media text and short-horizon equity price movements. The objective is to evaluate whether sentiment signals provide incremental predictive value over price-based features.

The system is designed as an end-to-end NLP-driven analytics pipeline and is being developed with future deployment as a web-based sentiment analysis platform in mind.

---

## Problem Statement
Financial markets are influenced by qualitative information such as news and investor sentiment. However, sentiment signals are often noisy, non-stationary, and regime-dependent.  
This project investigates:
- Whether sentiment shifts lead short-term market movements
- The robustness and limitations of sentiment-based prediction

---

## Methodology
- Collected financial news and social media text data
- Performed text preprocessing, tokenization, and embedding generation
- Extracted sentiment signals using transformer-based models (BERT, FinBERT)
- Aligned sentiment scores with equity price time series across multiple lag horizons
- Evaluated predictive relevance using regression-based analysis and rolling-window evaluation

---

## Key Findings (In Progress)
- Observed weak but statistically meaningful lead–lag correlations between sentiment shifts and short-horizon returns
- Sentiment signals exhibited regime dependence and signal decay under certain market conditions

---

## Tech Stack
- Python
- Pandas, NumPy
- Hugging Face Transformers
- Scikit-learn
- Matplotlib / Seaborn

---

## Project Status
🚧 Actively under development  
Planned additions:
- Web-based dashboard for sentiment visualization
- Expanded asset universe
- Improved regime-aware modeling

---

## Disclaimer
This project is for educational and research purposes only and does not constitute financial advice.
