# NVDA AI Trading Market Intelligence

An AI-powered market intelligence dashboard for **NVIDIA (NVDA)** that merges historical price data, technical indicators, a trained Random Forest classifier, and financial news sentiment into a single, readable market signal.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [System Workflow](#system-workflow)
- [Technical Analysis](#technical-analysis)
- [Machine Learning](#machine-learning)
- [News & Sentiment](#news--sentiment)
- [AI Decision Engine](#ai-decision-engine)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)
- [Author](#author)

---

## Overview

Most retail-facing stock analysis looks at either price action or the news cycle, rarely both at once. This project brings the two together in one pipeline.

The system loads NVDA historical market data, engineers a set of technical features, feeds them to a trained Random Forest classifier to predict the next directional move, independently scores recent NVIDIA-related headlines for sentiment, and reconciles both into a final market signal — all surfaced through an interactive Streamlit dashboard.

**What the system produces:**

| Output | Description |
|---|---|
| Price analysis | Historical NVDA closing-price movement |
| Technical indicators | Engineered market features across trend, momentum, volatility, and volume |
| ML prediction | Directional call — UP or DOWN |
| Class probabilities | Confidence split between both directions |
| Financial news | Latest NVDA-related headline with source and link |
| Sentiment score | Bullish / Neutral / Bearish classification |
| Final signal | Combined, human-readable market verdict |

---

## System Workflow

```text
              NVDA Historical Market Data
                          │
                          ▼
                   Data Processing
                          │
                          ▼
                 Feature Engineering
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
  Technical Indicators              ML Features
           │                             │
           └──────────────┬──────────────┘
                          ▼
                 Random Forest Model
                          │
                          ▼
                 UP / DOWN Prediction
                          │
                          ▼
                Prediction Probability
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
     Financial News ────────►   Sentiment Analysis
           │                             │
           └──────────────┬──────────────┘
                          ▼
                  AI Decision Engine
                          ▼
                 Final Market Signal
                          ▼
                Streamlit Dashboard
```

---

## Technical Analysis

The feature pipeline derives the following columns from raw NVDA market data.

| Feature | Description |
|---|---|
| `Open` / `High` / `Low` / `Close` | Raw OHLC price data |
| `Volume` | Trading volume |
| `Daily Return` | Day-over-day percentage price change |
| `Volatility 20D` | Rolling 20-day volatility |
| `SMA 20` | 20-day Simple Moving Average |
| `SMA 50` | 50-day Simple Moving Average |
| `SMA Trend` | Relationship between the two moving averages |
| `EMA 20` | 20-day Exponential Moving Average |
| `RSI 14` | 14-period Relative Strength Index |
| `MACD` | Moving Average Convergence Divergence |
| `MACD Signal` | MACD signal line |
| `MACD Histogram` | Difference between MACD and its signal line |
| `Volume SMA 20` | 20-day average trading volume |
| `Volume Ratio` | Current volume relative to its 20-day average |

Together these describe trend, momentum, volatility, and participation — the four dimensions the classifier learns from.

---

## Machine Learning

A `RandomForestClassifier` predicts the next directional move and returns probabilities for both outcomes, so confidence is visible rather than hidden behind a binary call.

**Model configuration**

```python
RandomForestClassifier(
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
```

**Sample output**

```text
UP Probability   : 47.47%
DOWN Probability : 52.53%
Prediction       : DOWN
```

The training feature list is persisted separately and reloaded by the prediction pipeline, guaranteeing identical feature ordering between training and inference.

---

## News & Sentiment

The news module retrieves financial headlines and filters for NVIDIA/NVDA-relevant coverage, extracting the article title, publisher, publication date, and link.

```text
Title     : INTC, AMD, MU, NVDA: Chip Stocks Rally As Investors
            Look Past AI Concerns
Source    : Stocktwits
Date      : 18 Sep 2026
```

Each retrieved article is then scored, producing a sentiment classification and a numeric score that feeds the decision engine as an input independent of the price model.

```text
Sentiment : Bearish
Score     : -0.250
```

---

## AI Decision Engine

The engine reconciles the model's directional call with the prevailing news tone into one verdict.

```text
ML Prediction  +  News Sentiment  ──►  Final Market Signal
```

```text
ML Prediction  : DOWN
News Sentiment : Bearish

FINAL SIGNAL   : NO BUY / EXIT
```

> **Note:** The final signal is a model-generated analytical output, not financial advice.

---

## Dashboard

The Streamlit app is organised into the following sections.

| Section | Contents |
|---|---|
| **Market Snapshot** | Current price, ML prediction, up/down probability, news sentiment |
| **AI Decision Engine** | The final combined market signal |
| **Market Overview** | Historical NVDA closing-price chart |
| **Model Probability** | Probability distribution across both predicted directions |
| **Technical Indicators** | Latest values for every engineered feature |
| **Latest NVDA News** | Most recent relevant article with source, date, and link |
| **Market Summary** | News score, latest market data date, 20-day volatility |

**Example dashboard output**

```text
NVDA Market Intelligence
────────────────────────────────────────
NVDA Price        $219.34
ML Prediction     DOWN
Up Probability    47.47%
Down Probability  52.53%
News Sentiment    Bearish

AI DECISION ENGINE
🔴  NO BUY / EXIT
ML: DOWN  •  News: Bearish

Market Summary
News Score        -0.250
Last Market Data  17 Sep 2026
20D Volatility    2.91
```

---

## Project Structure

```text
AI-Trading-Market-Intelligence/
│
├── data/
│   └── NVDA_clean.csv                  # Cleaned historical market data
│
├── notebooks/
│   ├── 01_market_data_exploration.ipynb
│   ├── nvda_features.pkl               # Engineered feature set
│   ├── nvda_final_features.pkl         # Final training feature list
│   ├── nvda_final_model.pkl            # Production model artifact
│   └── nvda_random_forest.pkl          # Baseline Random Forest model
│
├── src/
│   ├── __init__.py
│   ├── analysis.py                     # Full CLI analysis pipeline
│   ├── config.py                       # Paths and configuration
│   ├── decision.py                     # Final signal logic
│   ├── features.py                     # Technical feature engineering
│   ├── news.py                         # News retrieval and filtering
│   ├── predictor.py                    # Model loading and inference
│   └── sentiment.py                    # Sentiment scoring
│
├── tests/
│   ├── __init__.py
│   └── test_features.py
│
├── app.py                              # Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Modules

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit dashboard — renders the snapshot, prediction, probabilities, indicators, news, sentiment, and final signal |
| `src/features.py` | Builds all technical features from raw market data |
| `src/predictor.py` | Loads the model and feature list, prepares the latest row, returns prediction and probabilities |
| `src/news.py` | Retrieves and filters NVIDIA-related financial news |
| `src/sentiment.py` | Computes per-article and overall news sentiment scores |
| `src/decision.py` | Combines model output and sentiment into the final signal |
| `src/analysis.py` | Runs the complete analysis pipeline from the command line |
| `src/config.py` | Central configuration — data, model, and feature file paths |
| `tests/test_features.py` | Unit tests for the feature-engineering layer |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/sujalsingh09/AI-Trading-Market-Intelligence.git
cd AI-Trading-Market-Intelligence

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Launch the dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### Run individual components

```bash
python -m src.predictor     # Market prediction
python -m src.analysis      # Full market analysis
python -m src.news          # News collection
python -m src.sentiment     # News sentiment
```

**Sample prediction output**

```text
========== LOADED MODEL ==========
RandomForestClassifier(
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

======= FINAL NVDA PREDICTION =======
Close Price       : $219.34
Down Probability  : 52.53%
Up Probability    : 47.47%
Prediction        : DOWN
Trading Signal    : NO BUY / EXIT
=====================================
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data analysis | Pandas, NumPy |
| Machine learning | scikit-learn (Random Forest), Joblib |
| Dashboard & visuals | Streamlit, Matplotlib |
| News & data | Requests, financial market and news sources |
| Development | Jupyter Notebook, Git, GitHub |
| Testing | Pytest |

---

## Testing

```bash
pytest
```

---

## Roadmap

- [ ] Real-time market data updates
- [ ] Backtesting module with Sharpe ratio, max drawdown, and hit rate
- [ ] Historical prediction evaluation and model performance tracking
- [ ] Transformer-based sentiment model in place of lexicon scoring
- [ ] More robust news filtering and multi-source aggregation
- [ ] Additional technical indicators
- [ ] Portfolio-level analysis and multi-ticker support
- [ ] Cloud deployment

---

## Disclaimer

This project is built for **educational, research, and demonstration purposes only**.

The machine learning predictions, sentiment scores, and trading signals it produces are algorithmic outputs and must not be treated as financial advice. Past market behaviour does not guarantee future results. Do not trade real capital on the basis of these signals.

---

## Author

**Sujal Singh Jhala**
B.Tech, Electronics Engineering — Madhav Institute of Technology & Science, Gwalior

GitHub: [@sujalsingh09](https://github.com/sujalsingh09)

---

<p align="center">
  <b>NVDA AI Trading Market Intelligence</b><br>
  Machine Learning &nbsp;•&nbsp; Technical Analysis &nbsp;•&nbsp; Financial News Sentiment
</p>
