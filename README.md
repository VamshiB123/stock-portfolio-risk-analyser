# 📈 Stock Portfolio Risk Analyser

A full-stack tool to analyse a portfolio of stocks using Yahoo Finance market data, a multi-factor allocation model, and plain-English summaries that explain the results for non-experts.

## Features

- **Daily Returns** calculation per stock
- **Annualised Volatility** (252-trading-day basis)
- **Sharpe Ratio** (with configurable risk-free rate)
- **Sortino Ratio** and **Calmar Ratio** for deeper risk context
- **Maximum Drawdown** per stock and for the overall portfolio
- **Cumulative & Annualised Returns**
- **Technical indicators** including moving averages, RSI, MACD, ADX, Bollinger context, momentum, ATR, and volume signals
- **Normalised Price Chart** to compare performance fairly across different share prices
- **Correlation Matrix** to compare how holdings move together
- **Plain-English portfolio and chart summaries** so the output is easier to understand
- **Per-stock explanation cards** with technical, quality, and selection scores
- Configurable **time period** (3M, 6M, 1Y, 2Y, 5Y)
- **Exchange-aware ticker handling** for US, NSE, and BSE symbols
- **Automatic portfolio optimisation** that blends technical strength, quality signals, and risk-adjusted return
- **Screener CSV import** to load Indian stock baskets from exported screens

## Project Structure

```
stock-portfolio-risk-analyser/
├── backend/
│   ├── app.py              # Flask REST API
│   └── requirements.txt
└── frontend/
    └── index.html          # Single-page dashboard (HTML/CSS/JS)
```

## Getting Started

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The API will run at `http://localhost:5001`.

### Frontend

Simply open `frontend/index.html` in a browser (with the backend running).

Or serve it with:

```bash
cd frontend
python -m http.server 8080
```

Then visit `http://localhost:8080`.

## API

### `POST /api/analyse`

**Request body:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "period": "1y",
  "risk_free_rate": 5
}
```

**Response includes:**
- `portfolio_metrics` — overall portfolio stats
- `stock_metrics` — per-stock risk metrics
- `price_history` — normalised price series for charting
- `weights` — model-selected allocation percentages
- `allocation_model` — optimisation method and simulation metadata
- `correlation_matrix` — pairwise return correlations
- `portfolio_story` — beginner-friendly summary of what the model is seeing
- `chart_summary` — plain-English explanation of what the chart means

### `POST /api/import/screener`

Upload a Screener CSV export as `multipart/form-data` with:

- `file` — the CSV file
- `exchange` — optional `AUTO`, `NSE`, or `BSE`
- `limit` — optional row cap between `1` and `100`

**Response includes:**
- `tickers` — extracted Yahoo-compatible symbols such as `RELIANCE.NS`
- `rows_imported` / `rows_skipped` — import summary
- `detected_exchange` — detected exchange from the CSV rows
- `warnings` — notes such as imported weight columns being ignored

## What The Model Uses

The current backend uses:

- Historical **OHLCV** price data from Yahoo Finance
- Trend signals from **SMA/EMA structure**
- Momentum signals from **RSI, MACD, and multi-period returns**
- Risk signals from **volatility, drawdown, ATR, and correlation**
- Participation signals from **volume ratio, OBV behaviour, and ADX**
- Available yfinance company fields such as **sector, market cap, beta, margins, growth, and leverage**

It does not literally use every field available inside yfinance, but it now uses a much broader mix of technical and company-level signals than a simple return/volatility model.

## Risk Metrics Explained

| Metric | Description |
|---|---|
| Annualised Volatility | Std dev of daily returns × √252 |
| Sharpe Ratio | (Ann. Return − Risk-Free Rate) / Ann. Volatility |
| Max Drawdown | Largest peak-to-trough decline |
| Annualised Return | Mean daily return × 252 |

> **Disclaimer**: For educational purposes only. Not financial advice.
