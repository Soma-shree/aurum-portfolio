# Hybrid Portfolio Optimization System

A Python system that combines **LSTM return forecasts**, **Black-Litterman** view
blending, and **Markowitz mean-variance optimization (MVO)** across a large-cap
stock universe, with an automated daily scheduler and a Streamlit + Plotly
monitoring dashboard.

---

## Architecture

```
yfinance prices  →  Technical Features  →  LSTM (per ticker)  →  Black-Litterman  →  MVO  →  Portfolio
                     RSI-14, MACD,           Next-day return       Blend LSTM         Max      Weights +
                     SMA-7/30, log-returns,  forecast, evaluated   views with         Sharpe   Metrics +
                     rolling volatility,     vs naive/MA           market             + risk   Backtest
                     price/SMA-30            baselines (RMSE,      equilibrium        caps
                                              MAPE, dir. accuracy)
```

A market-close scheduler (`scheduler.py`) re-runs this pipeline daily —
retraining each ticker's LSTM on the latest data — and the Streamlit
dashboard visualizes live prices, allocation drift, Sharpe ratio history,
model accuracy vs baselines, and a long/flat backtest vs buy-and-hold.

## Setup

```bash
pip install -r requirements.txt
```

TensorFlow (CPU) is required for the LSTM forecaster; everything else is
plain numpy/pandas/scipy.

## Run

```bash
# One-off optimization run (prints report + saves portfolio_report.json)
python portfolio_system.py

# Live dashboard
streamlit run dashboard.py

# Scheduler: run once now, or leave running for daily 4:30 PM ET execution
python scheduler.py --now
python scheduler.py
```

## Pipeline detail

1. **Feature engineering** (`technical_features.py`) — 7 features per ticker:
   RSI-14, MACD, SMA-7, SMA-30, 1-day log-return, 14-day rolling volatility,
   and price-to-SMA-30 ratio.
2. **LSTM forecasting** (`lstm_forecaster.py`) — a 2-layer LSTM per ticker
   predicts next-day log-return from a 20-day lookback window of the above
   features. Data is split **chronologically 80/20** (no shuffling) into
   train/test.
3. **Evaluation** (`model_evaluation.py`) — the LSTM is scored against a
   naive persistence baseline and a 5-day moving-average baseline using
   RMSE, MAPE, and directional accuracy on the held-out test set.
4. **Backtest** (`backtester.py`) — a long/flat signal (long when the LSTM
   predicts a positive next-day return, flat otherwise) is backtested
   against buy-and-hold over the same test window, tracking cumulative
   return, Sharpe ratio, and max drawdown.
5. **Black-Litterman** (`classical_optimizer.py`) — LSTM forecasts become
   the "views," with confidence derived from each ticker's test-set
   directional accuracy, blended with the market-implied equilibrium prior.
6. **Markowitz MVO** — maximizes Sharpe (configurable) on the posterior
   returns, subject to per-position and per-sector weight caps.

## Customisation

Edit `CONFIG` in `portfolio_system.py`:

```python
CONFIG = {
    "tickers": ["AAPL", "MSFT", ...],    # your 8-stock universe
    "period_days": 365,                   # price history lookback window
    "optimizer_target": "sharpe",         # sharpe | min_vol | max_return
    "use_black_litterman": True,          # False = pure Markowitz on historical returns
    "max_single_weight": 0.25,            # max allocation per stock
    "max_sector_weight": 0.40,            # max sector concentration
    "risk_free_rate": 0.04,               # annualised risk-free rate
    "lstm_lookback": 20,                  # trading days per input sequence
    "lstm_epochs": 40,                    # training epochs per ticker
}
```

## Files

```
├── portfolio_system.py      # main orchestrator + entry point
├── data_fetcher.py          # price data, returns, sector map via yfinance
├── technical_features.py    # RSI-14, MACD, SMA-7/30, log-returns, rolling volatility
├── lstm_forecaster.py       # per-ticker LSTM training + forecasting + view generation
├── model_evaluation.py      # RMSE / MAPE / directional accuracy vs baselines
├── backtester.py            # long/flat signal backtest vs buy-and-hold
├── classical_optimizer.py   # Markowitz MVO + Black-Litterman + risk constraints
├── dashboard.py             # Streamlit + Plotly monitoring dashboard
├── scheduler.py             # daily market-close automation
├── llm_agent.py             # legacy single-pass LLM view generator (use_lstm=False)
├── at2po_agent.py           # legacy AT²PO tree-expansion agent (research variant)
└── webapp/                  # separate FastAPI + LLM-broker site deployed on Render
```

## Output

The pipeline prints a formatted report and saves `portfolio_report.json` with:
- Final weights per ticker and sector
- Portfolio expected return, volatility, Sharpe ratio (BL-adjusted and historical)
- LSTM forecasts, confidence, and evaluation metrics vs baselines per ticker
- Long/flat backtest results (per ticker and equal-weight portfolio blend)

## Extending the system

- **Add more data sources**: plug additional fetchers into `data_fetcher.py`
- **Swap the forecaster**: pass `use_lstm=False` to `run_portfolio_optimization()`
  to fall back to the legacy LLM-agent view generator
- **Add rebalancing**: `scheduler.py` already re-runs and retrains daily; extend
  it to place trades via a brokerage API
- **Add constraints**: pass custom sector maps or exclusion lists into
  `apply_risk_constraints()`
