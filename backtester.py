"""
backtester.py
-------------
Backtests a long/flat trading signal derived from LSTM return predictions
against a buy-and-hold benchmark, on the chronological test window.

Signal rule: go long (weight 1) when the LSTM predicts a positive next-day
return, go flat (weight 0, i.e. cash) when it predicts non-positive.
"""

import numpy as np
import pandas as pd


def generate_long_flat_signal(predicted_returns: pd.Series) -> pd.Series:
    """1 = long, 0 = flat, based on sign of predicted next-day return."""
    return (predicted_returns > 0).astype(int)


def backtest_long_flat(actual_returns: pd.Series, signal: pd.Series) -> pd.Series:
    """
    Cumulative growth of $1 following the long/flat signal.
    Signal for day t is applied to the realised return on day t
    (the signal was generated from a same-day prediction of that return).
    """
    strategy_returns = actual_returns * signal
    equity = np.exp(strategy_returns.cumsum())
    return equity


def backtest_buy_and_hold(actual_returns: pd.Series) -> pd.Series:
    """Cumulative growth of $1 holding the asset the entire test period."""
    return np.exp(actual_returns.cumsum())


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min() * 100)


def run_backtest(ticker: str, actual_returns: pd.Series, predicted_returns: pd.Series) -> dict:
    """
    Full backtest for one ticker over its test window.
    Returns equity curves (as lists, JSON-friendly) plus summary stats.
    """
    signal = generate_long_flat_signal(predicted_returns)
    strategy_equity = backtest_long_flat(actual_returns, signal)
    bh_equity = backtest_buy_and_hold(actual_returns)

    strategy_total_return = float((strategy_equity.iloc[-1] - 1) * 100)
    bh_total_return = float((bh_equity.iloc[-1] - 1) * 100)

    strategy_daily = actual_returns * signal
    ann_factor = np.sqrt(252)
    strategy_sharpe = float(
        (strategy_daily.mean() / (strategy_daily.std() + 1e-9)) * ann_factor
    )
    bh_sharpe = float(
        (actual_returns.mean() / (actual_returns.std() + 1e-9)) * ann_factor
    )

    return {
        "ticker": ticker,
        "dates": [d.strftime("%Y-%m-%d") for d in strategy_equity.index],
        "strategy_equity": strategy_equity.round(4).tolist(),
        "buy_and_hold_equity": bh_equity.round(4).tolist(),
        "strategy_total_return_pct": round(strategy_total_return, 2),
        "buy_and_hold_total_return_pct": round(bh_total_return, 2),
        "strategy_sharpe": round(strategy_sharpe, 3),
        "buy_and_hold_sharpe": round(bh_sharpe, 3),
        "strategy_max_drawdown_pct": round(max_drawdown(strategy_equity), 2),
        "buy_and_hold_max_drawdown_pct": round(max_drawdown(bh_equity), 2),
        "pct_days_long": round(float(signal.mean() * 100), 1),
    }


def run_portfolio_backtest(per_ticker_backtests: list[dict]) -> dict:
    """
    Equal-weight blend of the per-ticker long/flat strategies vs an
    equal-weight buy-and-hold benchmark, aligned on shared dates.
    """
    if not per_ticker_backtests:
        return {}

    strat_frames, bh_frames = [], []
    for bt in per_ticker_backtests:
        idx = pd.to_datetime(bt["dates"])
        strat_frames.append(pd.Series(bt["strategy_equity"], index=idx, name=bt["ticker"]))
        bh_frames.append(pd.Series(bt["buy_and_hold_equity"], index=idx, name=bt["ticker"]))

    strat_df = pd.concat(strat_frames, axis=1).dropna()
    bh_df = pd.concat(bh_frames, axis=1).dropna()

    portfolio_strategy = strat_df.mean(axis=1)
    portfolio_bh = bh_df.mean(axis=1)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in portfolio_strategy.index],
        "strategy_equity": portfolio_strategy.round(4).tolist(),
        "buy_and_hold_equity": portfolio_bh.round(4).tolist(),
        "strategy_total_return_pct": round(float((portfolio_strategy.iloc[-1] - 1) * 100), 2),
        "buy_and_hold_total_return_pct": round(float((portfolio_bh.iloc[-1] - 1) * 100), 2),
    }
