"""
model_evaluation.py
--------------------
Baseline forecasters + evaluation metrics used to benchmark the LSTM:
    - Naive baseline        : predict tomorrow's return = 0 (no-change persistence
                               on returns; equivalent to "tomorrow's price = today's price")
    - Moving-average baseline: predict tomorrow's return = mean of last N returns

Metrics: RMSE, MAPE, directional accuracy — computed on the chronological
test split so the LSTM can be judged against something honest.
"""

import numpy as np
import pandas as pd


def naive_baseline_forecast(returns: pd.Series) -> pd.Series:
    """
    Persistence baseline on returns: predict tomorrow's return equals
    today's realised return (last known value carried forward).
    """
    return returns.shift(1)


def moving_average_baseline_forecast(returns: pd.Series, window: int = 5) -> pd.Series:
    """Predict tomorrow's return as the trailing N-day mean return."""
    return returns.rolling(window=window, min_periods=1).mean().shift(1)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    MAPE on log-returns. Since returns can sit near zero (undefined % error),
    we floor the denominator at `eps` rather than dropping those points,
    which keeps the metric stable without silently discarding data.
    """
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of days where predicted sign(return) matches actual sign(return)."""
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return float(np.mean(true_dir == pred_dir) * 100)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": round(compute_rmse(y_true, y_pred), 6),
        "mape": round(compute_mape(y_true, y_pred), 2),
        "directional_accuracy": round(compute_directional_accuracy(y_true, y_pred), 2),
    }


def evaluate_against_baselines(
    test_dates,
    y_true: np.ndarray,
    lstm_pred: np.ndarray,
    all_returns: pd.Series,
) -> dict:
    """
    Compare LSTM predictions to naive and moving-average baselines
    over the same test window.

    all_returns : full log-return series (train+test) so the baselines
                  have history to look back on at the test boundary.
    """
    naive_full = naive_baseline_forecast(all_returns)
    ma_full = moving_average_baseline_forecast(all_returns, window=5)

    naive_pred = naive_full.reindex(test_dates).fillna(0.0).values
    ma_pred = ma_full.reindex(test_dates).fillna(0.0).values

    return {
        "lstm": evaluate_forecast(y_true, lstm_pred),
        "naive": evaluate_forecast(y_true, naive_pred),
        "moving_average": evaluate_forecast(y_true, ma_pred),
    }
