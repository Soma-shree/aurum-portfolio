"""
lstm_forecaster.py
-------------------
Trains a per-ticker LSTM on the 7 engineered technical features to predict
next-day log-return, evaluates it on a chronological 80/20 train-test split
against naive and moving-average baselines, and turns the live forecast into
a Black-Litterman "view" (expected return + confidence).

Design notes:
  - One LSTM per ticker (small universe of 8 large-caps — cheap to retrain
    daily via the scheduler, and keeps each model specialised to that
    stock's own volatility regime).
  - Features are min-max scaled using only the training slice's statistics
    (no leakage from the test window).
  - Target is next-day log-return (continuous regression), which both the
    Black-Litterman view and the long/flat backtest signal are derived from.
"""

import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF info/warning spam

from technical_features import FEATURE_COLUMNS, build_feature_matrix
from model_evaluation import evaluate_against_baselines
from backtester import run_backtest


def _get_tf():
    """Lazy import so the rest of the codebase doesn't pay TensorFlow's
    import cost (or require it) unless an LSTM run is actually requested."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.random.set_seed(42)
    return tf


class MinMaxScalerNP:
    """Tiny dependency-free min-max scaler fit on training data only."""

    def fit(self, X: np.ndarray):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_


def _make_sequences(X_scaled: np.ndarray, y: np.ndarray, lookback: int):
    """
    Sliding-window sequences: row j's sequence is the `lookback` feature
    rows ending at j (inclusive); its target is y[j] (next-day return
    from day j to day j+1, per technical_features.build_feature_matrix).
    """
    n = len(X_scaled)
    xs, ys, idxs = [], [], []
    for j in range(lookback - 1, n):
        xs.append(X_scaled[j - lookback + 1: j + 1])
        ys.append(y[j])
        idxs.append(j)
    return np.array(xs), np.array(ys), np.array(idxs)


def build_lstm_model(input_shape: tuple):
    tf = _get_tf()
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_forecast(
    ticker: str,
    prices: pd.Series,
    lookback: int = 20,
    test_frac: float = 0.2,
    epochs: int = 40,
    batch_size: int = 16,
    verbose: bool = False,
) -> dict:
    """
    Full per-ticker pipeline: features -> chronological 80/20 split ->
    train LSTM -> evaluate on test set vs baselines -> backtest long/flat
    vs buy-and-hold -> forecast tomorrow's return for the BL view.

    Returns a JSON-serialisable dict (no model object) plus the model
    under "_model" for callers that want to reuse/inspect it in-process.
    """
    tf = _get_tf()

    df = build_feature_matrix(prices)
    n = len(df)
    min_rows = lookback + 10  # need enough rows for a meaningful split
    if n < min_rows:
        raise ValueError(f"{ticker}: only {n} usable rows after feature engineering, "
                          f"need at least {min_rows} (increase period_days).")

    split_idx = int(n * (1 - test_frac))

    X_raw = df[FEATURE_COLUMNS].values.astype("float32")
    y_raw = df["target_next_return"].values.astype("float32")

    scaler = MinMaxScalerNP().fit(X_raw[:split_idx])
    X_scaled = scaler.transform(X_raw)

    X_seq, y_seq, seq_row_idx = _make_sequences(X_scaled, y_raw, lookback)
    train_mask = seq_row_idx < split_idx
    test_mask = ~train_mask

    if train_mask.sum() < 10 or test_mask.sum() < 5:
        raise ValueError(f"{ticker}: not enough sequences for train/test "
                          f"({train_mask.sum()} train / {test_mask.sum()} test)")

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]
    test_dates = df.index[seq_row_idx[test_mask]]

    model = build_lstm_model(input_shape=(lookback, len(FEATURE_COLUMNS)))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1 if verbose else 0,
    )

    y_pred_test = model.predict(X_test, verbose=0).flatten()

    all_returns = df["log_return"]
    eval_metrics = evaluate_against_baselines(test_dates, y_test, y_pred_test, all_returns)

    actual_series = pd.Series(y_test, index=test_dates, name=ticker)
    pred_series = pd.Series(y_pred_test, index=test_dates, name=ticker)
    backtest_result = run_backtest(ticker, actual_series, pred_series)

    # Live forecast: predict tomorrow's return using the most recent window
    latest_window = X_scaled[-lookback:].reshape(1, lookback, len(FEATURE_COLUMNS))
    next_day_return = float(model.predict(latest_window, verbose=0).flatten()[0])

    return {
        "ticker": ticker,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "next_day_return_forecast": next_day_return,
        "eval_metrics": eval_metrics,
        "backtest": backtest_result,
        "_model": model,
        "_test_dates": test_dates,
        "_y_test": y_test,
        "_y_pred": y_pred_test,
    }


def generate_lstm_views(
    market_data: dict,
    lookback: int = 20,
    epochs: int = 40,
    verbose: bool = False,
) -> dict:
    """
    Runs train_and_forecast for every ticker in market_data and packages
    the results into the shape portfolio_system.py / classical_optimizer.py
    expect: views (expected annual return), view_confidence, plus rich
    diagnostics (per-ticker eval metrics + backtests) for the dashboard.
    """
    prices = market_data["prices"]
    tickers = market_data["tickers"]

    views, view_confidence = {}, {}
    per_ticker_eval, per_ticker_backtest = {}, {}
    skipped, risk_flags = [], []

    for ticker in tickers:
        try:
            result = train_and_forecast(
                ticker, prices[ticker], lookback=lookback, epochs=epochs, verbose=verbose
            )
        except ValueError as e:
            skipped.append(str(e))
            continue

        # Annualise the daily log-return forecast for the BL view (252 trading days).
        # Clipped to +/-60% — a single day's forecast compounded over a year is
        # noisy by construction, and an unclipped outlier can dominate the BL blend.
        annualised = result["next_day_return_forecast"] * 252
        views[ticker] = float(np.clip(annualised, -0.60, 0.60))

        dir_acc = result["eval_metrics"]["lstm"]["directional_accuracy"]
        # Map directional accuracy (50% = coin flip, 100% = perfect) to a
        # confidence in [0.1, 0.9] for the Black-Litterman omega matrix
        view_confidence[ticker] = float(np.clip((dir_acc - 50) / 50, 0.1, 0.9))

        per_ticker_eval[ticker] = result["eval_metrics"]
        per_ticker_backtest[ticker] = result["backtest"]

        # Flag tickers where the LSTM fails to beat the naive persistence
        # baseline on RMSE — a signal the forecast for that name is noise.
        if result["eval_metrics"]["lstm"]["rmse"] > result["eval_metrics"]["naive"]["rmse"]:
            risk_flags.append(f"{ticker}: LSTM RMSE worse than naive baseline this run")

    avg_view = float(np.mean(list(views.values()))) if views else 0.0
    if avg_view > 0.02:
        overall_market_view = "bullish"
    elif avg_view < -0.02:
        overall_market_view = "bearish"
    else:
        overall_market_view = "neutral"

    return {
        "views": views,
        "view_confidence": view_confidence,
        "eval_metrics": per_ticker_eval,
        "backtests": per_ticker_backtest,
        "overall_market_view": overall_market_view,
        "risk_flags": risk_flags,
        "failures": skipped,
    }
