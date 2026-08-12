"""
technical_features.py
----------------------
Engineers the 7 technical features used as LSTM inputs:
    1. RSI-14
    2. MACD (12/26/9)
    3. SMA-7
    4. SMA-30
    5. Log-returns (1-day)
    6. Rolling volatility (14-day std of log-returns)
    7. Price relative to SMA-30 (mean-reversion signal)

All indicators are computed with plain pandas/numpy (no TA-Lib dependency,
so this installs cleanly on Render / any minimal Python environment).
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "sma_7",
    "sma_30",
    "log_return",
    "rolling_vol_14",
    "price_to_sma30",
]


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index via Wilder's smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # neutral RSI where undefined (flat/insufficient history)


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line (fast EMA - slow EMA), not the signal line."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=1).mean()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


def compute_rolling_volatility(log_returns: pd.Series, window: int = 14) -> pd.Series:
    return log_returns.rolling(window=window, min_periods=2).std()


def build_feature_matrix(prices: pd.Series) -> pd.DataFrame:
    """
    Given a single ticker's close-price series (indexed by date),
    return a DataFrame of the 7 engineered features plus the
    next-day log-return target column ("target_next_return").
    """
    prices = prices.astype(float)
    log_ret = compute_log_returns(prices)
    sma_7 = compute_sma(prices, 7)
    sma_30 = compute_sma(prices, 30)

    df = pd.DataFrame(index=prices.index)
    df["rsi_14"] = compute_rsi(prices, 14)
    df["macd"] = compute_macd(prices)
    df["sma_7"] = sma_7
    df["sma_30"] = sma_30
    df["log_return"] = log_ret
    df["rolling_vol_14"] = compute_rolling_volatility(log_ret, 14)
    df["price_to_sma30"] = prices / sma_30 - 1.0

    # Target: next trading day's log return (what the LSTM learns to predict)
    df["target_next_return"] = log_ret.shift(-1)

    df = df.dropna()
    return df


def build_all_feature_matrices(prices_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run build_feature_matrix for every ticker column in a price DataFrame."""
    out = {}
    for ticker in prices_df.columns:
        out[ticker] = build_feature_matrix(prices_df[ticker])
    return out
