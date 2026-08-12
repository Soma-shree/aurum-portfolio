"""
data_fetcher.py
---------------
Fetches price history, computes returns, covariance, and
market-cap weights (used as Black-Litterman equilibrium priors).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


def fetch_price_data(
    tickers: list[str],
    period_days: int = 365,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download adjusted-close prices for a list of tickers.
    Returns a DataFrame with dates as index, tickers as columns.
    """
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today()
    start = end - pd.Timedelta(days=period_days)

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single-ticker or flat DataFrame — normalise to a named column
        if "Close" in raw.columns:
            prices = raw[["Close"]]
            prices.columns = [t for t in tickers if t in raw.columns or True][:1]
        else:
            # yfinance may return the ticker name directly as the column
            close_cols = [c for c in raw.columns if str(c).lower() == "close" or c in tickers]
            prices = raw[close_cols] if close_cols else raw.iloc[:, :1]
            prices.columns = tickers[:len(prices.columns)]

    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    prices = prices.ffill().bfill()
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_statistics(
    returns: pd.DataFrame,
    annualization: int = 252,
) -> dict:
    """
    Compute annualised mean returns, covariance matrix,
    and correlation matrix from daily returns.
    """
    mu = returns.mean() * annualization
    cov = returns.cov() * annualization
    corr = returns.corr()
    return {"mu": mu, "cov": cov, "corr": corr}


def fetch_market_caps(tickers: list[str]) -> pd.Series:
    """
    Fetch market caps; fall back to equal weights on failure.
    Used to compute equilibrium (implied) returns in Black-Litterman.
    """
    caps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            cap = info.get("marketCap") or info.get("totalAssets")
            caps[ticker] = cap if cap else 1.0
        except Exception:
            caps[ticker] = 1.0

    series = pd.Series(caps, dtype=float)
    return series / series.sum()


# Static fallback for the default 8-ticker universe, used when yfinance's
# `.info` sector lookup fails or is rate-limited.
_SECTOR_FALLBACK = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication Services",
    "JPM": "Financials", "JNJ": "Healthcare", "XOM": "Energy",
    "AMZN": "Consumer Discretionary", "BRK-B": "Financials",
}


def fetch_sector_map(tickers: list[str]) -> dict:
    """Best-effort ticker -> sector mapping, used for sector risk caps."""
    sectors = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector")
            sectors[ticker] = sector or _SECTOR_FALLBACK.get(ticker, "Unclassified")
        except Exception:
            sectors[ticker] = _SECTOR_FALLBACK.get(ticker, "Unclassified")
    return sectors


def get_recent_news_headlines(ticker: str, max_items: int = 5) -> list[str]:
    """
    Pull recent news headlines for a ticker via yfinance.
    Returns a list of headline strings (may be empty).
    """
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        headlines = []
        for item in news[:max_items]:
            # yfinance may return title as a top-level key or nested under "content"
            title = item.get("title") or ""
            if not title:
                content = item.get("content") or {}
                title = content.get("title") or content.get("summary") or ""
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def summarise_portfolio_data(
    tickers: list[str],
    period_days: int = 365,
    fetch_news: bool = False,
) -> dict:
    """
    One-stop function that returns everything the forecaster and optimizer need.
    News headlines are skipped by default (only the LLM-agent path uses them);
    pass fetch_news=True if you need that path.
    """
    print(f"  Fetching prices for {tickers} over {period_days} days...")
    prices = fetch_price_data(tickers, period_days)
    available = list(prices.columns)
    if len(available) < len(tickers):
        missing = set(tickers) - set(available)
        print(f"  Warning: dropped tickers with insufficient data: {missing}")

    returns = compute_returns(prices)
    stats = compute_statistics(returns)
    market_weights = fetch_market_caps(available)
    sector_map = fetch_sector_map(available)

    news = {}
    if fetch_news:
        for t in available:
            news[t] = get_recent_news_headlines(t)

    return {
        "tickers": available,
        "prices": prices,
        "returns": returns,
        "mu": stats["mu"],
        "cov": stats["cov"],
        "corr": stats["corr"],
        "sector_map": sector_map,
        "market_weights": market_weights,
        "news": news,
    }
