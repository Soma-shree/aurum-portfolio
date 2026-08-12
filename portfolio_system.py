"""
portfolio_system.py
-------------------
Top-level orchestrator that wires together:
  data_fetcher → llm_agent → classical_optimizer → report

Usage:
    python portfolio_system.py

Or import and call run_portfolio_optimization() directly.
"""

import json
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


from data_fetcher import summarise_portfolio_data
from lstm_forecaster import generate_lstm_views
from classical_optimizer import (
    markowitz_optimize,
    black_litterman,
    apply_risk_constraints,
    portfolio_metrics,
)
from backtester import run_portfolio_backtest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = [
    "AAPL",   # Apple          – Technology
    "MSFT",   # Microsoft      – Technology
    "GOOGL",  # Alphabet       – Communication Services
    "JPM",    # JPMorgan       – Financials
    "JNJ",    # J&J            – Healthcare
    "XOM",    # ExxonMobil     – Energy
    "AMZN",   # Amazon         – Consumer Discretionary
    "BRK-B",  # Berkshire      – Financials
]

CONFIG = {
    "tickers": DEFAULT_TICKERS,
    "period_days": 365,
    "risk_free_rate": 0.04,
    "max_single_weight": 0.25,
    "min_single_weight": 0.02,
    "max_sector_weight": 0.40,
    "optimizer_target": "sharpe",     # "sharpe" | "min_vol" | "max_return"
    "use_black_litterman": True,      # False → pure Markowitz
    "lstm_lookback": 20,              # trading days per input sequence
    "lstm_epochs": 40,
    "verbose": True,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_portfolio_optimization(
    tickers: list[str] = None,
    config: dict = None,
    use_lstm: bool = True,
    api_key: str = None,
) -> dict:
    """
    Full pipeline:
    1. Fetch market data + engineer technical features
    2. Train an LSTM per ticker on those features → next-day return
       forecasts, evaluated vs naive/moving-average baselines on a
       chronological 80/20 split, and backtested long/flat vs buy-and-hold
       (use_lstm=False falls back to the legacy single-pass LLM agent)
    3. Run Black-Litterman → posterior expected returns, blending the
       forecasts with the market-implied equilibrium prior
    4. Run Markowitz MVO → optimal weights
    5. Apply sector/position risk constraints
    6. Compute and return final metrics + report
    """
    cfg = {**CONFIG, **(config or {})}
    tickers = tickers or cfg["tickers"]
    verbose = cfg["verbose"]

    print("\n" + "="*60)
    print("  HYBRID PORTFOLIO OPTIMIZATION SYSTEM")
    print("="*60)

    # ── Step 1: Data ────────────────────────────────────────────────
    print("\n[1/4] Fetching market data...")
    market_data = summarise_portfolio_data(tickers, cfg["period_days"])
    available = market_data["tickers"]
    print(f"      Available tickers: {available}")

    # ── Step 2: Return forecasts ─────────────────────────────────────
    if use_lstm:
        print("\n[2/4] Training per-ticker LSTM forecasters "
              f"(lookback={cfg['lstm_lookback']}, epochs={cfg['lstm_epochs']})...")
        agent_output = generate_lstm_views(
            market_data,
            lookback=cfg["lstm_lookback"],
            epochs=cfg["lstm_epochs"],
            verbose=verbose,
        )
        agent_output["sector_map"] = market_data.get("sector_map", {})
        if agent_output["failures"]:
            print(f"      Skipped tickers (insufficient history): {agent_output['failures']}")
    else:
        print("\n[2/4] Running legacy single-pass LLM agent...")
        from llm_agent import run_agent  # lazy import: not a default dependency
        agent_output = run_agent(market_data, api_key=api_key, verbose=verbose)

    print("\n  Return-forecast views (expected annual return | confidence):")
    for t, ret in agent_output["views"].items():
        conf = agent_output["view_confidence"].get(t, 0.5)
        print(f"    {t:8s}  {ret*100:+.1f}%  conf={conf:.2f}")

    # ── Step 3: Expected Returns ─────────────────────────────────────
    print("\n[3/4] Running optimizer...")

    mu = market_data["mu"]
    cov = market_data["cov"]

    if cfg["use_black_litterman"] and agent_output["views"]:
        print("      Mode: Black-Litterman (blending LSTM views with market prior)")
        mu_final = black_litterman(
            mu_hist=mu,
            cov=cov,
            market_weights=market_data["market_weights"],
            views=agent_output["views"],
            view_confidence=agent_output["view_confidence"],
        )
    else:
        print("      Mode: Pure Markowitz (historical returns)")
        mu_final = mu

    # ── Step 4: MVO + Risk Constraints ──────────────────────────────
    raw_weights = markowitz_optimize(
        mu=mu_final,
        cov=cov,
        target=cfg["optimizer_target"],
        risk_free=cfg["risk_free_rate"],
        max_weight=cfg["max_single_weight"],
        min_weight=cfg["min_single_weight"],
    )

    final_weights = apply_risk_constraints(
        weights=raw_weights,
        sector_map=agent_output.get("sector_map"),
        max_sector_weight=cfg["max_sector_weight"],
        max_single_weight=cfg["max_single_weight"],
        min_single_weight=cfg["min_single_weight"],
    )

    # ── Metrics ─────────────────────────────────────────────────────
    metrics_bl = portfolio_metrics(final_weights, mu_final, cov, cfg["risk_free_rate"])
    metrics_hist = portfolio_metrics(final_weights, mu, cov, cfg["risk_free_rate"])

    # ── Portfolio-level backtest (equal-weight blend of per-ticker signals) ──
    portfolio_backtest = {}
    if use_lstm and agent_output.get("backtests"):
        portfolio_backtest = run_portfolio_backtest(list(agent_output["backtests"].values()))

    # ── Results ─────────────────────────────────────────────────────
    result = {
        "timestamp": datetime.now().isoformat(),
        "tickers": available,
        "weights": final_weights.to_dict(),
        "metrics_bl": metrics_bl,
        "metrics_historical": metrics_hist,
        "agent_output": agent_output,
        "portfolio_backtest": portfolio_backtest,
        "config": cfg,
    }

    _print_report(result)
    return result


def _print_report(result: dict):
    """Pretty-print the final portfolio report."""
    print("\n" + "="*60)
    print("  FINAL PORTFOLIO ALLOCATION")
    print("="*60)

    weights = pd.Series(result["weights"]).sort_values(ascending=False)
    sector_map = result["agent_output"].get("sector_map", {})

    print(f"\n  {'Ticker':<10} {'Weight':>8}  {'Sector'}")
    print(f"  {'-'*10} {'-'*8}  {'-'*20}")
    for ticker, w in weights.items():
        sector = sector_map.get(ticker, "—")
        bar = "█" * int(w * 40)
        print(f"  {ticker:<10} {w*100:7.1f}%  {sector}  {bar}")

    m = result["metrics_bl"]
    print(f"\n  Portfolio metrics (BL-adjusted returns):")
    print(f"    Expected return : {m['expected_return']:+.2f}%")
    print(f"    Volatility      : {m['volatility']:.2f}%")
    print(f"    Sharpe ratio    : {m['sharpe_ratio']:.3f}")

    mh = result["metrics_historical"]
    print(f"\n  Portfolio metrics (historical returns for reference):")
    print(f"    Expected return : {mh['expected_return']:+.2f}%")
    print(f"    Volatility      : {mh['volatility']:.2f}%")
    print(f"    Sharpe ratio    : {mh['sharpe_ratio']:.3f}")

    flags = result["agent_output"].get("risk_flags", [])
    if flags:
        print(f"\n  Risk flags from agent:")
        for f in flags:
            print(f"    - {f}")

    eval_metrics = result["agent_output"].get("eval_metrics", {})
    if eval_metrics:
        print(f"\n  LSTM vs baselines (test-set RMSE | MAPE | directional accuracy):")
        for t, ev in eval_metrics.items():
            l, nv, ma = ev["lstm"], ev["naive"], ev["moving_average"]
            print(f"    {t:8s}  LSTM {l['rmse']:.4f}|{l['mape']:.1f}%|{l['directional_accuracy']:.1f}%"
                  f"   Naive {nv['rmse']:.4f}|{nv['mape']:.1f}%|{nv['directional_accuracy']:.1f}%"
                  f"   MA {ma['rmse']:.4f}|{ma['mape']:.1f}%|{ma['directional_accuracy']:.1f}%")

    pbt = result.get("portfolio_backtest")
    if pbt:
        print(f"\n  Backtest (long/flat signal vs buy-and-hold, equal-weight blend):")
        print(f"    Strategy total return    : {pbt['strategy_total_return_pct']:+.2f}%")
        print(f"    Buy-and-hold total return: {pbt['buy_and_hold_total_return_pct']:+.2f}%")

    print("\n" + "="*60)


def save_report(result: dict, path: str = "portfolio_report.json"):
    """Serialise result to JSON (converts pandas objects)."""
    serialisable = json.loads(
        json.dumps(result, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  Report saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_portfolio_optimization()
    save_report(result)
