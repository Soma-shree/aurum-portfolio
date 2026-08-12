"""
dashboard.py
------------
Streamlit dashboard for monitoring portfolio performance.

Usage:
    streamlit run dashboard.py

Features:
  - Current portfolio weights (pie chart + bar chart)
  - Historical allocation changes over time
  - Real-time current prices via yfinance
  - Sharpe ratio / return / volatility tracked across runs
  - Agent views and risk flags from each run
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HISTORY_DIR = Path(__file__).parent / "history"
LATEST_REPORT = Path(__file__).parent / "portfolio_report.json"

st.set_page_config(
    page_title="Portfolio Monitor",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)  # cache 5 minutes
def load_latest_report() -> dict | None:
    if LATEST_REPORT.exists():
        with open(LATEST_REPORT) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_history() -> list[dict]:
    """Load all historical reports sorted by timestamp."""
    reports = []
    if HISTORY_DIR.exists():
        for p in sorted(HISTORY_DIR.glob("report_*.json")):
            with open(p) as f:
                reports.append(json.load(f))
    # Also include the latest report if it exists and isn't already in history
    latest = load_latest_report()
    if latest:
        existing_ts = {r["timestamp"] for r in reports}
        if latest["timestamp"] not in existing_ts:
            reports.append(latest)
    return sorted(reports, key=lambda r: r["timestamp"])


@st.cache_data(ttl=60)  # refresh prices every minute
def fetch_current_prices(tickers: list[str]) -> dict:
    """Fetch latest market prices for a list of tickers."""
    prices = {}
    if not tickers:
        return prices
    try:
        data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
        # Normalise to a DataFrame with ticker columns regardless of how many tickers
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            # Single ticker: yfinance returns flat columns like Open/High/Low/Close
            if "Close" in data.columns:
                close = data[["Close"]]
                close.columns = tickers[:1]
            else:
                close = pd.DataFrame()

        for t in tickers:
            if t not in close.columns:
                continue
            series = close[t].dropna()
            if len(series) >= 1:
                prices[t] = {
                    "price": round(float(series.iloc[-1]), 2),
                    "prev":  round(float(series.iloc[-2]), 2) if len(series) >= 2 else None,
                }
    except Exception:
        pass
    return prices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct_change(current: float, prev: float | None) -> str:
    if prev is None or prev == 0:
        return ""
    chg = (current - prev) / prev * 100
    arrow = "▲" if chg >= 0 else "▼"
    color = "green" if chg >= 0 else "red"
    return f"<span style='color:{color}'>{arrow} {abs(chg):.2f}%</span>"


def build_weights_df(report: dict) -> pd.DataFrame:
    weights = report["weights"]
    sector_map = report["agent_output"].get("sector_map", {})
    views = report["agent_output"].get("views", {})
    confidence = report["agent_output"].get("view_confidence", {})
    rows = []
    for ticker, w in weights.items():
        rows.append({
            "Ticker": ticker,
            "Weight (%)": round(w * 100, 2),
            "Sector": sector_map.get(ticker, "—"),
            "LSTM Forecast (%)": round(views.get(ticker, 0) * 100, 1),
            "Confidence": round(confidence.get(ticker, 0), 2),
        })
    return pd.DataFrame(rows).sort_values("Weight (%)", ascending=False)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("📈 Portfolio Optimization Monitor")
st.caption("Real-time monitoring of the hybrid LSTM + Black-Litterman + Markowitz optimizer")

report = load_latest_report()
history = load_history()

if report is None:
    st.warning("No portfolio report found. Run `python portfolio_system.py` or `python scheduler.py --now` first.")
    st.stop()

# ── Header metrics ──────────────────────────────────────────────────────────
ts = datetime.fromisoformat(report["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
st.subheader(f"Latest run: {ts}")

m = report["metrics_bl"]
mh = report["metrics_historical"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Return (BL)",    f"{m['expected_return']:+.2f}%")
col2.metric("Volatility",              f"{m['volatility']:.2f}%")
col3.metric("Sharpe Ratio (BL)",       f"{m['sharpe_ratio']:.3f}")
col4.metric("Sharpe Ratio (Hist.)",    f"{mh['sharpe_ratio']:.3f}")

market_view = report["agent_output"].get("overall_market_view", "neutral").upper()
view_color  = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(market_view, "⚪")
st.markdown(f"**LSTM aggregate market view:** {view_color} {market_view}")

risk_flags = report["agent_output"].get("risk_flags", [])
if risk_flags:
    st.warning("⚠️ Risk flags: " + " · ".join(risk_flags))

skipped = report["agent_output"].get("failures", [])
if skipped:
    st.info("ℹ️ Skipped (insufficient history): " + " · ".join(skipped))

st.divider()

# ── Current weights + prices ─────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Current Allocation")
    weights_df = build_weights_df(report)

    fig_pie = px.pie(
        weights_df,
        names="Ticker",
        values="Weight (%)",
        color="Ticker",
        hole=0.35,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Live Prices")
    tickers = report["tickers"]
    prices = fetch_current_prices(tickers)

    price_rows = []
    for ticker in tickers:
        p = prices.get(ticker, {})
        current = p.get("price")
        prev    = p.get("prev")
        weight  = report["weights"].get(ticker, 0)
        if current:
            chg = (current - prev) / prev * 100 if prev else 0
            price_rows.append({
                "Ticker":     ticker,
                "Price ($)":  current,
                "Day Chg (%)": round(chg, 2),
                "Portfolio Wt (%)": round(weight * 100, 1),
            })
        else:
            price_rows.append({
                "Ticker":     ticker,
                "Price ($)":  "—",
                "Day Chg (%)": "—",
                "Portfolio Wt (%)": round(weight * 100, 1),
            })

    price_df = pd.DataFrame(price_rows)
    st.dataframe(price_df, use_container_width=True, hide_index=True)

st.divider()

# ── LSTM forecasts bar chart ─────────────────────────────────────────────────
st.subheader("LSTM Forecasts vs. Portfolio Weights")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    name="Portfolio Weight (%)",
    x=weights_df["Ticker"],
    y=weights_df["Weight (%)"],
    marker_color="steelblue",
))
fig_bar.add_trace(go.Bar(
    name="LSTM Forecast (%)",
    x=weights_df["Ticker"],
    y=weights_df["LSTM Forecast (%)"],
    marker_color="darkorange",
))
fig_bar.update_layout(
    barmode="group",
    xaxis_title="Ticker",
    yaxis_title="%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=30, b=10),
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Historical performance ───────────────────────────────────────────────────
if len(history) > 1:
    st.subheader("Historical Performance")

    hist_rows = []
    for r in history:
        row = {
            "Timestamp": r["timestamp"],
            "Sharpe (BL)": r["metrics_bl"]["sharpe_ratio"],
            "Return (BL) %": r["metrics_bl"]["expected_return"],
            "Volatility %": r["metrics_bl"]["volatility"],
            "Market View": r["agent_output"].get("overall_market_view", "neutral"),
        }
        for t in r.get("tickers", []):
            row[t] = round(r["weights"].get(t, 0) * 100, 1)
        hist_rows.append(row)

    hist_df = pd.DataFrame(hist_rows)
    hist_df["Timestamp"] = pd.to_datetime(hist_df["Timestamp"])
    hist_df = hist_df.sort_values("Timestamp")

    # Metrics over time
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(
        x=hist_df["Timestamp"], y=hist_df["Sharpe (BL)"],
        name="Sharpe Ratio", mode="lines+markers", line=dict(color="steelblue"),
    ))
    fig_metrics.add_trace(go.Scatter(
        x=hist_df["Timestamp"], y=hist_df["Return (BL) %"],
        name="Expected Return (%)", mode="lines+markers", line=dict(color="green"),
        yaxis="y2",
    ))
    fig_metrics.update_layout(
        title="Portfolio Metrics Over Time",
        xaxis_title="Date",
        yaxis=dict(title="Sharpe Ratio"),
        yaxis2=dict(title="Return (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Weight allocation drift
    all_tickers = report["tickers"]
    weight_cols = [t for t in all_tickers if t in hist_df.columns]
    if weight_cols:
        fig_alloc = go.Figure()
        for t in weight_cols:
            fig_alloc.add_trace(go.Scatter(
                x=hist_df["Timestamp"], y=hist_df[t],
                name=t, mode="lines+markers", stackgroup="one",
            ))
        fig_alloc.update_layout(
            title="Allocation Drift Over Time (%)",
            xaxis_title="Date",
            yaxis_title="Weight (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    # History table
    with st.expander("Full history table"):
        display_cols = ["Timestamp", "Sharpe (BL)", "Return (BL) %", "Volatility %", "Market View"]
        st.dataframe(hist_df[display_cols], use_container_width=True, hide_index=True)

else:
    st.info("Historical charts will appear after multiple runs. Run the scheduler to accumulate data.")

st.divider()

# ── LSTM vs baseline model evaluation ────────────────────────────────────────
st.subheader("Model Evaluation: LSTM vs. Baselines")
st.caption("Chronological 80/20 train-test split · RMSE and MAPE on next-day log-returns")

eval_metrics = report["agent_output"].get("eval_metrics", {})
if eval_metrics:
    eval_rows = []
    for ticker, ev in eval_metrics.items():
        for model_name, key in [("LSTM", "lstm"), ("Naive", "naive"), ("Moving Avg (5d)", "moving_average")]:
            m = ev[key]
            eval_rows.append({
                "Ticker": ticker,
                "Model": model_name,
                "RMSE": m["rmse"],
                "MAPE (%)": m["mape"],
                "Directional Accuracy (%)": m["directional_accuracy"],
            })
    eval_df = pd.DataFrame(eval_rows)
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    fig_dir_acc = go.Figure()
    for model_name, key in [("LSTM", "lstm"), ("Naive", "naive"), ("Moving Avg (5d)", "moving_average")]:
        fig_dir_acc.add_trace(go.Bar(
            name=model_name,
            x=list(eval_metrics.keys()),
            y=[eval_metrics[t][key]["directional_accuracy"] for t in eval_metrics],
        ))
    fig_dir_acc.add_hline(y=50, line_dash="dash", line_color="gray",
                           annotation_text="coin flip", annotation_position="bottom right")
    fig_dir_acc.update_layout(
        barmode="group",
        title="Directional Accuracy by Model",
        yaxis_title="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_dir_acc, use_container_width=True)
else:
    st.info("No model evaluation data in this report — re-run with the LSTM pipeline enabled.")

st.divider()

# ── Backtest: long/flat signal vs buy-and-hold ───────────────────────────────
st.subheader("Backtest: LSTM Long/Flat Signal vs. Buy-and-Hold")
st.caption("Signal: long when the LSTM forecasts a positive next-day return, flat (cash) otherwise")

portfolio_backtest = report.get("portfolio_backtest", {})
backtests = report["agent_output"].get("backtests", {})

if portfolio_backtest:
    bt_dates = pd.to_datetime(portfolio_backtest["dates"])
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=bt_dates, y=portfolio_backtest["strategy_equity"],
        name="LSTM Long/Flat", mode="lines", line=dict(color="steelblue"),
    ))
    fig_bt.add_trace(go.Scatter(
        x=bt_dates, y=portfolio_backtest["buy_and_hold_equity"],
        name="Buy-and-Hold", mode="lines", line=dict(color="darkorange"),
    ))
    fig_bt.update_layout(
        title="Growth of $1 — Equal-Weight Blend (Test Period)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    bt_col1, bt_col2 = st.columns(2)
    bt_col1.metric("Strategy Total Return", f"{portfolio_backtest['strategy_total_return_pct']:+.2f}%")
    bt_col2.metric("Buy-and-Hold Total Return", f"{portfolio_backtest['buy_and_hold_total_return_pct']:+.2f}%")

if backtests:
    bt_rows = []
    for ticker, bt in backtests.items():
        bt_rows.append({
            "Ticker": ticker,
            "Strategy Return (%)": bt["strategy_total_return_pct"],
            "Buy-and-Hold Return (%)": bt["buy_and_hold_total_return_pct"],
            "Strategy Sharpe": bt["strategy_sharpe"],
            "Buy-and-Hold Sharpe": bt["buy_and_hold_sharpe"],
            "Strategy Max DD (%)": bt["strategy_max_drawdown_pct"],
            "Days Long (%)": bt["pct_days_long"],
        })
    bt_df = pd.DataFrame(bt_rows).sort_values("Strategy Return (%)", ascending=False)
    with st.expander("Per-ticker backtest detail"):
        st.dataframe(bt_df, use_container_width=True, hide_index=True)
elif not portfolio_backtest:
    st.info("No backtest data in this report — re-run with the LSTM pipeline enabled.")

st.divider()

# ── Allocation detail table ──────────────────────────────────────────────────
st.subheader("Allocation Detail")
st.dataframe(weights_df, use_container_width=True, hide_index=True)

# ── Refresh button ────────────────────────────────────────────────────────────
if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

st.caption("Prices refresh every 60 seconds · Reports refresh every 5 minutes · Click Refresh to force update")
