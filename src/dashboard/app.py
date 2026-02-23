"""Streamlit monitoring dashboard for the systematic crypto trading system.

Pages:
1. Overview - Positions, equity, PnL
2. Forecasts - Per-rule forecast values and history
3. Risk - Drawdown, leverage, kill switch status
4. System Health - Run history, errors, scheduler status

Run: streamlit run src/dashboard/app.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Data directories
DATA_DIR = Path("data")
LOGS_DIR = Path("logs")
STATE_FILE = DATA_DIR / "system_state.json"
PNL_FILE = DATA_DIR / "pnl_history.json"
RUN_FILE = DATA_DIR / "run_history.json"
TRADE_FILE = LOGS_DIR / "trades.csv"

st.set_page_config(
    page_title="Systematic Crypto Trading",
    page_icon="📊",
    layout="wide",
)


def load_json(path: Path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def load_state() -> dict:
    return load_json(STATE_FILE, default={})


def load_pnl_history() -> list[dict]:
    return load_json(PNL_FILE, default=[])


def load_run_history() -> list[dict]:
    return load_json(RUN_FILE, default=[])


def load_trades() -> pd.DataFrame:
    if TRADE_FILE.exists():
        try:
            df = pd.read_csv(TRADE_FILE)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def main():
    st.title("Systematic Crypto Trading Dashboard")

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Forecasts", "Risk", "Trades", "System Health"],
    )

    state = load_state()

    if not state:
        st.warning("No system state found. Run the trading system first.")
        st.info("Start with: `python scripts/run_paper_trading.py --once`")
        return

    if page == "Overview":
        render_overview(state)
    elif page == "Forecasts":
        render_forecasts(state)
    elif page == "Risk":
        render_risk(state)
    elif page == "Trades":
        render_trades()
    elif page == "System Health":
        render_health()


def render_overview(state: dict):
    """Main overview page: positions, equity, PnL."""
    st.header("Portfolio Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    equity = state.get("equity", state.get("capital", 0))
    capital = state.get("capital", 5000)
    pnl = equity - capital
    pnl_pct = (pnl / capital * 100) if capital > 0 else 0
    peak = state.get("peak_equity", equity)
    drawdown = ((peak - equity) / peak * 100) if peak > 0 else 0

    col1.metric("Equity", f"${equity:,.2f}", f"{pnl_pct:+.2f}%")
    col2.metric("PnL", f"${pnl:+,.2f}")
    col3.metric("Peak Equity", f"${peak:,.2f}")
    col4.metric("Drawdown", f"-{drawdown:.2f}%")

    # Positions table
    st.subheader("Current Positions")
    positions = state.get("positions", {})
    prices = state.get("prices", {})

    if positions:
        pos_rows = []
        for symbol, size in positions.items():
            price = prices.get(symbol, 0)
            notional = abs(size) * price
            side = "LONG" if size > 0 else "SHORT" if size < 0 else "FLAT"
            pos_rows.append({
                "Symbol": symbol.split("/")[0] if "/" in symbol else symbol,
                "Side": side,
                "Size": f"{size:.6f}",
                "Price": f"${price:,.2f}" if price > 0 else "-",
                "Notional": f"${notional:,.2f}" if price > 0 else "-",
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

    # PnL history chart
    pnl_history = load_pnl_history()
    if pnl_history:
        st.subheader("Daily PnL History")
        pnl_df = pd.DataFrame(pnl_history)
        if "date" in pnl_df.columns and "equity" in pnl_df.columns:
            pnl_df["date"] = pd.to_datetime(pnl_df["date"])
            pnl_df = pnl_df.set_index("date")
            st.line_chart(pnl_df["equity"])

    # Last updated
    ts = state.get("timestamp", "")
    if ts:
        st.caption(f"Last updated: {ts}")


def render_forecasts(state: dict):
    """Forecasts page: per-rule breakdown."""
    st.header("Trading Forecasts")

    last_forecasts = state.get("last_forecasts", {})

    if not last_forecasts:
        st.info("No forecast data available yet.")
        return

    for instrument, rule_forecasts in last_forecasts.items():
        st.subheader(f"{instrument}")

        # Forecast bar chart
        if rule_forecasts:
            fc_df = pd.DataFrame({
                "Rule": list(rule_forecasts.keys()),
                "Forecast": list(rule_forecasts.values()),
            })
            fc_df["Color"] = fc_df["Forecast"].apply(
                lambda x: "green" if x > 0 else "red"
            )

            st.bar_chart(fc_df.set_index("Rule")["Forecast"])

            # Forecast details table
            fc_df["Direction"] = fc_df["Forecast"].apply(
                lambda x: "LONG" if x > 0 else "SHORT" if x < 0 else "FLAT"
            )
            fc_df["Strength"] = fc_df["Forecast"].abs().apply(
                lambda x: "Strong" if x > 15 else "Medium" if x > 5 else "Weak"
            )
            st.dataframe(fc_df[["Rule", "Forecast", "Direction", "Strength"]],
                        use_container_width=True, hide_index=True)

    # Run history forecasts
    run_history = load_run_history()
    if len(run_history) > 2:
        st.subheader("Forecast History (Recent Runs)")
        rows = []
        for run in run_history[-50:]:
            ts = run.get("timestamp", "")
            forecasts = run.get("forecasts", {})
            for inst, rule_fcs in forecasts.items():
                for rule, val in rule_fcs.items():
                    rows.append({"timestamp": ts, "instrument": inst, "rule": rule, "forecast": val})
        if rows:
            hist_df = pd.DataFrame(rows)
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

            # Show combined forecast evolution
            for inst in hist_df["instrument"].unique():
                inst_df = hist_df[hist_df["instrument"] == inst]
                pivot = inst_df.pivot_table(index="timestamp", columns="rule", values="forecast")
                if len(pivot) > 1:
                    st.write(f"**{inst} Forecast Evolution**")
                    st.line_chart(pivot)


def render_risk(state: dict):
    """Risk page: drawdown, leverage, kill switch."""
    st.header("Risk Management")

    # Kill switch status
    kill_switch = state.get("kill_switch", False)
    if kill_switch:
        st.error("KILL SWITCH ACTIVE - All trading halted")
    else:
        st.success("Kill switch: OFF")

    # Risk metrics
    col1, col2, col3 = st.columns(3)

    equity = state.get("equity", state.get("capital", 0))
    peak = state.get("peak_equity", equity)
    daily_start = state.get("daily_start_equity", equity)

    drawdown = ((peak - equity) / peak) if peak > 0 else 0
    daily_loss = ((daily_start - equity) / daily_start) if daily_start > 0 else 0

    # Calculate leverage
    positions = state.get("positions", {})
    prices = state.get("prices", {})
    total_notional = sum(abs(p) * prices.get(s, 0) for s, p in positions.items())
    leverage = total_notional / equity if equity > 0 else 0

    col1.metric("Drawdown from Peak", f"{drawdown:.2%}", delta=None)
    col2.metric("Daily Loss", f"{daily_loss:.2%}", delta=None)
    col3.metric("Leverage", f"{leverage:.2f}x")

    # Risk limits
    st.subheader("Risk Limits")
    limits = pd.DataFrame([
        {"Metric": "Max Drawdown", "Current": f"{drawdown:.2%}", "Limit": "15%",
         "Status": "OK" if drawdown < 0.15 else "BREACH"},
        {"Metric": "Daily Loss", "Current": f"{daily_loss:.2%}", "Limit": "3%",
         "Status": "OK" if daily_loss < 0.03 else "BREACH"},
        {"Metric": "Leverage", "Current": f"{leverage:.2f}x", "Limit": "5.0x",
         "Status": "OK" if leverage < 5.0 else "BREACH"},
    ])
    st.dataframe(limits, use_container_width=True, hide_index=True)


def render_trades():
    """Trades page: trade log."""
    st.header("Trade Log")

    trades_df = load_trades()
    if trades_df.empty:
        st.info("No trades recorded yet.")
        return

    # Summary metrics
    n_trades = len(trades_df)
    if "success" in trades_df.columns:
        n_success = (trades_df["success"] == True).sum()  # noqa: E712
    else:
        n_success = n_trades

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", n_trades)
    col2.metric("Successful", n_success)
    col3.metric("Failed", n_trades - n_success)

    # Recent trades
    st.subheader("Recent Trades")
    display_cols = [c for c in ["timestamp", "instrument", "side", "size", "filled_price",
                                "notional", "forecast", "success"] if c in trades_df.columns]
    st.dataframe(trades_df[display_cols].tail(50).sort_index(ascending=False),
                use_container_width=True, hide_index=True)

    # Trade frequency chart
    if "timestamp" in trades_df.columns and len(trades_df) > 1:
        st.subheader("Trade Activity")
        trades_df["date"] = trades_df["timestamp"].dt.date
        daily_counts = trades_df.groupby("date").size()
        st.bar_chart(daily_counts)


def render_health():
    """System health page: runs, errors, scheduler."""
    st.header("System Health")

    run_history = load_run_history()
    if not run_history:
        st.info("No run history available.")
        return

    # Recent run stats
    recent = run_history[-100:]
    n_runs = len(recent)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", len(run_history))
    col2.metric("Recent (100)", n_runs)

    if recent:
        last_run = recent[-1]
        col3.metric("Last Run", last_run.get("timestamp", "")[:19])
        col4.metric("Duration", f"{last_run.get('duration_seconds', 0):.1f}s")

    # Equity over runs
    if len(recent) > 1:
        st.subheader("Equity Over Runs")
        equity_series = pd.Series(
            [r.get("equity", 0) for r in recent],
            index=pd.to_datetime([r.get("timestamp", "") for r in recent]),
        )
        if equity_series.sum() > 0:
            st.line_chart(equity_series)

    # Run duration distribution
    durations = [r.get("duration_seconds", 0) for r in recent if r.get("duration_seconds")]
    if durations:
        st.subheader("Run Duration (seconds)")
        st.bar_chart(pd.Series(durations, name="duration"))

    # Instruments traded
    st.subheader("Instruments per Run")
    inst_counts = [len(r.get("instruments_traded", [])) for r in recent]
    if inst_counts:
        st.line_chart(pd.Series(inst_counts, name="instruments"))


if __name__ == "__main__":
    main()
