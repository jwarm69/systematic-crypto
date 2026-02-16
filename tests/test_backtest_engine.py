"""Tests for backtest timeframe scaling and funding-aware accounting."""

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, periods_per_year_from_timeframe


def _make_prices(n: int = 400) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    base = 0.0008
    wiggle = 0.0015 * np.sin(np.arange(n) / 12.0)
    returns = base + wiggle
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=idx)


def test_periods_per_year_mapping():
    assert periods_per_year_from_timeframe("1h") == 365 * 24
    assert periods_per_year_from_timeframe("4h") == 365 * 6
    assert periods_per_year_from_timeframe("1d") == 365


def test_timeframe_controls_annualization_metrics():
    prices = _make_prices(500)
    forecast = pd.Series(10.0, index=prices.index)

    hourly = BacktestEngine(timeframe="1h", taker_fee_bps=0.0, slippage_bps=0.0)
    daily = BacktestEngine(timeframe="1d", taker_fee_bps=0.0, slippage_bps=0.0)

    hourly_metrics = hourly.run(prices, forecast)["metrics"]
    daily_metrics = daily.run(prices, forecast)["metrics"]

    assert hourly_metrics["bars_per_year"] == 365 * 24
    assert daily_metrics["bars_per_year"] == 365
    assert hourly_metrics["annualized_vol"] != daily_metrics["annualized_vol"]


def test_constant_funding_reduces_pnl():
    prices = _make_prices(500)
    forecast = pd.Series(10.0, index=prices.index)

    no_funding = BacktestEngine(
        timeframe="1h",
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        funding_bps_per_8h=0.0,
    )
    with_funding = BacktestEngine(
        timeframe="1h",
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        funding_bps_per_8h=8.0,
    )

    r0 = no_funding.run(prices, forecast)
    r1 = with_funding.run(prices, forecast)

    assert r1["metrics"]["total_funding_pnl"] < 0
    assert r1["metrics"]["final_value"] < r0["metrics"]["final_value"]
