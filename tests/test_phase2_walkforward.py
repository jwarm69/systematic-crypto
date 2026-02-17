"""Tests for constrained Phase 2 walk-forward tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.phase2_walkforward import constrained_weight_search, walk_forward_weight_selection
from src.rules.base import AbstractTradingRule


class DummyRule(AbstractTradingRule):
    """Simple deterministic rule for test scenarios."""

    def __init__(self, name: str, lookback: int):
        super().__init__(name=name, lookback=lookback)
        self.lookback = lookback

    def calculate_raw_forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        return prices.pct_change(self.lookback).fillna(0.0)


def _prices(n: int = 900) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    trend = np.linspace(100.0, 140.0, n)
    cyc = 2.0 * np.sin(np.linspace(0, 12 * np.pi, n))
    return pd.Series(trend + cyc, index=idx)


def _forecasts(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    x = np.linspace(0.0, 8 * np.pi, len(index))
    return {
        "r1": pd.Series(np.sin(x) * 10, index=index),
        "r2": pd.Series(np.cos(x) * 8, index=index),
        "r3": pd.Series(np.sin(x / 2.0) * 6, index=index),
        "r4": pd.Series(np.cos(x / 3.0) * 5, index=index),
    }


def test_constrained_weight_search_respects_bounds():
    prices = _prices(700)
    forecasts = _forecasts(prices.index)
    engine = BacktestEngine(timeframe="1h", max_leverage=2.0)

    result = constrained_weight_search(
        train_prices=prices,
        train_forecasts=forecasts,
        engine=engine,
        n_trials=200,
        max_weight=0.55,
        min_active_rules=3,
        min_rule_weight=0.05,
        seed=123,
    )

    weights = pd.Series(result.weights)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(weights.max()) <= 0.55 + 1e-9
    assert int((weights > 0).sum()) >= 3
    assert result.fdm >= 1.0


def test_constrained_weight_search_rejects_infeasible_constraints():
    prices = _prices(500)
    forecasts = _forecasts(prices.index)
    engine = BacktestEngine(timeframe="1h")

    with pytest.raises(ValueError, match="Infeasible constraints"):
        constrained_weight_search(
            train_prices=prices,
            train_forecasts=forecasts,
            engine=engine,
            max_weight=0.24,
            min_active_rules=4,
        )


def test_walk_forward_weight_selection_returns_summary_and_weights():
    prices = _prices(1_100)
    rules = {
        "fast": DummyRule("fast", 4),
        "mid": DummyRule("mid", 16),
        "slow": DummyRule("slow", 48),
    }
    engine = BacktestEngine(timeframe="1h", max_leverage=2.0)

    report = walk_forward_weight_selection(
        prices=prices,
        rules=rules,
        engine=engine,
        n_splits=4,
        test_size=150,
        purge=12,
        search_trials=120,
        max_weight=0.70,
        min_active_rules=2,
        min_rule_weight=0.05,
        seed=11,
    )

    assert report["summary"]["n_folds"] > 0
    assert "candidate_oos_sharpe" in report["summary"]
    weights = pd.Series(report["recommended_weights"])
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(weights.max()) <= 0.70 + 1e-9
    assert int((weights > 0).sum()) >= 2
