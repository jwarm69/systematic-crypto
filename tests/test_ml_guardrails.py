"""Tests for ML walk-forward guardrails."""

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.ml.guardrails import (
    evaluate_candidate_vs_baseline,
    promotion_gate,
    purged_walk_forward_splits,
)


def _make_trending_prices(n: int = 700) -> pd.Series:
    rng = np.random.RandomState(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = 0.0007 + rng.randn(n) * 0.001
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=idx)


def test_purged_walk_forward_splits_respect_gap():
    splits = purged_walk_forward_splits(
        n_samples=200,
        n_splits=4,
        test_size=20,
        purge=5,
        embargo=3,
    )
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert train_idx[-1] <= test_idx[0] - 8


def test_promotion_gate_accepts_better_candidate():
    prices = _make_trending_prices()
    baseline = pd.Series(0.0, index=prices.index)
    candidate = pd.Series(10.0, index=prices.index)

    engine = BacktestEngine(
        timeframe="1h",
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        buffer_fraction=0.0,
    )
    evaluation = evaluate_candidate_vs_baseline(
        prices=prices,
        baseline_forecasts=baseline,
        candidate_forecasts=candidate,
        engine=engine,
        n_splits=4,
        test_size=100,
        purge=24,
        embargo=24,
    )
    decision = promotion_gate(
        evaluation,
        min_mean_sharpe_delta=0.01,
        min_outperform_fraction=0.75,
        max_drawdown_worsening=0.10,
    )

    assert decision.approved


def test_promotion_gate_rejects_worse_candidate():
    prices = _make_trending_prices()
    baseline = pd.Series(0.0, index=prices.index)
    candidate = pd.Series(-10.0, index=prices.index)

    engine = BacktestEngine(
        timeframe="1h",
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        buffer_fraction=0.0,
    )
    evaluation = evaluate_candidate_vs_baseline(
        prices=prices,
        baseline_forecasts=baseline,
        candidate_forecasts=candidate,
        engine=engine,
        n_splits=4,
        test_size=100,
        purge=24,
        embargo=24,
    )
    decision = promotion_gate(
        evaluation,
        min_mean_sharpe_delta=0.01,
        min_outperform_fraction=0.75,
        max_drawdown_worsening=0.10,
    )

    assert not decision.approved
