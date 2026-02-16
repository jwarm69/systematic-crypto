"""Tests for Phase 2 trading rules and tools."""

import numpy as np
import pandas as pd
import pytest

from src.rules.breakout import BreakoutRule
from src.rules.carry import CarryRule
from src.rules.mean_reversion import MeanReversionRule
from src.rules.momentum import MomentumRule
from src.rules.ewmac import EWMACRule
from src.portfolio.handcraft import handcraft_weights, handcraft_with_fdm, compute_forecast_correlation
from src.backtest.significance import information_coefficient, rule_significance_report, marginal_value_test


def _make_prices(n=1000, start=50000, drift=0.0001, vol=0.02, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = np.random.randn(n) * vol + drift
    prices = start * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


def _make_trending(n=1000, start=50000, trend=0.001, vol=0.01):
    """Make a clearly trending price series."""
    np.random.seed(123)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = np.ones(n) * trend + np.random.randn(n) * vol
    prices = start * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


class TestBreakout:
    def test_produces_valid_forecast(self):
        prices = _make_prices()
        rule = BreakoutRule(lookback=20)
        fc = rule.forecast(prices)
        assert not fc.empty
        assert fc.abs().max() <= 20.0

    def test_trending_up_positive(self):
        prices = _make_trending(trend=0.002)
        rule = BreakoutRule(lookback=20)
        fc = rule.forecast(prices)
        # In a strong uptrend, the average forecast should be positive
        # (individual bars may fluctuate due to scaling dynamics)
        assert fc.mean() > 0

    def test_different_lookbacks(self):
        prices = _make_prices()
        short = BreakoutRule(20).forecast(prices)
        long = BreakoutRule(80).forecast(prices)
        # Different lookbacks should produce different forecasts
        assert not short.equals(long)


class TestCarry:
    def test_no_funding_returns_zero(self):
        prices = _make_prices(200)
        rule = CarryRule()
        # Without funding_rates, raw forecast is zero. Scaling zero gives NaN/zero.
        raw = rule.calculate_raw_forecast(prices)
        assert (raw == 0).all()

    def test_with_funding_data(self):
        prices = _make_prices(500)
        # Simulate positive funding (longs pay shorts -> bearish carry)
        np.random.seed(42)
        funding = pd.Series(
            0.001 + np.random.randn(500) * 0.0001,
            index=prices.index,
        )
        rule = CarryRule(smooth_days=30)
        fc = rule.forecast(prices, funding_rates=funding)
        assert len(fc) > 0
        # Positive funding -> negative carry forecast (should be short)
        assert fc.iloc[-1] < 0


class TestMeanReversion:
    def test_produces_valid_forecast(self):
        prices = _make_prices()
        rule = MeanReversionRule(lookback=20)
        fc = rule.forecast(prices)
        assert not fc.empty
        assert fc.abs().max() <= 20.0

    def test_above_mean_gives_negative(self):
        """Price above recent mean should give sell signal."""
        dates = pd.date_range("2024-01-01", periods=200, freq="1h")
        # Flat then sharp up
        prices = pd.Series(
            np.concatenate([np.ones(180) * 50000, np.linspace(50000, 55000, 20)]),
            index=dates,
        )
        rule = MeanReversionRule(lookback=20)
        raw = rule.calculate_raw_forecast(prices)
        # Price above mean -> negative forecast (sell)
        assert raw.iloc[-1] < 0


class TestMomentum:
    def test_produces_valid_forecast(self):
        prices = _make_prices()
        rule = MomentumRule(lookback=100)
        fc = rule.forecast(prices)
        assert not fc.empty
        assert fc.abs().max() <= 20.0

    def test_trending_up_positive(self):
        prices = _make_trending(trend=0.002)
        rule = MomentumRule(lookback=100)
        fc = rule.forecast(prices)
        assert fc.iloc[-1] > 0


class TestHandcraftWeights:
    def test_equal_weight_uncorrelated(self):
        """Uncorrelated rules should get roughly equal weight."""
        corr = pd.DataFrame(
            [[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]],
            index=["a", "b", "c"], columns=["a", "b", "c"],
        )
        weights = handcraft_weights(corr, correlation_threshold=0.65)
        assert len(weights) == 3
        # All in separate groups -> roughly equal weight
        for w in weights.values():
            assert abs(w - 1/3) < 0.01

    def test_correlated_pair_grouped(self):
        """Highly correlated rules should be grouped and share weight."""
        corr = pd.DataFrame(
            [[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]],
            index=["a", "b", "c"], columns=["a", "b", "c"],
        )
        weights = handcraft_weights(corr, correlation_threshold=0.65)
        # a and b grouped -> each gets 0.25, c gets 0.5
        assert abs(weights["c"] - 0.5) < 0.01
        assert abs(weights["a"] - 0.25) < 0.01

    def test_single_rule(self):
        corr = pd.DataFrame([[1.0]], index=["a"], columns=["a"])
        weights = handcraft_weights(corr)
        assert weights == {"a": 1.0}

    def test_with_fdm(self):
        prices = _make_prices(500)
        forecasts = {
            "ewmac_8_32": EWMACRule(8, 32).forecast(prices),
            "breakout_20": BreakoutRule(20).forecast(prices),
            "mean_rev_20": MeanReversionRule(20).forecast(prices),
        }
        weights, fdm = handcraft_with_fdm(forecasts)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
        assert fdm >= 1.0
        assert fdm <= 2.5


class TestSignificance:
    def test_ic_random_near_zero(self):
        """Random forecast should have IC near zero."""
        np.random.seed(42)
        fc = pd.Series(np.random.randn(1000))
        ret = pd.Series(np.random.randn(1000))
        ic, t_stat, p_val = information_coefficient(fc, ret)
        assert abs(ic) < 0.1  # Near zero

    def test_ic_perfect_signal(self):
        """Perfect signal should have high IC."""
        np.random.seed(99)
        returns = pd.Series(np.random.randn(500))
        # forecast[t] should predict returns[t+1], so forecast = returns shifted forward
        # IC tests: corr(forecast[t], returns[t+lag])
        # If forecast[t] = returns[t+1], then forecast[t] and returns.shift(-1)[t] should correlate
        forecast = returns.shift(-1).fillna(0)
        ic, t_stat, p_val = information_coefficient(forecast, returns, lag=1)
        assert ic > 0.8

    def test_significance_report(self):
        prices = _make_prices(500)
        rule = EWMACRule(16, 64)
        fc = rule.forecast(prices)
        report = rule_significance_report(fc, prices, "ewmac_16_64")
        assert "ic_1h" in report
        assert "p_value_1h" in report
        assert "sharpe_ratio" in report
        assert report["n_observations"] > 0

    def test_marginal_value(self):
        prices = _make_prices(500)
        existing = {"ewmac": EWMACRule(16, 64).forecast(prices)}
        new_fc = BreakoutRule(20).forecast(prices)
        result = marginal_value_test(existing, new_fc, prices)
        assert "sharpe_without" in result
        assert "sharpe_with" in result
        assert "improves_system" in result
