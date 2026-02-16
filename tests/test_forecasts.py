"""Tests for trading rules and forecast scaling."""

import numpy as np
import pandas as pd
import pytest

from src.rules.ewmac import EWMACRule
from src.rules.scaling import scale_forecast, combine_forecasts, calculate_fdm


def _make_prices(n=1000, start=50000, drift=0.0001, vol=0.02):
    """Generate synthetic price series."""
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = np.random.randn(n) * vol + drift
    prices = start * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


class TestEWMAC:
    def test_basic_forecast(self):
        """EWMAC should produce valid forecasts."""
        prices = _make_prices(500)
        rule = EWMACRule(fast_span=16, slow_span=64)
        forecast = rule.forecast(prices)

        assert not forecast.empty
        assert forecast.abs().max() <= 20.0  # Capped
        assert len(forecast) > 0

    def test_trending_market_positive_forecast(self):
        """Strong uptrend should produce positive forecast."""
        dates = pd.date_range("2024-01-01", periods=500, freq="1h")
        # Strong uptrend
        prices = pd.Series(
            50000 * np.exp(np.cumsum(np.ones(500) * 0.001 + np.random.randn(500) * 0.005)),
            index=dates,
        )
        rule = EWMACRule(fast_span=16, slow_span=64)
        forecast = rule.forecast(prices)

        # Last forecast should be positive (trending up)
        assert forecast.iloc[-1] > 0

    def test_raw_forecast_unscaled(self):
        """Raw forecast should not be capped at 20."""
        prices = _make_prices(500)
        rule = EWMACRule(fast_span=16, slow_span=64)
        raw = rule.calculate_raw_forecast(prices)

        # Raw can exceed 20 (not yet scaled)
        assert len(raw) > 0

    def test_different_speeds(self):
        """Faster EWMAC should be more reactive."""
        prices = _make_prices(1000)

        fast_rule = EWMACRule(fast_span=4, slow_span=16)
        slow_rule = EWMACRule(fast_span=32, slow_span=128)

        fast_fc = fast_rule.forecast(prices)
        slow_fc = slow_rule.forecast(prices)

        # Fast signal should change more frequently
        fast_changes = fast_fc.diff().abs().mean()
        slow_changes = slow_fc.diff().abs().mean()
        assert fast_changes > slow_changes


class TestScaling:
    def test_scale_to_target_abs(self):
        """Scaled forecast should have avg abs close to 10."""
        raw = pd.Series(np.random.randn(1000) * 5)
        scaled = scale_forecast(raw, target_abs=10.0)

        # After warm-up period, avg abs should be close to 10
        avg_abs = scaled.iloc[300:].abs().mean()
        assert 5 < avg_abs < 15  # Reasonable range

    def test_capped_at_20(self):
        """Scaled forecast should never exceed +-20."""
        raw = pd.Series(np.random.randn(1000) * 100)
        scaled = scale_forecast(raw)
        assert scaled.max() <= 20.0
        assert scaled.min() >= -20.0


class TestCombineForecasts:
    def test_equal_weight(self):
        """Equal weighted combination of two forecasts."""
        idx = pd.date_range("2024-01-01", periods=100, freq="1h")
        f1 = pd.Series(10.0, index=idx)
        f2 = pd.Series(-10.0, index=idx)

        combined = combine_forecasts({"f1": f1, "f2": f2})
        # Equal weight: (10 + -10) / 2 = 0
        assert abs(combined.iloc[-1]) < 0.01

    def test_fdm_applied(self):
        """FDM should boost combined forecast."""
        idx = pd.date_range("2024-01-01", periods=100, freq="1h")
        f1 = pd.Series(5.0, index=idx)
        f2 = pd.Series(5.0, index=idx)

        no_fdm = combine_forecasts({"f1": f1, "f2": f2}, fdm=1.0)
        with_fdm = combine_forecasts({"f1": f1, "f2": f2}, fdm=1.5)

        assert with_fdm.iloc[-1] > no_fdm.iloc[-1]

    def test_capped_after_fdm(self):
        """Combined forecast should still be capped at +-20."""
        idx = pd.date_range("2024-01-01", periods=100, freq="1h")
        f1 = pd.Series(18.0, index=idx)

        combined = combine_forecasts({"f1": f1}, fdm=2.0)
        assert combined.max() <= 20.0


class TestFDM:
    def test_uncorrelated_forecasts(self):
        """Uncorrelated forecasts should have FDM > 1."""
        corr = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["f1", "f2"], columns=["f1", "f2"],
        )
        weights = pd.Series([0.5, 0.5], index=["f1", "f2"])
        fdm = calculate_fdm(corr, weights)
        assert abs(fdm - np.sqrt(2)) < 0.01  # ~1.41

    def test_perfectly_correlated(self):
        """Perfectly correlated forecasts should have FDM = 1."""
        corr = pd.DataFrame(
            [[1.0, 1.0], [1.0, 1.0]],
            index=["f1", "f2"], columns=["f1", "f2"],
        )
        weights = pd.Series([0.5, 0.5], index=["f1", "f2"])
        fdm = calculate_fdm(corr, weights)
        assert abs(fdm - 1.0) < 0.01

    def test_fdm_capped_at_2_5(self):
        """FDM should not exceed 2.5."""
        # Many uncorrelated signals
        n = 10
        corr = pd.DataFrame(np.eye(n))
        weights = pd.Series(1.0 / n, index=range(n))
        fdm = calculate_fdm(corr, weights)
        assert fdm <= 2.5
