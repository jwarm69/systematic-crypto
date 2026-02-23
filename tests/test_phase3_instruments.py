"""Tests for Phase 3: multi-instrument weights, IDM, and portfolio backtest."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.instrument_weights import (
    calculate_idm,
    compute_instrument_correlation,
    equal_weights,
    recommend_weights_and_idm,
)
from src.backtest.multi_instrument import MultiInstrumentBacktest
from src.data.hyperliquid_data import (
    align_price_series,
    compute_cross_asset_features,
    compute_returns,
    liquidity_filter,
)
from src.rules.ewmac import EWMACRule
from src.rules.scaling import combine_forecasts


def _make_prices(n=1000, start=50000, drift=0.0001, vol=0.02, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = np.random.randn(n) * vol + drift
    prices = start * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


def _make_ohlcv(n=1000, start=50000, drift=0.0001, vol=0.02, seed=42):
    prices = _make_prices(n, start, drift, vol, seed)
    df = pd.DataFrame({
        "open": prices * (1 - 0.001),
        "high": prices * (1 + 0.005),
        "low": prices * (1 - 0.005),
        "close": prices,
        "volume": np.random.rand(n) * 1000 + 100,
    }, index=prices.index)
    return df


def _make_correlated_prices(n=1000, seed=42, corr=0.8):
    """Make two correlated price series."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    z1 = np.random.randn(n) * 0.02
    z2 = corr * z1 + np.sqrt(1 - corr**2) * np.random.randn(n) * 0.02
    prices1 = 50000 * np.exp(np.cumsum(z1 + 0.0001))
    prices2 = 3000 * np.exp(np.cumsum(z2 + 0.0001))
    return pd.Series(prices1, index=dates), pd.Series(prices2, index=dates)


class TestEqualWeights:
    def test_single_instrument(self):
        w = equal_weights(["BTC"])
        assert w == {"BTC": 1.0}

    def test_three_instruments(self):
        w = equal_weights(["BTC", "ETH", "SOL"])
        assert len(w) == 3
        for v in w.values():
            assert abs(v - 1/3) < 0.001
        assert abs(sum(w.values()) - 1.0) < 0.001

    def test_empty(self):
        assert equal_weights([]) == {}


class TestInstrumentCorrelation:
    def test_single_instrument(self):
        returns = {"BTC": pd.Series(np.random.randn(200))}
        corr = compute_instrument_correlation(returns)
        assert corr.shape == (1, 1)
        assert corr.iloc[0, 0] == 1.0

    def test_uncorrelated(self):
        np.random.seed(42)
        returns = {
            "A": pd.Series(np.random.randn(2000)),
            "B": pd.Series(np.random.randn(2000)),
        }
        corr = compute_instrument_correlation(returns)
        assert abs(corr.loc["A", "B"]) < 0.1

    def test_highly_correlated(self):
        np.random.seed(42)
        z = np.random.randn(2000)
        returns = {
            "A": pd.Series(z),
            "B": pd.Series(z * 0.9 + np.random.randn(2000) * 0.1),
        }
        corr = compute_instrument_correlation(returns)
        assert corr.loc["A", "B"] > 0.7

    def test_diagonal_is_one(self):
        np.random.seed(42)
        returns = {
            "A": pd.Series(np.random.randn(500)),
            "B": pd.Series(np.random.randn(500)),
            "C": pd.Series(np.random.randn(500)),
        }
        corr = compute_instrument_correlation(returns)
        for name in ["A", "B", "C"]:
            assert corr.loc[name, name] == 1.0


class TestIDM:
    def test_single_instrument(self):
        w = {"BTC": 1.0}
        corr = pd.DataFrame([[1.0]], index=["BTC"], columns=["BTC"])
        idm = calculate_idm(w, corr)
        assert idm == 1.0

    def test_two_uncorrelated(self):
        w = {"A": 0.5, "B": 0.5}
        corr = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["A", "B"], columns=["A", "B"],
        )
        idm = calculate_idm(w, corr)
        # IDM = 1/sqrt(0.5^2 + 0.5^2) = 1/sqrt(0.5) = 1.414
        assert abs(idm - 1.414) < 0.01

    def test_two_perfectly_correlated(self):
        w = {"A": 0.5, "B": 0.5}
        corr = pd.DataFrame(
            [[1.0, 1.0], [1.0, 1.0]],
            index=["A", "B"], columns=["A", "B"],
        )
        idm = calculate_idm(w, corr)
        # IDM = 1/sqrt(1.0) = 1.0
        assert abs(idm - 1.0) < 0.01

    def test_three_uncorrelated(self):
        w = {"A": 1/3, "B": 1/3, "C": 1/3}
        corr = pd.DataFrame(
            np.eye(3), index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        idm = calculate_idm(w, corr)
        # IDM = 1/sqrt(3*(1/3)^2) = 1/sqrt(1/3) = sqrt(3) = 1.732
        assert abs(idm - 1.732) < 0.01

    def test_cap(self):
        """IDM should be capped at 2.5."""
        w = {f"inst_{i}": 1/10 for i in range(10)}
        corr = pd.DataFrame(
            np.eye(10),
            index=list(w.keys()), columns=list(w.keys()),
        )
        idm = calculate_idm(w, corr, cap=2.5)
        assert idm == 2.5

    def test_realistic_crypto(self):
        """Crypto instruments are typically 0.6-0.9 correlated."""
        w = {"BTC": 1/3, "ETH": 1/3, "SOL": 1/3}
        corr = pd.DataFrame(
            [[1.0, 0.80, 0.70],
             [0.80, 1.0, 0.75],
             [0.70, 0.75, 1.0]],
            index=["BTC", "ETH", "SOL"], columns=["BTC", "ETH", "SOL"],
        )
        idm = calculate_idm(w, corr)
        # With high crypto correlations, IDM should be modest: ~1.1-1.3
        assert 1.0 <= idm <= 1.5


class TestRecommendWeightsAndIDM:
    def test_end_to_end(self):
        np.random.seed(42)
        returns = {
            "BTC": pd.Series(np.random.randn(500) * 0.02),
            "ETH": pd.Series(np.random.randn(500) * 0.03),
        }
        weights, idm, corr = recommend_weights_and_idm(returns)
        assert abs(weights["BTC"] - 0.5) < 0.01
        assert abs(weights["ETH"] - 0.5) < 0.01
        assert idm >= 1.0
        assert corr.shape == (2, 2)


class TestMultiInstrumentBacktest:
    def test_single_instrument_matches_single_engine(self):
        """Multi-instrument engine with 1 instrument should match single engine."""
        prices = _make_prices(500, seed=42)
        rule = EWMACRule(16, 64)
        forecasts = rule.forecast(prices)

        from src.backtest.engine import BacktestEngine
        single = BacktestEngine(capital=5000, vol_target=0.12)
        single_result = single.run(prices, forecasts)

        multi = MultiInstrumentBacktest(capital=5000, vol_target=0.12)
        multi_result = multi.run(
            {"BTC": prices},
            {"BTC": forecasts},
            instrument_weights={"BTC": 1.0},
            idm=1.0,
        )

        # Sharpe should be similar (not exact due to slight implementation differences)
        assert abs(
            multi_result["metrics"]["sharpe_ratio"] -
            single_result["metrics"]["sharpe_ratio"]
        ) < 0.5

    def test_two_instruments(self):
        btc = _make_prices(500, start=50000, seed=42)
        eth = _make_prices(500, start=3000, seed=99)

        rule = EWMACRule(16, 64)
        fc_btc = rule.forecast(btc)
        fc_eth = rule.forecast(eth)

        multi = MultiInstrumentBacktest(capital=5000, vol_target=0.12)
        result = multi.run(
            {"BTC": btc, "ETH": eth},
            {"BTC": fc_btc, "ETH": fc_eth},
        )

        assert result["metrics"]["n_instruments"] == 2
        assert result["idm"] >= 1.0
        assert "BTC" in result["instrument_results"]
        assert "ETH" in result["instrument_results"]

    def test_idm_increases_returns(self):
        """Higher IDM should lead to larger positions and different returns."""
        btc = _make_prices(500, seed=42)
        eth = _make_prices(500, start=3000, seed=99)

        rule = EWMACRule(16, 64)
        fc_btc = rule.forecast(btc)
        fc_eth = rule.forecast(eth)

        multi = MultiInstrumentBacktest(capital=5000, vol_target=0.12)

        r_low = multi.run(
            {"BTC": btc, "ETH": eth},
            {"BTC": fc_btc, "ETH": fc_eth},
            idm=1.0,
        )
        r_high = multi.run(
            {"BTC": btc, "ETH": eth},
            {"BTC": fc_btc, "ETH": fc_eth},
            idm=1.5,
        )

        # Higher IDM -> larger positions -> different returns
        btc_pos_low = r_low["instrument_results"]["BTC"]["positions"].abs().mean()
        btc_pos_high = r_high["instrument_results"]["BTC"]["positions"].abs().mean()
        assert btc_pos_high > btc_pos_low


class TestCrossAssetFeatures:
    def test_compute_returns(self):
        data = {
            "BTC": _make_ohlcv(100, seed=42),
            "ETH": _make_ohlcv(100, start=3000, seed=99),
        }
        returns = compute_returns(data)
        assert "BTC" in returns
        assert "ETH" in returns
        assert len(returns["BTC"]) == 99

    def test_cross_features(self):
        data = {
            "BTC": _make_ohlcv(500, seed=42),
            "ETH": _make_ohlcv(500, start=3000, seed=99),
        }
        features = compute_cross_asset_features(data, reference="BTC")
        assert "BTC" in features
        assert "ETH" in features
        # ETH should have relative strength features
        eth_cols = list(features["ETH"].columns)
        assert any("relative_strength" in c for c in eth_cols)
        assert any("ref_momentum" in c for c in eth_cols)

    def test_align_prices(self):
        data = {
            "BTC": _make_ohlcv(200, seed=42),
            "ETH": _make_ohlcv(200, start=3000, seed=99),
        }
        aligned = align_price_series(data)
        assert "BTC" in aligned.columns
        assert "ETH" in aligned.columns
        assert len(aligned) == 200


class TestLiquidityFilter:
    def test_high_volume_passes(self):
        # Create data with high volume
        df = _make_ohlcv(500, seed=42)
        df["volume"] = 100_000  # High volume
        data = {"BTC": df}
        passed = liquidity_filter(data, min_daily_volume_usd=1_000)
        assert "BTC" in passed

    def test_low_volume_fails(self):
        df = _make_ohlcv(500, seed=42)
        df["volume"] = 0.001  # Tiny volume
        data = {"SHITCOIN": df}
        passed = liquidity_filter(data, min_daily_volume_usd=1_000_000_000)
        assert "SHITCOIN" not in passed
