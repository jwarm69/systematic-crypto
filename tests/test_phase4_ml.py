"""Tests for Phase 4: ML forecast rule, feature pipeline, and training."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.technical import (
    add_returns,
    add_rsi,
    add_macd,
    add_bollinger,
    add_atr,
    add_target,
    build_features,
    get_feature_columns,
)
from src.ml.trainer import (
    select_features,
    train_lightgbm,
    train_xgboost,
    train_with_cv,
    ensemble_predict_proba,
)
from src.rules.ml_forecast import MLForecastRule


def _make_ohlcv(n=500, start=50000, drift=0.0001, vol=0.02, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = np.random.randn(n) * vol + drift
    close = start * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close": close,
        "volume": np.abs(np.random.randn(n)) * 1000 + 500,
    }, index=dates)
    return df


class TestFeaturePipeline:
    def test_add_returns(self):
        df = _make_ohlcv(100)
        df = add_returns(df)
        assert "return_1" in df.columns
        assert "return_24" in df.columns

    def test_add_rsi(self):
        df = _make_ohlcv(100)
        df = add_rsi(df)
        assert "rsi_14" in df.columns
        # RSI should be between 0 and 100
        valid = df["rsi_14"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_add_macd(self):
        df = _make_ohlcv(100)
        df = add_macd(df)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_add_bollinger(self):
        df = _make_ohlcv(100)
        df = add_bollinger(df)
        assert "bb_position" in df.columns
        assert "bb_width" in df.columns

    def test_add_atr(self):
        df = _make_ohlcv(100)
        df = add_atr(df)
        assert "atr" in df.columns
        assert "atr_pct" in df.columns
        valid = df["atr"].dropna()
        assert (valid > 0).all()

    def test_build_features(self):
        df = _make_ohlcv(200)
        featured = build_features(df)
        feature_cols = get_feature_columns(featured)
        # Should generate a good number of features
        assert len(feature_cols) >= 20
        # Original columns should still be there
        assert "close" in featured.columns
        assert "volume" in featured.columns

    def test_add_target(self):
        df = _make_ohlcv(100)
        df = add_target(df, lookahead=1)
        assert "target" in df.columns
        assert "future_return" in df.columns
        # Target is binary
        assert set(df["target"].dropna().unique()).issubset({0, 1})
        # Last row should have NaN target (no future data)
        assert pd.isna(df["future_return"].iloc[-1])


class TestFeatureSelection:
    def test_selects_features(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 10), columns=[f"f{i}" for i in range(10)])
        # Make f0 informative
        y = pd.Series((X["f0"] > 0).astype(int))
        selected = select_features(X, y, top_k=5)
        assert len(selected) == 5
        # f0 should be selected (it's informative)
        assert "f0" in selected


class TestModelTraining:
    def test_train_lightgbm(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(300, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((X["f0"] + np.random.randn(300) * 0.5 > 0).astype(int))
        X_train, y_train = X.iloc[:200], y.iloc[:200]
        X_val, y_val = X.iloc[200:], y.iloc[200:]

        model = train_lightgbm(X_train, y_train, X_val, y_val)
        proba = model.predict_proba(X_val)
        assert proba.shape == (100, 2)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_train_xgboost(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(300, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((X["f0"] + np.random.randn(300) * 0.5 > 0).astype(int))
        X_train, y_train = X.iloc[:200], y.iloc[:200]
        X_val, y_val = X.iloc[200:], y.iloc[200:]

        model = train_xgboost(X_train, y_train, X_val, y_val)
        proba = model.predict_proba(X_val)
        assert proba.shape == (100, 2)

    def test_train_with_cv(self):
        np.random.seed(42)
        n = 500
        X = pd.DataFrame(np.random.randn(n, 8), columns=[f"f{i}" for i in range(8)])
        # Slightly informative features
        signal = X["f0"] * 0.3 + X["f1"] * 0.2
        y = pd.Series((signal + np.random.randn(n) * 0.8 > 0).astype(int))

        result = train_with_cv(
            X, y,
            features=list(X.columns),
            n_cv_splits=3,
            purge=10,
            embargo=5,
            n_optuna_trials=5,  # Small for speed
            feature_selection_k=8,
        )

        assert len(result.models) == 2  # LGB + XGB
        assert len(result.weights) == 2
        assert abs(sum(result.weights) - 1.0) < 0.01
        assert result.test_auc > 0.4  # Should be somewhat better than random
        assert 0 <= result.test_accuracy <= 1

    def test_ensemble_predict(self):
        np.random.seed(42)
        n = 500
        X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((X["f0"] > 0).astype(int))

        result = train_with_cv(
            X, y,
            features=list(X.columns),
            n_cv_splits=3,
            n_optuna_trials=3,
            feature_selection_k=5,
        )

        probas = ensemble_predict_proba(result, X)
        assert len(probas) == n
        assert probas.min() >= 0
        assert probas.max() <= 1


class TestMLForecastRule:
    def test_train_and_forecast(self):
        df = _make_ohlcv(500)
        rule = MLForecastRule(
            n_cv_splits=3,
            n_optuna_trials=3,
            feature_selection_k=15,
        )
        rule.train(df)

        assert rule.ensemble is not None

        forecast = rule.forecast(df["close"])
        assert len(forecast) > 0
        assert forecast.abs().max() <= 20.0
        # Should have some non-zero forecasts
        assert (forecast != 0).any()

    def test_forecast_range(self):
        df = _make_ohlcv(500)
        rule = MLForecastRule(n_cv_splits=3, n_optuna_trials=3, feature_selection_k=10)
        rule.train(df)
        fc = rule.forecast(df["close"])
        # All forecasts should be in [-20, 20]
        assert fc.min() >= -20.0
        assert fc.max() <= 20.0

    def test_raises_without_training(self):
        rule = MLForecastRule()
        prices = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3, freq="1h"))
        with pytest.raises(RuntimeError, match="not trained"):
            rule.forecast(prices)
