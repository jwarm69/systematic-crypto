"""Technical indicator feature pipeline for ML models.

Timeframe-agnostic: works on any OHLCV DataFrame with consistent bar frequency.
Adapted from crypto-kalshi-predictor/src/features.py, stripped to pure functions.
"""

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Add return features over multiple lookback periods."""
    if periods is None:
        periods = [1, 4, 12, 24]
    for p in periods:
        df[f"return_{p}"] = df["close"].pct_change(p)
    return df


def add_rsi(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Add RSI features."""
    if periods is None:
        periods = [14, 7, 21]
    for p in periods:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / p, min_periods=p).mean()
        avg_loss = loss.ewm(alpha=1 / p, min_periods=p).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD, signal, histogram."""
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Band features."""
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = upper - lower
    df["bb_position"] = (df["close"] - lower) / width.replace(0, np.nan)
    df["bb_width"] = width / sma
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR and normalized ATR."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=period, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Add Stochastic K and D."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_min) / denom
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ADX trend strength."""
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)

    # When plus_dm > minus_dm, keep plus_dm; otherwise zero (and vice versa)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    atr = df.get("atr")
    if atr is None:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(span=period, adjust=False).mean()
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features."""
    vol_sma = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)
    df["volume_trend"] = df["volume"].rolling(24).mean() / df["volume"].rolling(72).mean().replace(0, np.nan)
    return df


def add_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add price position relative to moving averages."""
    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    df["price_vs_sma20"] = (df["close"] - sma20) / sma20
    df["price_vs_sma50"] = (df["close"] - sma50) / sma50
    df["price_vs_ema12"] = (df["close"] - ema12) / ema12
    df["sma_trend"] = (sma20 - sma50) / sma50
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features (hour-of-day, day-of-week)."""
    if not hasattr(df.index, "hour"):
        return df
    hour = df.index.hour
    dow = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add lagged indicator features."""
    if lags is None:
        lags = [6, 12, 24]
    base_cols = ["rsi_14", "macd_hist", "bb_position", "volume_ratio", "return_1"]
    existing = [c for c in base_cols if c in df.columns]
    for lag in lags:
        for col in existing:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_stats(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Add rolling statistics of returns."""
    ret = df["close"].pct_change()
    df["return_std_24"] = ret.rolling(window).std()
    df["return_skew_24"] = ret.rolling(window).skew()
    df["return_kurt_24"] = ret.rolling(window).kurt()
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    return df


def build_features(df: pd.DataFrame, include_time: bool = True) -> pd.DataFrame:
    """Build full feature set from OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame (must have open, high, low, close, volume columns)
        include_time: Whether to add time-based features

    Returns:
        DataFrame with all features added (original columns preserved)
    """
    df = df.copy()
    df = add_returns(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_stochastic(df)
    df = add_adx(df)
    df = add_volume_features(df)
    df = add_price_position(df)
    df = add_rolling_stats(df)
    if include_time:
        df = add_time_features(df)
    df = add_lag_features(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (everything except OHLCV and target)."""
    exclude = {"open", "high", "low", "close", "volume", "target", "future_return"}
    return [c for c in df.columns if c not in exclude]


def add_target(df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
    """Add binary classification target: 1 if next-bar return > 0.

    Args:
        df: DataFrame with 'close' column
        lookahead: Number of bars to look ahead

    Returns:
        DataFrame with 'future_return' and 'target' columns
    """
    df = df.copy()
    df["future_return"] = df["close"].pct_change(lookahead).shift(-lookahead)
    df["target"] = (df["future_return"] > 0).astype(int)
    return df
