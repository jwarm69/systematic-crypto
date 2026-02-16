"""Forecast scaling and combination (Carver's methodology).

Adapted from crypto_momo/momo/signals.py.
"""

import numpy as np
import pandas as pd


def scale_forecast(
    raw_forecast: pd.Series,
    target_abs: float = 10.0,
    cap: float = 20.0,
    scalar_lookback: int = 252,
    min_periods: int = 50,
) -> pd.Series:
    """Scale raw forecast to Carver's standard [-20, +20] range.

    The scalar makes the average absolute forecast equal to target_abs (10).
    Then caps at +/-cap (20).

    Args:
        raw_forecast: Raw forecast values (any scale)
        target_abs: Target average absolute forecast
        cap: Maximum absolute forecast
        scalar_lookback: Rolling window for computing scalar
        min_periods: Minimum periods before rolling scalar is valid

    Returns:
        Scaled and capped forecast series
    """
    # Rolling mean of absolute forecast -> scalar
    rolling_abs_mean = raw_forecast.abs().rolling(
        window=scalar_lookback, min_periods=min_periods
    ).mean()

    # Avoid division by zero
    rolling_abs_mean = rolling_abs_mean.replace(0, np.nan)

    forecast_scalar = target_abs / rolling_abs_mean
    scaled = raw_forecast * forecast_scalar

    # Cap
    capped = scaled.clip(-cap, cap)

    # Fill early NaN with simple (full-sample) scaling
    if capped.isna().any():
        overall_abs_mean = raw_forecast.abs().mean()
        if overall_abs_mean > 0:
            simple_scalar = target_abs / overall_abs_mean
            simple_scaled = (raw_forecast * simple_scalar).clip(-cap, cap)
            capped = capped.fillna(simple_scaled)

    return capped


def combine_forecasts(
    forecasts: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    fdm: float = 1.0,
    cap: float = 20.0,
) -> pd.Series:
    """Combine multiple forecasts using Carver's methodology.

    Steps:
    1. Weight each forecast
    2. Apply Forecast Diversification Multiplier (FDM)
    3. Re-cap at +/-20

    Args:
        forecasts: Dict of rule_name -> scaled forecast series
        weights: Dict of rule_name -> weight. Default: equal weight.
        fdm: Forecast Diversification Multiplier (>= 1.0)
        cap: Forecast cap

    Returns:
        Combined forecast series, capped at +/-cap
    """
    if weights is None:
        weights = {name: 1.0 / len(forecasts) for name in forecasts}

    # Weighted sum
    index = list(forecasts.values())[0].index
    combined = pd.Series(0.0, index=index)
    for name, series in forecasts.items():
        w = weights.get(name, 0.0)
        combined = combined + w * series.reindex(index, fill_value=0.0)

    # Apply FDM
    combined = combined * fdm

    # Re-cap
    return combined.clip(-cap, cap)


def calculate_fdm(
    forecast_correlation: pd.DataFrame,
    weights: pd.Series,
) -> float:
    """Calculate Forecast Diversification Multiplier.

    FDM = 1 / sqrt(w' * C * w) where w = weights, C = correlation matrix.
    Capped at 2.5 (Carver's recommendation).

    Args:
        forecast_correlation: Correlation matrix between forecasts
        weights: Forecast weights (must sum to 1)

    Returns:
        FDM, typically between 1.0 and 2.5
    """
    weights = weights / weights.sum()
    weighted_corr = float(weights @ forecast_correlation @ weights)

    if weighted_corr <= 0:
        return 1.0

    fdm = 1.0 / np.sqrt(weighted_corr)
    return min(fdm, 2.5)
