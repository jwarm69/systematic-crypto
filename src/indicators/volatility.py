"""Volatility estimation for position sizing and forecast normalization.

Implements EWMA volatility as recommended by Robert Carver.
"""

import numpy as np
import pandas as pd


def ewma_volatility(
    prices: pd.Series,
    span: int = 25,
    annualization: float = np.sqrt(365 * 24),
    min_periods: int = 10,
) -> pd.Series:
    """Calculate EWMA volatility of returns, annualized.

    Carver recommends exponentially weighted volatility (more responsive
    to recent conditions than simple rolling std).

    Args:
        prices: Price series (close prices)
        span: EWMA span in bars (25 = ~1 day for hourly data)
        annualization: Factor to annualize. sqrt(365*24) for hourly crypto bars.
        min_periods: Minimum observations before producing a value

    Returns:
        Annualized volatility series
    """
    returns = prices.pct_change()
    ewm_vol = returns.ewm(span=span, min_periods=min_periods).std()
    return ewm_vol * annualization


def current_volatility(
    prices: pd.Series,
    span: int = 25,
    annualization: float = np.sqrt(365 * 24),
) -> float:
    """Get the most recent volatility estimate.

    Args:
        prices: Price series
        span: EWMA span
        annualization: Annualization factor

    Returns:
        Current annualized volatility as a float (e.g., 0.60 for 60%)
    """
    vol_series = ewma_volatility(prices, span=span, annualization=annualization)
    return vol_series.iloc[-1]


def returns_volatility(
    returns: pd.Series,
    span: int = 25,
    annualization: float = np.sqrt(365 * 24),
) -> pd.Series:
    """Calculate EWMA volatility from a returns series directly.

    Args:
        returns: Returns series (percentage, not log)
        span: EWMA span
        annualization: Annualization factor

    Returns:
        Annualized volatility series
    """
    return returns.ewm(span=span).std() * annualization
