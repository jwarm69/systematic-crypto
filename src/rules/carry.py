"""Carry trading rule based on funding rates.

In crypto perpetual futures, the funding rate is a periodic payment
between longs and shorts. When funding is positive, longs pay shorts
(bullish sentiment = crowded long). Negative funding means shorts pay longs.

Carver's carry rule: negative funding = bullish carry (go long),
positive funding = bearish carry (go short). We're paid to hold the
position in the direction of the carry.
"""

import numpy as np
import pandas as pd

from .base import AbstractTradingRule


class CarryRule(AbstractTradingRule):
    """Carry rule from funding rates.

    Raw forecast = -smoothed_funding_rate (normalized by vol)

    Negative funding -> positive carry for longs -> positive forecast
    Positive funding -> positive carry for shorts -> negative forecast
    """

    def __init__(self, smooth_days: int = 90):
        """Initialize carry rule.

        Args:
            smooth_days: EWMA span for smoothing funding rates.
                        Longer = more stable signal, shorter = more responsive.
                        90 days is Carver's recommendation for carry signals.
        """
        super().__init__(name=f"carry_{smooth_days}", smooth_days=smooth_days)
        self.smooth_days = smooth_days

    def calculate_raw_forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        """Calculate raw carry forecast from funding rates.

        Args:
            prices: Close price series (used for alignment)
            funding_rates: Series of funding rates (typically 8-hourly).
                          Must be passed as keyword arg.

        Returns:
            Raw forecast (unscaled). Positive = go long.
        """
        funding_rates = kwargs.get("funding_rates")
        if funding_rates is None:
            # No funding data -> zero forecast
            return pd.Series(0.0, index=prices.index)

        # Align to price index
        funding = funding_rates.reindex(prices.index, method="ffill")

        # Annualize: funding is per-period (8h on Hyperliquid = 3x daily = 1095x yearly)
        # But we don't need to annualize for the forecast, just smooth and negate
        smoothed = funding.ewm(span=self.smooth_days).mean()

        # Negate: negative funding = bullish carry
        # Normalize by rolling std of funding to make signal scale-independent
        funding_vol = funding.rolling(self.smooth_days, min_periods=20).std()
        funding_vol = funding_vol.replace(0, np.nan)

        raw_forecast = -smoothed / funding_vol
        return raw_forecast.dropna()
