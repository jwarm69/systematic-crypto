"""Donchian channel breakout rule.

Carver's second-favorite trading rule after EWMAC. The idea is simple:
if price breaks above the N-period high, go long. If it breaks below
the N-period low, go short. The forecast is continuous, not binary.
"""

import numpy as np
import pandas as pd

from .base import AbstractTradingRule


class BreakoutRule(AbstractTradingRule):
    """Donchian channel breakout.

    Raw forecast = (price - channel_midpoint) / (channel_width / 2)

    This gives a continuous signal:
        +1 when price is at the upper channel
        -1 when price is at the lower channel
         0 when price is at the midpoint

    The normalization by channel width makes it volatility-adjusted.
    """

    def __init__(self, lookback: int = 20):
        """Initialize breakout rule.

        Args:
            lookback: Number of bars for Donchian channel.
                     20 = ~1 day for hourly bars (short-term breakout)
                     80 = ~3 days (medium-term)
                     250 = ~10 days (long-term)
        """
        super().__init__(name=f"breakout_{lookback}", lookback=lookback)
        self.lookback = lookback

    def calculate_raw_forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        """Calculate raw breakout forecast.

        Args:
            prices: Close price series

        Returns:
            Raw forecast in range [-1, +1] (before Carver scaling)
        """
        rolling_max = prices.rolling(self.lookback).max()
        rolling_min = prices.rolling(self.lookback).min()

        channel_width = rolling_max - rolling_min
        channel_mid = (rolling_max + rolling_min) / 2

        # Avoid division by zero (flat market)
        channel_width = channel_width.replace(0, np.nan)

        # Continuous breakout signal: where is price in the channel?
        raw_forecast = (prices - channel_mid) / (channel_width / 2)

        return raw_forecast.dropna()
