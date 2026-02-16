"""Momentum (rate of change) trading rule.

Simple price momentum: if price has risen over lookback period, go long.
Different from EWMAC in that it uses raw returns rather than EMA crossover.

Adapted from crypto_momo/momo/signals.py momentum logic.
"""

import numpy as np
import pandas as pd

from .base import AbstractTradingRule


class MomentumRule(AbstractTradingRule):
    """Price momentum rule.

    Raw forecast = return over lookback period, normalized by volatility.

    This captures the "time-series momentum" effect documented in
    academic literature (Moskowitz, Ooi, Pedersen 2012).
    """

    def __init__(self, lookback: int = 252):
        """Initialize momentum rule.

        Args:
            lookback: Bars to measure return over.
                     252 = ~10.5 days for hourly bars (~2 weeks)
                     For daily bars this would be ~1 year.
        """
        super().__init__(name=f"momentum_{lookback}", lookback=lookback)
        self.lookback = lookback

    def calculate_raw_forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        """Calculate raw momentum forecast.

        Raw forecast = lookback_return / vol

        This normalization by vol makes the signal comparable to EWMAC
        and other rules.

        Args:
            prices: Close price series

        Returns:
            Raw forecast (vol-normalized momentum)
        """
        # Return over lookback period
        lookback_return = prices.pct_change(self.lookback)

        # Volatility normalization (EWMA vol over same window)
        returns = prices.pct_change()
        vol = returns.ewm(span=min(self.lookback, 25)).std()

        # Avoid division by zero
        vol = vol.replace(0, np.nan)

        # Risk-adjusted momentum
        raw_forecast = lookback_return / vol

        return raw_forecast.dropna()
