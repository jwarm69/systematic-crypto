"""EWMAC (Exponentially Weighted Moving Average Crossover) trading rule.

This is Carver's primary and most recommended trading rule.
The raw forecast is the difference between fast and slow EMA,
normalized by volatility.
"""

import numpy as np
import pandas as pd

from .base import AbstractTradingRule


class EWMACRule(AbstractTradingRule):
    """EWMAC trading rule.

    Raw forecast = (EMA_fast - EMA_slow) / daily_vol

    The normalization by volatility makes the signal comparable across
    instruments and time periods with different volatility levels.

    Carver's recommended speed variants:
        EWMAC(2,8)    - very fast
        EWMAC(4,16)   - fast
        EWMAC(8,32)   - medium-fast
        EWMAC(16,64)  - medium (DEFAULT, good starting point)
        EWMAC(32,128) - slow
        EWMAC(64,256) - very slow
    """

    def __init__(self, fast_span: int = 16, slow_span: int = 64, vol_span: int = 25):
        """Initialize EWMAC rule.

        Args:
            fast_span: Fast EMA span
            slow_span: Slow EMA span (typically 4x fast)
            vol_span: EWMA volatility span for normalization
        """
        super().__init__(
            name=f"ewmac_{fast_span}_{slow_span}",
            fast_span=fast_span,
            slow_span=slow_span,
            vol_span=vol_span,
        )
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.vol_span = vol_span

    def calculate_raw_forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        """Calculate raw EWMAC forecast.

        Raw forecast = (EMA_fast - EMA_slow) / vol_price

        Where vol_price is the EWMA standard deviation of price changes
        (not returns), so the forecast is in "number of standard deviations" units.

        Args:
            prices: Close price series

        Returns:
            Raw forecast (unscaled)
        """
        fast_ema = prices.ewm(span=self.fast_span).mean()
        slow_ema = prices.ewm(span=self.slow_span).mean()

        # Volatility in price units (std of price changes, not returns)
        price_changes = prices.diff()
        vol_price = price_changes.ewm(span=self.vol_span).std()

        # Avoid division by zero
        vol_price = vol_price.replace(0, np.nan)

        raw_forecast = (fast_ema - slow_ema) / vol_price

        return raw_forecast.dropna()
