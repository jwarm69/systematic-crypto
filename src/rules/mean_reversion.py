"""Mean reversion trading rule using z-score.

Adapted from crypto-stat-arb/src/kalman_filter.py z-score logic.
This is a contrarian rule: buy when price is below its recent average,
sell when above. Works best in ranging markets.

Note: In Carver's framework, this gets combined with trend-following
rules. The combination works because they're negatively correlated.
"""

import numpy as np
import pandas as pd

from .base import AbstractTradingRule


class MeanReversionRule(AbstractTradingRule):
    """Z-score mean reversion rule.

    Raw forecast = -z_score (negated because we trade against the move)

    When z > 0 (price above mean) -> negative forecast (sell)
    When z < 0 (price below mean) -> positive forecast (buy)
    """

    def __init__(self, lookback: int = 20, z_cap: float = 3.0):
        """Initialize mean reversion rule.

        Args:
            lookback: Rolling window for mean/std calculation (in bars).
                     20 = ~1 day for hourly bars.
            z_cap: Cap raw z-score at this level to limit extreme values.
        """
        super().__init__(
            name=f"mean_rev_{lookback}",
            lookback=lookback,
            z_cap=z_cap,
        )
        self.lookback = lookback
        self.z_cap = z_cap

    def calculate_raw_forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        """Calculate raw mean reversion forecast.

        Args:
            prices: Close price series

        Returns:
            Raw forecast: -z_score, capped at +/-z_cap
        """
        rolling_mean = prices.rolling(self.lookback).mean()
        rolling_std = prices.rolling(self.lookback).std()

        # Volatility floor to prevent extreme z-scores
        vol_floor = prices.rolling(self.lookback * 5, min_periods=self.lookback).std() * 0.1
        rolling_std = rolling_std.clip(lower=vol_floor)

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        z_score = (prices - rolling_mean) / rolling_std

        # Cap z-score
        z_score = z_score.clip(-self.z_cap, self.z_cap)

        # Negate: high z -> sell, low z -> buy
        raw_forecast = -z_score

        return raw_forecast.dropna()
