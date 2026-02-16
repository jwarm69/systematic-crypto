"""Abstract base class for trading rules.

Every trading rule outputs a "forecast" scaled to [-20, +20] with
target average absolute value of 10 (Carver's standard).
"""

from abc import ABC, abstractmethod

import pandas as pd


class AbstractTradingRule(ABC):
    """Base class for all trading rules.

    A trading rule takes price data (and optionally other data) and
    produces a forecast: a number from -20 to +20 indicating the
    strength and direction of the signal.

    Carver's convention:
        +20 = maximum long conviction
        +10 = average long conviction
          0 = no position
        -10 = average short conviction
        -20 = maximum short conviction
    """

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    def calculate_raw_forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        """Calculate raw (unscaled) forecast.

        Args:
            prices: Close price series
            **kwargs: Additional data (funding rates, volume, etc.)

        Returns:
            Raw forecast series (any scale)
        """
        ...

    def forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        """Calculate scaled and capped forecast.

        Calls calculate_raw_forecast, then scales to Carver's [-20, +20].

        Args:
            prices: Close price series
            **kwargs: Additional data

        Returns:
            Forecast series scaled to [-20, +20], avg abs ~10
        """
        from .scaling import scale_forecast

        raw = self.calculate_raw_forecast(prices, **kwargs)
        return scale_forecast(raw)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({self.name}, {params_str})"
