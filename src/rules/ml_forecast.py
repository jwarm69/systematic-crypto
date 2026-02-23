"""ML-based trading rule: wraps an ML ensemble as a Carver forecast.

Converts probability output to Carver forecast:
    raw_forecast = (prob_up - 0.5) * 2 * 20  # Maps [0,1] to [-20, +20]

The ML rule is "just another trading rule" in the Carver framework.
It gets 10-20% weight max and must pass marginal value testing.
"""

import logging

import numpy as np
import pandas as pd

from .base import AbstractTradingRule
from ..indicators.technical import add_target, build_features, get_feature_columns
from ..ml.trainer import EnsembleResult, ensemble_predict_proba, train_with_cv

logger = logging.getLogger(__name__)


class MLForecastRule(AbstractTradingRule):
    """ML ensemble as a Carver trading rule.

    Trains on historical data and generates probability-based forecasts.
    The forecast is scaled to [-20, +20] based on model confidence.

    Usage:
        1. Call train() with OHLCV data to fit the ensemble
        2. Call forecast() with close prices to generate forecasts
           (requires self._ohlcv_data to be set for feature computation)
    """

    name = "ml_forecast"

    def __init__(
        self,
        n_cv_splits: int = 5,
        n_optuna_trials: int = 30,
        feature_selection_k: int = 25,
        purge: int = 24,
        embargo: int = 12,
    ):
        self.n_cv_splits = n_cv_splits
        self.n_optuna_trials = n_optuna_trials
        self.feature_selection_k = feature_selection_k
        self.purge = purge
        self.embargo = embargo

        self.ensemble: EnsembleResult | None = None
        self._ohlcv_data: pd.DataFrame | None = None

    def train(self, ohlcv_df: pd.DataFrame) -> EnsembleResult:
        """Train the ML ensemble on OHLCV data.

        Args:
            ohlcv_df: DataFrame with open, high, low, close, volume

        Returns:
            EnsembleResult with trained models
        """
        logger.info(f"Training ML forecast on {len(ohlcv_df)} bars...")

        # Build features and target
        df = build_features(ohlcv_df)
        df = add_target(df, lookahead=1)
        df = df.dropna(subset=["target"])

        feature_cols = get_feature_columns(df)
        # Drop any feature columns that are all NaN
        valid_cols = [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.5]

        X = df[valid_cols]
        y = df["target"]

        logger.info(f"  Features: {len(valid_cols)}, Samples: {len(X)}")

        self.ensemble = train_with_cv(
            X=X,
            y=y,
            features=None,  # Auto-select
            n_cv_splits=self.n_cv_splits,
            purge=self.purge,
            embargo=self.embargo,
            n_optuna_trials=self.n_optuna_trials,
            feature_selection_k=self.feature_selection_k,
        )
        self._ohlcv_data = ohlcv_df

        logger.info(
            f"  ML ensemble trained: AUC={self.ensemble.test_auc:.4f}, "
            f"Acc={self.ensemble.test_accuracy:.1%}"
        )
        return self.ensemble

    def set_ohlcv(self, ohlcv_df: pd.DataFrame) -> None:
        """Set OHLCV data for forecast generation (when not re-training)."""
        self._ohlcv_data = ohlcv_df

    def calculate_raw_forecast(self, prices: pd.Series) -> pd.Series:
        """Generate raw forecast from ML probability.

        Maps probability [0, 1] to raw forecast [-20, +20]:
            raw = (prob - 0.5) * 2 * 20

        At prob=0.5 (no edge): forecast = 0
        At prob=0.6: forecast = +4
        At prob=0.7: forecast = +8
        At prob=1.0: forecast = +20
        """
        if self.ensemble is None:
            raise RuntimeError("ML model not trained. Call train() first.")

        if self._ohlcv_data is None:
            raise RuntimeError("OHLCV data not set. Call set_ohlcv() or train() first.")

        # Build features from OHLCV
        df = build_features(self._ohlcv_data)
        feature_cols = self.ensemble.features

        # Align with price index
        common_idx = prices.index.intersection(df.index)
        if len(common_idx) == 0:
            return pd.Series(0.0, index=prices.index)

        X = df.loc[common_idx, feature_cols].fillna(0)
        probas = ensemble_predict_proba(self.ensemble, X)

        # Convert probability to forecast
        raw_forecast = (probas - 0.5) * 2 * 20

        result = pd.Series(raw_forecast, index=common_idx)
        # Reindex to full price index, forward-fill
        result = result.reindex(prices.index).ffill().fillna(0)

        return result

    def forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        """Generate scaled forecast.

        For ML rules, we skip the standard scale_forecast() because
        the raw forecast is already in the [-20, +20] range by construction.
        We still cap at +-20.
        """
        raw = self.calculate_raw_forecast(prices)
        return raw.clip(-20, 20)
