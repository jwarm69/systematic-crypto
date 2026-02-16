"""Vectorized backtesting engine with cost-adjusted performance metrics.

Adapted from crypto_momo/momo/backtest.py with Carver-specific enhancements:
- Forecast-based position sizing (not raw signals)
- Cost-adjusted Sharpe ratio
- Proper accounting of Hyperliquid fees
"""

import logging

import numpy as np
import pandas as pd

from ..indicators.volatility import ewma_volatility
from ..portfolio.position_sizing import carver_position_size
from ..portfolio.buffering import apply_buffer

logger = logging.getLogger(__name__)


def periods_per_year_from_timeframe(timeframe: str) -> float:
    """Approximate number of bars per year for common timeframes."""
    mapping = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "30m": 365 * 24 * 2,
        "1h": 365 * 24,
        "4h": 365 * 6,
        "1d": 365,
    }
    return float(mapping.get(timeframe, 365 * 24))


def timeframe_hours(timeframe: str) -> float:
    """Return bar duration in hours for common timeframes."""
    mapping = {
        "1m": 1.0 / 60.0,
        "5m": 5.0 / 60.0,
        "15m": 15.0 / 60.0,
        "30m": 30.0 / 60.0,
        "1h": 1.0,
        "4h": 4.0,
        "1d": 24.0,
    }
    return float(mapping.get(timeframe, 1.0))


class BacktestEngine:
    """Vectorized backtest engine for Carver-style systematic trading."""

    def __init__(
        self,
        capital: float = 5000,
        vol_target: float = 0.12,
        maker_fee_bps: float = 2.0,
        taker_fee_bps: float = 5.0,
        slippage_bps: float = 2.0,
        buffer_fraction: float = 0.10,
        max_leverage: float = 2.0,
        timeframe: str = "1h",
        bars_per_year: float | None = None,
        funding_bps_per_8h: float = 0.0,
    ):
        self.capital = capital
        self.vol_target = vol_target
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.slippage_bps = slippage_bps
        self.buffer_fraction = buffer_fraction
        self.max_leverage = max_leverage
        self.timeframe = timeframe
        self.bars_per_year = bars_per_year or periods_per_year_from_timeframe(timeframe)
        self.annualization = np.sqrt(self.bars_per_year)
        self.funding_bps_per_8h = funding_bps_per_8h

    def run(
        self,
        prices: pd.Series,
        forecasts: pd.Series,
        vol_span: int = 25,
        funding_rates: pd.Series | None = None,
    ) -> dict:
        """Run backtest for a single instrument.

        Args:
            prices: Close price series (DatetimeIndex)
            forecasts: Forecast series scaled to [-20, +20]
            vol_span: EWMA vol span
            funding_rates: Optional per-bar funding rates (decimal, signed)

        Returns:
            Dict with portfolio_value, positions, metrics, etc.
        """
        # Align data
        common_idx = prices.index.intersection(forecasts.index)
        prices = prices.loc[common_idx]
        forecasts = forecasts.loc[common_idx]

        # Calculate volatility
        vol = ewma_volatility(prices, span=vol_span, annualization=self.annualization)

        # Calculate target positions
        target_positions = pd.Series(0.0, index=common_idx)
        for i, dt in enumerate(common_idx):
            v = vol.iloc[i]
            if pd.isna(v) or v <= 0:
                continue
            target_positions.iloc[i] = carver_position_size(
                capital=self.capital,
                price=prices.iloc[i],
                instrument_vol=v,
                forecast=forecasts.iloc[i],
                vol_target=self.vol_target,
                max_leverage=self.max_leverage,
            )

        # Apply buffering
        positions = pd.Series(0.0, index=common_idx)
        for i, dt in enumerate(common_idx):
            v = vol.iloc[i]
            if pd.isna(v) or v <= 0:
                positions.iloc[i] = 0
                continue
            if i == 0:
                positions.iloc[i] = target_positions.iloc[i]
            else:
                positions.iloc[i] = apply_buffer(
                    current_position=positions.iloc[i - 1],
                    target_position=target_positions.iloc[i],
                    capital=self.capital,
                    price=prices.iloc[i],
                    instrument_vol=v,
                    vol_target=self.vol_target,
                    buffer_fraction=self.buffer_fraction,
                )

        # Calculate returns and PnL
        price_returns = prices.pct_change()

        # Gross PnL: position(t-1) * return(t) * price(t-1)
        gross_pnl = positions.shift(1) * price_returns * prices.shift(1)
        gross_pnl = gross_pnl.fillna(0)

        # Transaction costs
        position_changes = positions.diff().fillna(positions.iloc[0])
        trade_notional = position_changes.abs() * prices
        cost_bps = self.taker_fee_bps + self.slippage_bps
        transaction_costs = trade_notional * (cost_bps / 10000)

        # Funding PnL
        if funding_rates is not None:
            aligned_funding = funding_rates.reindex(common_idx).fillna(0.0)
            # Positive funding means longs pay, shorts receive.
            funding_pnl = -(positions.shift(1) * prices.shift(1) * aligned_funding).fillna(0.0)
        else:
            funding_rate_per_bar = (self.funding_bps_per_8h / 10000.0) * (
                timeframe_hours(self.timeframe) / 8.0
            )
            funding_pnl = -(positions.shift(1).abs() * prices.shift(1) * funding_rate_per_bar).fillna(0.0)

        # Net PnL
        net_pnl = gross_pnl - transaction_costs + funding_pnl

        # Portfolio value
        portfolio_value = self.capital + net_pnl.cumsum()

        # Metrics
        metrics = self._compute_metrics(
            portfolio_value,
            net_pnl,
            positions,
            transaction_costs,
            funding_pnl,
        )

        return {
            "portfolio_value": portfolio_value,
            "positions": positions,
            "target_positions": target_positions,
            "forecasts": forecasts,
            "volatility": vol,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "transaction_costs": transaction_costs,
            "funding_pnl": funding_pnl,
            "metrics": metrics,
        }

    def _compute_metrics(
        self,
        portfolio_value: pd.Series,
        net_pnl: pd.Series,
        positions: pd.Series,
        transaction_costs: pd.Series,
        funding_pnl: pd.Series,
    ) -> dict:
        """Compute comprehensive performance metrics."""
        returns = portfolio_value.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0}

        # Annualize based on configured timeframe
        bars_per_year = self.bars_per_year
        ann_factor = self.annualization

        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        ann_return = (1 + total_return) ** (bars_per_year / len(returns)) - 1
        ann_vol = returns.std() * ann_factor
        sharpe = (returns.mean() / returns.std()) * ann_factor if returns.std() > 0 else 0

        # Drawdown
        cummax = portfolio_value.expanding().max()
        drawdown = (portfolio_value - cummax) / cummax
        max_drawdown = drawdown.min()

        # Turnover
        position_changes = positions.diff().abs()
        avg_turnover = position_changes.mean()
        total_costs = transaction_costs.sum()
        total_funding_pnl = funding_pnl.sum()

        # Win rate (per bar)
        win_rate = (net_pnl > 0).sum() / (net_pnl != 0).sum() if (net_pnl != 0).sum() > 0 else 0

        # Profit factor
        gains = net_pnl[net_pnl > 0].sum()
        losses = abs(net_pnl[net_pnl < 0].sum())
        profit_factor = gains / losses if losses > 0 else float("inf")

        # Sortino
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0
        sortino = (returns.mean() / downside_std * ann_factor) if downside_std > 0 else 0

        # Calmar
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_turnover": avg_turnover,
            "total_costs": total_costs,
            "total_funding_pnl": total_funding_pnl,
            "final_value": portfolio_value.iloc[-1],
            "n_bars": len(returns),
            "bars_per_year": bars_per_year,
        }
