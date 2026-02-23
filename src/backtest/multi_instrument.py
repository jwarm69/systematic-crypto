"""Multi-instrument backtesting engine.

Runs per-instrument backtests with proper IDM, instrument weights,
and aggregates into a portfolio-level equity curve.
"""

import logging

import numpy as np
import pandas as pd

from ..indicators.volatility import ewma_volatility
from ..portfolio.buffering import apply_buffer
from ..portfolio.instrument_weights import (
    calculate_idm,
    compute_instrument_correlation,
    equal_weights,
)
from ..portfolio.position_sizing import carver_position_size
from .engine import BacktestEngine, periods_per_year_from_timeframe

logger = logging.getLogger(__name__)


class MultiInstrumentBacktest:
    """Portfolio-level backtest across multiple instruments.

    Runs the Carver framework with IDM and instrument weights:
        position = (capital * vol_target * IDM * instrument_weight * forecast)
                   / (instrument_vol * price * 10)
    """

    def __init__(
        self,
        capital: float = 5000,
        vol_target: float = 0.12,
        maker_fee_bps: float = 2.0,
        taker_fee_bps: float = 5.0,
        slippage_bps: float = 2.0,
        buffer_fraction: float = 0.10,
        max_leverage: float = 5.0,
        timeframe: str = "1h",
    ):
        self.capital = capital
        self.vol_target = vol_target
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.slippage_bps = slippage_bps
        self.buffer_fraction = buffer_fraction
        self.max_leverage = max_leverage
        self.timeframe = timeframe
        self.bars_per_year = periods_per_year_from_timeframe(timeframe)
        self.annualization = np.sqrt(self.bars_per_year)

    def run(
        self,
        prices_dict: dict[str, pd.Series],
        forecasts_dict: dict[str, pd.Series],
        instrument_weights: dict[str, float] | None = None,
        idm: float | None = None,
        vol_span: int = 25,
    ) -> dict:
        """Run multi-instrument backtest.

        Args:
            prices_dict: instrument_name -> close price Series
            forecasts_dict: instrument_name -> forecast Series [-20, +20]
            instrument_weights: Pre-computed weights (default: equal)
            idm: Pre-computed IDM (default: computed from returns)
            vol_span: EWMA volatility span

        Returns:
            Dict with portfolio_value, per-instrument results, metrics
        """
        instruments = list(prices_dict.keys())
        n_instruments = len(instruments)

        if n_instruments == 0:
            return {"metrics": {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0}}

        # Compute weights and IDM if not provided
        if instrument_weights is None:
            instrument_weights = equal_weights(instruments)

        if idm is None:
            returns_dict = {
                name: prices.pct_change().dropna()
                for name, prices in prices_dict.items()
            }
            corr = compute_instrument_correlation(returns_dict)
            idm = calculate_idm(instrument_weights, corr)

        logger.info(
            f"Multi-instrument backtest: {n_instruments} instruments, "
            f"IDM={idm:.3f}, weights={instrument_weights}"
        )

        # Run per-instrument position sizing and PnL
        instrument_results = {}
        all_pnl = {}
        all_positions = {}
        all_costs = {}

        for name in instruments:
            prices = prices_dict[name]
            forecasts = forecasts_dict[name]
            weight = instrument_weights.get(name, 1.0 / n_instruments)

            result = self._run_single_instrument(
                prices=prices,
                forecasts=forecasts,
                instrument_weight=weight,
                idm=idm,
                vol_span=vol_span,
            )
            instrument_results[name] = result
            all_pnl[name] = result["net_pnl"]
            all_positions[name] = result["positions"]
            all_costs[name] = result["transaction_costs"]

        # Aggregate portfolio PnL
        pnl_df = pd.DataFrame(all_pnl)
        pnl_df = pnl_df.fillna(0)
        portfolio_pnl = pnl_df.sum(axis=1)
        portfolio_value = self.capital + portfolio_pnl.cumsum()

        # Portfolio-level metrics
        metrics = self._compute_portfolio_metrics(
            portfolio_value=portfolio_value,
            portfolio_pnl=portfolio_pnl,
            pnl_df=pnl_df,
            all_positions=all_positions,
            prices_dict=prices_dict,
            all_costs=all_costs,
        )

        return {
            "portfolio_value": portfolio_value,
            "portfolio_pnl": portfolio_pnl,
            "instrument_results": instrument_results,
            "instrument_weights": instrument_weights,
            "idm": idm,
            "metrics": metrics,
        }

    def _run_single_instrument(
        self,
        prices: pd.Series,
        forecasts: pd.Series,
        instrument_weight: float,
        idm: float,
        vol_span: int,
    ) -> dict:
        """Run backtest for a single instrument within the multi-instrument framework."""
        # Align data
        common_idx = prices.index.intersection(forecasts.index)
        prices = prices.loc[common_idx]
        forecasts = forecasts.loc[common_idx]

        vol = ewma_volatility(prices, span=vol_span, annualization=self.annualization)

        # Calculate target positions with IDM and instrument weight
        target_positions = pd.Series(0.0, index=common_idx)
        for i in range(len(common_idx)):
            v = vol.iloc[i]
            if pd.isna(v) or v <= 0:
                continue
            target_positions.iloc[i] = carver_position_size(
                capital=self.capital,
                price=prices.iloc[i],
                instrument_vol=v,
                forecast=forecasts.iloc[i],
                vol_target=self.vol_target,
                idm=idm,
                instrument_weight=instrument_weight,
                max_leverage=self.max_leverage,
            )

        # Apply buffering
        positions = pd.Series(0.0, index=common_idx)
        for i in range(len(common_idx)):
            v = vol.iloc[i]
            if pd.isna(v) or v <= 0:
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

        # PnL calculation
        price_returns = prices.pct_change()
        gross_pnl = (positions.shift(1) * price_returns * prices.shift(1)).fillna(0)

        # Transaction costs
        position_changes = positions.diff().fillna(positions.iloc[0])
        trade_notional = position_changes.abs() * prices
        cost_bps = self.taker_fee_bps + self.slippage_bps
        transaction_costs = trade_notional * (cost_bps / 10000)

        net_pnl = gross_pnl - transaction_costs

        return {
            "positions": positions,
            "target_positions": target_positions,
            "forecasts": forecasts,
            "volatility": vol,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "transaction_costs": transaction_costs,
        }

    def _compute_portfolio_metrics(
        self,
        portfolio_value: pd.Series,
        portfolio_pnl: pd.Series,
        pnl_df: pd.DataFrame,
        all_positions: dict[str, pd.Series],
        prices_dict: dict[str, pd.Series],
        all_costs: dict[str, pd.Series],
    ) -> dict:
        """Compute portfolio-level metrics."""
        returns = portfolio_value.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0}

        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        ann_return = (1 + total_return) ** (self.bars_per_year / len(returns)) - 1
        ann_vol = returns.std() * self.annualization
        sharpe = (returns.mean() / returns.std()) * self.annualization

        # Drawdown
        cummax = portfolio_value.expanding().max()
        drawdown = (portfolio_value - cummax) / cummax
        max_drawdown = drawdown.min()

        # Win rate
        nonzero = portfolio_pnl != 0
        win_rate = (portfolio_pnl > 0).sum() / nonzero.sum() if nonzero.sum() > 0 else 0

        # Profit factor
        gains = portfolio_pnl[portfolio_pnl > 0].sum()
        losses = abs(portfolio_pnl[portfolio_pnl < 0].sum())
        profit_factor = gains / losses if losses > 0 else float("inf")

        # Total costs
        total_costs = sum(c.sum() for c in all_costs.values())

        # Per-instrument contribution
        inst_contributions = {}
        for name in pnl_df.columns:
            inst_pnl = pnl_df[name]
            inst_contributions[name] = {
                "total_pnl": inst_pnl.sum(),
                "pnl_share": inst_pnl.sum() / portfolio_pnl.sum() if portfolio_pnl.sum() != 0 else 0,
            }

        # Sortino
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0
        sortino = (returns.mean() / downside_std * self.annualization) if downside_std > 0 else 0

        # Calmar
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Realized portfolio leverage
        leverage_series = []
        for dt in portfolio_value.index:
            total_notional = 0
            for name, pos in all_positions.items():
                if dt in pos.index and dt in prices_dict[name].index:
                    total_notional += abs(pos.loc[dt]) * prices_dict[name].loc[dt]
            equity = portfolio_value.loc[dt]
            leverage_series.append(total_notional / equity if equity > 0 else 0)

        avg_leverage = np.mean(leverage_series) if leverage_series else 0
        max_leverage = max(leverage_series) if leverage_series else 0

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
            "total_costs": total_costs,
            "final_value": portfolio_value.iloc[-1],
            "n_bars": len(returns),
            "n_instruments": len(pnl_df.columns),
            "avg_leverage": avg_leverage,
            "max_leverage_realized": max_leverage,
            "instrument_contributions": inst_contributions,
        }
