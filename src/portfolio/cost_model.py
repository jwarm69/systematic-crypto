"""Cost model for Hyperliquid trading.

Determines whether a trade is worthwhile after accounting for
maker/taker fees, slippage, and market impact.
"""

import numpy as np


def trade_cost_bps(
    is_maker: bool = False,
    maker_fee_bps: float = 2.0,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> float:
    """Calculate total cost of a round-trip trade in basis points.

    A round-trip includes entry + exit costs.

    Args:
        is_maker: Whether order is a maker (limit) or taker (market)
        maker_fee_bps: Maker fee in bps
        taker_fee_bps: Taker fee in bps
        slippage_bps: Estimated slippage per side in bps

    Returns:
        Total round-trip cost in bps
    """
    fee = maker_fee_bps if is_maker else taker_fee_bps
    # Round trip: 2x (entry fee + slippage)
    return 2 * (fee + slippage_bps)


def is_trade_worthwhile(
    current_forecast: float,
    new_forecast: float,
    instrument_vol: float,
    annual_turnover_per_forecast: float = 6.6,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    target_sharpe: float = 0.3,
) -> bool:
    """Determine if changing position is worthwhile after costs.

    Carver's speed limit: only trade if the expected improvement in
    forecast quality exceeds the cost of trading.

    The key insight: each forecast unit traded costs a fraction of Sharpe ratio.
    Cost in SR terms = (turnover * cost_per_trade) / vol_target.

    Args:
        current_forecast: Current forecast value
        new_forecast: New forecast value
        instrument_vol: Annualized volatility
        annual_turnover_per_forecast: How many times per year a forecast unit turns over
        taker_fee_bps: Taker fee
        slippage_bps: Slippage
        target_sharpe: Minimum Sharpe improvement to justify trade

    Returns:
        True if the trade is likely worthwhile
    """
    forecast_change = abs(new_forecast - current_forecast)

    # Cost of changing forecast by this amount, annualized
    cost_per_unit = (taker_fee_bps + slippage_bps) / 10000
    annual_cost = forecast_change * annual_turnover_per_forecast * cost_per_unit

    # Expected benefit (very rough): forecast improvement contributes to SR
    # A forecast change of 10 (average) should deliver target_sharpe
    expected_sr_improvement = (forecast_change / 10.0) * target_sharpe

    # Cost in SR terms
    cost_in_sr = annual_cost / instrument_vol if instrument_vol > 0 else float('inf')

    return expected_sr_improvement > cost_in_sr
