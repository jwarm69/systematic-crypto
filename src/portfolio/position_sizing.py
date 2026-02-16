"""Carver-style position sizing.

Core formula:
    position = (capital * vol_target * IDM * instrument_weight * forecast)
               / (instrument_vol * price * 10)

Adapted from crypto_momo/momo/sizing.py.
"""

import numpy as np


def carver_position_size(
    capital: float,
    price: float,
    instrument_vol: float,
    forecast: float,
    vol_target: float = 0.12,
    idm: float = 1.0,
    instrument_weight: float = 1.0,
    forecast_scalar: float = 10.0,
    max_leverage: float = 2.0,
) -> float:
    """Calculate position size using Carver's volatility-targeted method.

    Args:
        capital: Account capital in USD
        price: Current instrument price
        instrument_vol: Annualized volatility (e.g., 0.60 for 60%)
        forecast: Scaled forecast from -20 to +20
        vol_target: Target portfolio volatility (e.g., 0.12 for 12%)
        idm: Instrument Diversification Multiplier (1.0 for single instrument)
        instrument_weight: Weight of this instrument in portfolio (1.0 for single)
        forecast_scalar: Average absolute forecast (always 10 for Carver)
        max_leverage: Maximum allowed leverage

    Returns:
        Position size in instrument units (positive=long, negative=short).
        For BTC, this is in BTC units. For Hyperliquid, multiply by price for notional.
    """
    if instrument_vol <= 0 or price <= 0:
        return 0.0

    # Notional exposure for a "neutral" (forecast=10) position
    # This gives us vol_target contribution to portfolio vol
    target_notional = (capital * vol_target * idm * instrument_weight) / instrument_vol

    # Scale by forecast strength (forecast=10 -> 1x, forecast=20 -> 2x)
    scaled_notional = target_notional * (forecast / forecast_scalar)

    # Convert to units
    position_units = scaled_notional / price

    # Apply leverage cap
    max_position = (capital * max_leverage) / price
    position_units = np.clip(position_units, -max_position, max_position)

    return float(position_units)


def minimum_capital_for_instrument(
    price: float,
    instrument_vol: float,
    vol_target: float = 0.12,
    min_order_size: float = 0.001,
) -> float:
    """Calculate minimum capital needed to trade an instrument.

    At forecast=10, the position must be >= min_order_size.

    Args:
        price: Current price
        instrument_vol: Annualized volatility
        vol_target: Target vol
        min_order_size: Exchange minimum order (e.g., 0.001 BTC)

    Returns:
        Minimum capital in USD
    """
    # position = (capital * vol_target) / (vol * price * 10) * 10
    # -> capital = min_order_size * instrument_vol * price / vol_target
    return min_order_size * instrument_vol * price / vol_target
