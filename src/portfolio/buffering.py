"""Position buffering to reduce turnover (Carver's no-trade buffer).

Adapted from crypto_momo/momo/sizing.py:510-557.
"""

import numpy as np

from .position_sizing import carver_position_size


def apply_buffer(
    current_position: float,
    target_position: float,
    capital: float,
    price: float,
    instrument_vol: float,
    vol_target: float = 0.12,
    buffer_fraction: float = 0.10,
) -> float:
    """Apply Carver's no-trade buffer to reduce turnover.

    Only trade if the difference between current and target exceeds
    a threshold (fraction of the average position size).

    This significantly reduces transaction costs with minimal
    impact on performance.

    Args:
        current_position: Current position in units
        target_position: Calculated target position
        capital: Account capital
        price: Current price
        instrument_vol: Annualized volatility
        vol_target: Target volatility
        buffer_fraction: Fraction of avg position for no-trade zone

    Returns:
        Buffered position (either current or target)
    """
    # Average position at neutral forecast (forecast=10)
    avg_position = abs(carver_position_size(
        capital=capital,
        price=price,
        instrument_vol=instrument_vol,
        forecast=10.0,
        vol_target=vol_target,
    ))

    if avg_position == 0:
        return target_position

    # Buffer zone
    buffer = avg_position * buffer_fraction

    # Lower and upper bounds around target
    lower_bound = target_position - buffer
    upper_bound = target_position + buffer

    # If current position is within buffer of target, keep current
    if lower_bound <= current_position <= upper_bound:
        return current_position

    # If outside buffer, move to nearest edge of buffer
    if current_position < lower_bound:
        return lower_bound
    else:
        return upper_bound
