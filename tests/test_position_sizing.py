"""Tests for Carver position sizing."""

import numpy as np
import pytest

from src.portfolio.position_sizing import carver_position_size, minimum_capital_for_instrument
from src.portfolio.buffering import apply_buffer


class TestCarverPositionSize:
    """Test the core position sizing formula."""

    def test_basic_long_position(self):
        """Forecast of +10 should give a 'neutral' long position."""
        pos = carver_position_size(
            capital=10000,
            price=50000,
            instrument_vol=0.60,
            forecast=10,
            vol_target=0.12,
        )
        assert pos > 0
        # Notional = capital * vol_target / vol = 10000 * 0.12 / 0.60 = 2000
        # Position = 2000 / 50000 = 0.04 BTC
        assert abs(pos - 0.04) < 0.001

    def test_basic_short_position(self):
        """Negative forecast should give short position."""
        pos = carver_position_size(
            capital=10000,
            price=50000,
            instrument_vol=0.60,
            forecast=-10,
            vol_target=0.12,
        )
        assert pos < 0
        assert abs(pos + 0.04) < 0.001

    def test_forecast_scaling(self):
        """Forecast of +20 should give 2x the position of +10."""
        pos_10 = carver_position_size(
            capital=10000, price=50000, instrument_vol=0.60, forecast=10
        )
        pos_20 = carver_position_size(
            capital=10000, price=50000, instrument_vol=0.60, forecast=20
        )
        assert abs(pos_20 / pos_10 - 2.0) < 0.01

    def test_zero_forecast(self):
        """Zero forecast should give zero position."""
        pos = carver_position_size(
            capital=10000, price=50000, instrument_vol=0.60, forecast=0
        )
        assert pos == 0.0

    def test_higher_vol_smaller_position(self):
        """Higher volatility should result in smaller position."""
        pos_low_vol = carver_position_size(
            capital=10000, price=50000, instrument_vol=0.30, forecast=10
        )
        pos_high_vol = carver_position_size(
            capital=10000, price=50000, instrument_vol=0.90, forecast=10
        )
        assert abs(pos_low_vol) > abs(pos_high_vol)

    def test_leverage_cap(self):
        """Position should be capped by max leverage."""
        pos = carver_position_size(
            capital=1000,
            price=100,
            instrument_vol=0.01,  # Very low vol -> huge position
            forecast=20,
            max_leverage=2.0,
        )
        notional = abs(pos) * 100
        assert notional <= 1000 * 2.0 + 0.01  # Within leverage cap

    def test_zero_vol_returns_zero(self):
        pos = carver_position_size(capital=10000, price=50000, instrument_vol=0, forecast=10)
        assert pos == 0.0

    def test_zero_price_returns_zero(self):
        pos = carver_position_size(capital=10000, price=0, instrument_vol=0.5, forecast=10)
        assert pos == 0.0

    def test_idm_and_instrument_weight(self):
        """IDM and instrument weight should scale position proportionally."""
        pos_base = carver_position_size(
            capital=10000, price=50000, instrument_vol=0.60, forecast=10,
            idm=1.0, instrument_weight=1.0,
        )
        pos_multi = carver_position_size(
            capital=10000, price=50000, instrument_vol=0.60, forecast=10,
            idm=1.5, instrument_weight=0.5,
        )
        # IDM=1.5 * weight=0.5 = 0.75 of base
        assert abs(pos_multi / pos_base - 0.75) < 0.01


class TestBuffering:
    """Test Carver's no-trade buffer."""

    def test_within_buffer_keeps_current(self):
        """Small changes should not trigger a trade."""
        result = apply_buffer(
            current_position=0.04,
            target_position=0.041,  # Tiny change
            capital=10000,
            price=50000,
            instrument_vol=0.60,
            buffer_fraction=0.10,
        )
        assert result == 0.04  # Kept current

    def test_outside_buffer_moves_to_edge(self):
        """Large changes should move to buffer edge."""
        result = apply_buffer(
            current_position=0.04,
            target_position=0.08,  # Big change
            capital=10000,
            price=50000,
            instrument_vol=0.60,
            buffer_fraction=0.10,
        )
        # Should move toward target but stop at buffer edge
        assert result != 0.04
        assert result <= 0.08

    def test_flat_to_position(self):
        """Moving from flat to a position should work."""
        result = apply_buffer(
            current_position=0.0,
            target_position=0.04,
            capital=10000,
            price=50000,
            instrument_vol=0.60,
            buffer_fraction=0.10,
        )
        # Buffer edge should be target - buffer
        assert result > 0


class TestMinimumCapital:
    def test_btc_minimum(self):
        """BTC at $100k with 60% vol needs reasonable minimum capital."""
        min_cap = minimum_capital_for_instrument(
            price=100000,
            instrument_vol=0.60,
            min_order_size=0.001,
        )
        # 0.001 * 0.60 * 100000 / 0.12 = 500
        assert abs(min_cap - 500) < 1
