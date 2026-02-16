"""Tests for risk management."""

import pytest

from src.risk.risk_manager import RiskManager


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(
            max_drawdown=0.15,
            daily_loss_limit=0.03,
            max_leverage=2.0,
        )

    def test_no_drawdown_issue(self):
        self.rm.peak_equity = 10000
        breached, reason = self.rm.check_drawdown(9500)
        assert not breached

    def test_drawdown_breach(self):
        self.rm.peak_equity = 10000
        breached, reason = self.rm.check_drawdown(8000)  # 20% drawdown
        assert breached
        assert "20.0%" in reason

    def test_daily_loss_ok(self):
        self.rm.daily_start_equity = 10000
        breached, reason = self.rm.check_daily_loss(9800)  # 2% loss
        assert not breached

    def test_daily_loss_breach(self):
        self.rm.daily_start_equity = 10000
        breached, reason = self.rm.check_daily_loss(9600)  # 4% loss
        assert breached

    def test_leverage_ok(self):
        positions = {"BTC/USDC:USDC": 0.1}
        prices = {"BTC/USDC:USDC": 100000}
        breached, reason = self.rm.check_leverage(positions, prices, 10000)
        assert not breached  # 1x leverage

    def test_leverage_breach(self):
        positions = {"BTC/USDC:USDC": 0.3}
        prices = {"BTC/USDC:USDC": 100000}
        breached, reason = self.rm.check_leverage(positions, prices, 10000)
        assert breached  # 3x leverage

    def test_scale_for_risk(self):
        positions = {"BTC/USDC:USDC": 0.3}
        prices = {"BTC/USDC:USDC": 100000}
        scaled = self.rm.scale_for_risk(positions, 10000, prices)
        # 30k notional / 10k equity = 3x -> should scale to 2x
        notional = abs(scaled["BTC/USDC:USDC"]) * 100000
        assert notional <= 10000 * 2.0 + 1

    def test_kill_switch_persists(self):
        self.rm.peak_equity = 10000
        self.rm.check_drawdown(8000)  # Trigger
        assert self.rm.kill_switch_active

        should_reduce, reasons = self.rm.check_all(9500)
        assert should_reduce
        assert "Kill switch previously triggered" in reasons
