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

    def test_kill_switch_ratio_triggered_by_extreme_leverage(self):
        positions = {"BTC/USDC:USDC": 0.3}
        prices = {"BTC/USDC:USDC": 100000}
        # 30k notional on 10k equity = 3.0x leverage
        # Kill switch threshold = 2.0 * 1.3 = 2.6x
        should_reduce, reasons = self.rm.check_all(10000, positions=positions, prices=prices)
        assert should_reduce
        assert self.rm.kill_switch_active
        assert any("Kill switch leverage" in r for r in reasons)

    def test_kill_switch_ratio_not_triggered_when_below_threshold(self):
        positions = {"BTC/USDC:USDC": 0.25}
        prices = {"BTC/USDC:USDC": 100000}
        # 2.5x leverage breaches max_leverage but is below kill switch threshold 2.6x
        should_reduce, reasons = self.rm.check_all(10000, positions=positions, prices=prices)
        assert should_reduce
        assert not self.rm.kill_switch_active
        assert any("Leverage" in r for r in reasons)
        assert not any("Kill switch leverage" in r for r in reasons)
