"""Tests for Phase 5: production system components."""

import csv
import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.system.scheduler import Scheduler, ScheduleConfig, SchedulerState
from src.system.trade_logger import TradeLogger


class TestScheduler:
    def test_initial_state_all_due(self):
        """All tasks should be due on first run."""
        sched = Scheduler()
        assert sched.should_run_forecast()
        assert sched.should_update_carry()
        assert sched.should_rebalance()
        assert sched.should_retrain_ml()

    def test_forecast_interval(self):
        sched = Scheduler(ScheduleConfig(forecast_interval_hours=1))
        now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        sched.mark_forecast_done(now)

        # 30 min later: not due
        assert not sched.should_run_forecast(now + timedelta(minutes=30))
        # 61 min later: due
        assert sched.should_run_forecast(now + timedelta(minutes=61))

    def test_carry_interval(self):
        sched = Scheduler(ScheduleConfig(carry_interval_hours=8))
        now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        sched.mark_carry_done(now)

        assert not sched.should_update_carry(now + timedelta(hours=7))
        assert sched.should_update_carry(now + timedelta(hours=9))

    def test_rebalance_interval(self):
        sched = Scheduler(ScheduleConfig(rebalance_interval_hours=24))
        now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        sched.mark_rebalance_done(now)

        assert not sched.should_rebalance(now + timedelta(hours=23))
        assert sched.should_rebalance(now + timedelta(hours=25))

    def test_ml_retrain_weekly(self):
        sched = Scheduler(ScheduleConfig(ml_retrain_interval_hours=168))
        now = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        sched.mark_ml_retrain_done(now)

        assert not sched.should_retrain_ml(now + timedelta(days=6))
        assert sched.should_retrain_ml(now + timedelta(days=8))

    def test_run_counting(self):
        sched = Scheduler()
        sched.mark_run_complete(success=True)
        sched.mark_run_complete(success=True)
        sched.mark_run_complete(success=False)
        assert sched.state.run_count == 3
        assert sched.state.error_count == 1
        assert sched.state.consecutive_errors == 1

    def test_consecutive_errors_reset_on_success(self):
        sched = Scheduler()
        sched.mark_run_complete(success=False)
        sched.mark_run_complete(success=False)
        assert sched.state.consecutive_errors == 2
        sched.mark_run_complete(success=True)
        assert sched.state.consecutive_errors == 0

    def test_backoff(self):
        sched = Scheduler()
        assert not sched.should_backoff()
        for _ in range(5):
            sched.mark_run_complete(success=False)
        assert sched.should_backoff()
        assert sched.backoff_seconds() > 0

    def test_seconds_until_next_run(self):
        sched = Scheduler(ScheduleConfig(forecast_interval_hours=1))
        now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        sched.mark_forecast_done(now)

        # 30 min later: ~30 min remaining
        remaining = sched.seconds_until_next_run(now + timedelta(minutes=30))
        assert 1700 < remaining < 1900

        # 90 min later: 0 (overdue)
        remaining = sched.seconds_until_next_run(now + timedelta(minutes=90))
        assert remaining == 0

    def test_get_status(self):
        sched = Scheduler()
        sched.mark_run_complete(success=True)
        status = sched.get_status()
        assert "run_count" in status
        assert "error_count" in status
        assert "forecast_due" in status
        assert status["run_count"] == 1


class TestTradeLogger:
    def test_log_trade(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(
                trade_log_path=Path(tmpdir) / "trades.csv",
                pnl_history_path=Path(tmpdir) / "pnl.json",
                run_log_path=Path(tmpdir) / "runs.json",
            )

            logger.log_trade(
                instrument="BTC",
                symbol="BTC/USDC:USDC",
                side="buy",
                size=0.001,
                price=95000,
                filled_price=95010,
                forecast=15.5,
                volatility=0.65,
                target_pos=0.001,
                buffered_pos=0.001,
                success=True,
            )

            trades = logger.get_trade_history()
            assert len(trades) == 1
            assert trades[0]["instrument"] == "BTC"
            assert trades[0]["side"] == "buy"
            assert trades[0]["success"] == "True"

    def test_log_multiple_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(
                trade_log_path=Path(tmpdir) / "trades.csv",
                pnl_history_path=Path(tmpdir) / "pnl.json",
                run_log_path=Path(tmpdir) / "runs.json",
            )

            for i in range(5):
                logger.log_trade(
                    instrument="BTC", symbol="BTC/USDC:USDC",
                    side="buy" if i % 2 == 0 else "sell",
                    size=0.001, price=95000 + i * 100,
                    filled_price=95000 + i * 100,
                    forecast=10.0, volatility=0.65,
                    target_pos=0.001, buffered_pos=0.001,
                    success=True,
                )

            trades = logger.get_trade_history()
            assert len(trades) == 5

    def test_log_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(
                trade_log_path=Path(tmpdir) / "trades.csv",
                pnl_history_path=Path(tmpdir) / "pnl.json",
                run_log_path=Path(tmpdir) / "runs.json",
            )

            logger.log_run({"equity": 5100, "positions": {"BTC": 0.001}})
            logger.log_run({"equity": 5150, "positions": {"BTC": 0.002}})

            history = logger.get_run_history()
            assert len(history) == 2
            assert history[0]["equity"] == 5100
            assert history[1]["equity"] == 5150

    def test_log_daily_pnl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(
                trade_log_path=Path(tmpdir) / "trades.csv",
                pnl_history_path=Path(tmpdir) / "pnl.json",
                run_log_path=Path(tmpdir) / "runs.json",
            )

            logger.log_daily_pnl("2024-01-01", 5100, 100, {"BTC": 0.001}, {"BTC": 95000})
            logger.log_daily_pnl("2024-01-02", 5200, 100, {"BTC": 0.002}, {"BTC": 96000})

            history = logger.get_pnl_history()
            assert len(history) == 2
            assert history[0]["date"] == "2024-01-01"
            assert history[1]["equity"] == 5200

    def test_pnl_update_existing_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(
                trade_log_path=Path(tmpdir) / "trades.csv",
                pnl_history_path=Path(tmpdir) / "pnl.json",
                run_log_path=Path(tmpdir) / "runs.json",
            )

            logger.log_daily_pnl("2024-01-01", 5100, 100, {}, {})
            logger.log_daily_pnl("2024-01-01", 5150, 150, {}, {})

            history = logger.get_pnl_history()
            assert len(history) == 1
            assert history[0]["equity"] == 5150

    def test_empty_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradeLogger(
                trade_log_path=Path(tmpdir) / "trades.csv",
                pnl_history_path=Path(tmpdir) / "pnl.json",
                run_log_path=Path(tmpdir) / "runs.json",
            )

            assert logger.get_trade_history() == []
            assert logger.get_pnl_history() == []
            assert logger.get_run_history() == []
