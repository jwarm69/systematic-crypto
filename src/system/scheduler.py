"""Multi-frequency production scheduler.

Manages different update cadences:
- Hourly: forecast recalculation, position sizing, trade execution
- 8-hourly: carry/funding rate updates
- Daily: full rebalance, risk baseline reset, ML model retrain check
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Schedule configuration."""

    forecast_interval_hours: int = 1
    carry_interval_hours: int = 8
    rebalance_interval_hours: int = 24
    ml_retrain_interval_hours: int = 168  # Weekly


@dataclass
class SchedulerState:
    """Track when each task was last run."""

    last_forecast: datetime | None = None
    last_carry_update: datetime | None = None
    last_rebalance: datetime | None = None
    last_ml_retrain: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0


class Scheduler:
    """Multi-frequency task scheduler for trading system."""

    def __init__(self, config: ScheduleConfig | None = None):
        self.config = config or ScheduleConfig()
        self.state = SchedulerState()

    def should_run_forecast(self, now: datetime | None = None) -> bool:
        """Check if forecast recalculation is due."""
        now = now or datetime.now(timezone.utc)
        if self.state.last_forecast is None:
            return True
        elapsed = (now - self.state.last_forecast).total_seconds() / 3600
        return elapsed >= self.config.forecast_interval_hours

    def should_update_carry(self, now: datetime | None = None) -> bool:
        """Check if carry/funding rate update is due."""
        now = now or datetime.now(timezone.utc)
        if self.state.last_carry_update is None:
            return True
        elapsed = (now - self.state.last_carry_update).total_seconds() / 3600
        return elapsed >= self.config.carry_interval_hours

    def should_rebalance(self, now: datetime | None = None) -> bool:
        """Check if full rebalance is due."""
        now = now or datetime.now(timezone.utc)
        if self.state.last_rebalance is None:
            return True
        elapsed = (now - self.state.last_rebalance).total_seconds() / 3600
        return elapsed >= self.config.rebalance_interval_hours

    def should_retrain_ml(self, now: datetime | None = None) -> bool:
        """Check if ML model retraining is due."""
        now = now or datetime.now(timezone.utc)
        if self.state.last_ml_retrain is None:
            return True
        elapsed = (now - self.state.last_ml_retrain).total_seconds() / 3600
        return elapsed >= self.config.ml_retrain_interval_hours

    def mark_forecast_done(self, now: datetime | None = None) -> None:
        self.state.last_forecast = now or datetime.now(timezone.utc)

    def mark_carry_done(self, now: datetime | None = None) -> None:
        self.state.last_carry_update = now or datetime.now(timezone.utc)

    def mark_rebalance_done(self, now: datetime | None = None) -> None:
        self.state.last_rebalance = now or datetime.now(timezone.utc)

    def mark_ml_retrain_done(self, now: datetime | None = None) -> None:
        self.state.last_ml_retrain = now or datetime.now(timezone.utc)

    def mark_run_complete(self, success: bool = True) -> None:
        self.state.run_count += 1
        if success:
            self.state.consecutive_errors = 0
        else:
            self.state.error_count += 1
            self.state.consecutive_errors += 1

    def should_backoff(self, max_consecutive_errors: int = 5) -> bool:
        """Check if we should back off due to repeated errors."""
        return self.state.consecutive_errors >= max_consecutive_errors

    def backoff_seconds(self) -> int:
        """Calculate exponential backoff delay."""
        if self.state.consecutive_errors <= 1:
            return 0
        return min(300, 30 * (2 ** (self.state.consecutive_errors - 1)))

    def seconds_until_next_run(self, now: datetime | None = None) -> float:
        """Calculate seconds until the next scheduled forecast run."""
        now = now or datetime.now(timezone.utc)
        if self.state.last_forecast is None:
            return 0
        next_run = self.state.last_forecast.timestamp() + (
            self.config.forecast_interval_hours * 3600
        )
        remaining = next_run - now.timestamp()
        return max(0, remaining)

    def get_status(self) -> dict:
        """Return scheduler status for dashboard."""
        now = datetime.now(timezone.utc)
        return {
            "run_count": self.state.run_count,
            "error_count": self.state.error_count,
            "consecutive_errors": self.state.consecutive_errors,
            "last_forecast": self.state.last_forecast.isoformat() if self.state.last_forecast else None,
            "last_carry_update": self.state.last_carry_update.isoformat() if self.state.last_carry_update else None,
            "last_rebalance": self.state.last_rebalance.isoformat() if self.state.last_rebalance else None,
            "last_ml_retrain": self.state.last_ml_retrain.isoformat() if self.state.last_ml_retrain else None,
            "seconds_until_next": self.seconds_until_next_run(now),
            "forecast_due": self.should_run_forecast(now),
            "carry_due": self.should_update_carry(now),
            "rebalance_due": self.should_rebalance(now),
            "ml_retrain_due": self.should_retrain_ml(now),
        }
