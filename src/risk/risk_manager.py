"""Risk management: kill switch, drawdown limits, daily loss limits, correlation monitoring.

Adapted from crypto_momo/momo/risk.py with Carver-specific enhancements.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """Portfolio risk manager with kill switch and position limits."""

    def __init__(
        self,
        max_drawdown: float = 0.15,
        daily_loss_limit: float = 0.03,
        kill_switch_ratio: float = 1.3,
        max_leverage: float = 2.0,
        max_correlation: float = 0.90,
    ):
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.kill_switch_ratio = kill_switch_ratio
        self.max_leverage = max_leverage
        self.max_correlation = max_correlation

        self.kill_switch_active = False
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self._last_daily_reset: datetime | None = None

    def update_equity(self, equity: float) -> None:
        """Update equity tracking for drawdown calculation."""
        if equity > self.peak_equity:
            self.peak_equity = equity

    def reset_daily(self, equity: float) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        self.daily_start_equity = equity
        self._last_daily_reset = datetime.now(timezone.utc)

    def maybe_reset_daily(self, equity: float, now: datetime | None = None) -> bool:
        """Reset daily baseline if this is the first run or a new UTC day.

        Returns:
            True if a reset occurred, False otherwise
        """
        now_utc = now or datetime.now(timezone.utc)
        if self._last_daily_reset is None or self._last_daily_reset.date() != now_utc.date():
            self.daily_start_equity = equity
            self._last_daily_reset = now_utc
            return True
        return False

    def check_drawdown(self, equity: float) -> tuple[bool, str]:
        """Check if drawdown limit is breached.

        Returns:
            (is_breached, reason)
        """
        if self.peak_equity <= 0:
            return False, ""

        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            reason = f"Drawdown {drawdown:.1%} exceeds limit {self.max_drawdown:.1%}"
            logger.critical(f"KILL SWITCH: {reason}")
            self.kill_switch_active = True
            return True, reason
        return False, ""

    def check_daily_loss(self, equity: float) -> tuple[bool, str]:
        """Check if daily loss limit is breached.

        Returns:
            (is_breached, reason)
        """
        if self.daily_start_equity <= 0:
            return False, ""

        daily_loss = (self.daily_start_equity - equity) / self.daily_start_equity
        if daily_loss > self.daily_loss_limit:
            reason = f"Daily loss {daily_loss:.1%} exceeds limit {self.daily_loss_limit:.1%}"
            logger.warning(f"DAILY LIMIT: {reason}")
            return True, reason
        return False, ""

    def check_leverage(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        equity: float,
    ) -> tuple[bool, str]:
        """Check if total leverage exceeds limit.

        Args:
            positions: symbol -> position size in units
            prices: symbol -> current price
            equity: Current equity

        Returns:
            (is_breached, reason)
        """
        if equity <= 0:
            return True, "Zero equity"

        total_notional = sum(
            abs(pos) * prices.get(sym, 0) for sym, pos in positions.items()
        )
        leverage = total_notional / equity

        if leverage > self.max_leverage:
            reason = f"Leverage {leverage:.1f}x exceeds limit {self.max_leverage:.1f}x"
            return True, reason
        return False, ""

    def check_kill_switch_ratio(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        equity: float,
    ) -> tuple[bool, str]:
        """Trigger kill switch if leverage is far beyond configured limits.

        This uses `kill_switch_ratio` as an escalation multiplier above
        `max_leverage`.
        """
        if equity <= 0:
            self.kill_switch_active = True
            return True, "Kill switch: zero or negative equity"

        total_notional = sum(
            abs(pos) * prices.get(sym, 0) for sym, pos in positions.items()
        )
        leverage = total_notional / equity
        kill_switch_leverage = self.max_leverage * self.kill_switch_ratio

        if leverage > kill_switch_leverage:
            reason = (
                f"Kill switch leverage {leverage:.1f}x exceeds trigger "
                f"{kill_switch_leverage:.1f}x"
            )
            logger.critical(f"KILL SWITCH: {reason}")
            self.kill_switch_active = True
            return True, reason
        return False, ""

    def check_all(
        self,
        equity: float,
        positions: dict[str, float] | None = None,
        prices: dict[str, float] | None = None,
    ) -> tuple[bool, list[str]]:
        """Run all risk checks.

        Returns:
            (should_reduce, list of reasons)
        """
        reasons = []

        if self.kill_switch_active:
            reasons.append("Kill switch previously triggered")
            return True, reasons

        self.update_equity(equity)

        dd_breach, dd_reason = self.check_drawdown(equity)
        if dd_breach:
            reasons.append(dd_reason)

        dl_breach, dl_reason = self.check_daily_loss(equity)
        if dl_breach:
            reasons.append(dl_reason)

        if positions and prices:
            lev_breach, lev_reason = self.check_leverage(positions, prices, equity)
            if lev_breach:
                reasons.append(lev_reason)
            ks_breach, ks_reason = self.check_kill_switch_ratio(positions, prices, equity)
            if ks_breach:
                reasons.append(ks_reason)

        return len(reasons) > 0, reasons

    def scale_for_risk(
        self,
        target_positions: dict[str, float],
        equity: float,
        prices: dict[str, float],
    ) -> dict[str, float]:
        """Scale down positions if leverage limit would be breached.

        Args:
            target_positions: Desired positions
            equity: Current equity
            prices: Current prices

        Returns:
            Scaled positions that respect leverage limit
        """
        if equity <= 0:
            return {sym: 0 for sym in target_positions}

        total_notional = sum(
            abs(pos) * prices.get(sym, 0) for sym, pos in target_positions.items()
        )
        leverage = total_notional / equity

        if leverage <= self.max_leverage:
            return target_positions

        scale = self.max_leverage / leverage
        logger.warning(f"Scaling positions by {scale:.2f} to meet leverage limit")
        return {sym: pos * scale for sym, pos in target_positions.items()}

    def check_correlation_risk(
        self,
        returns_df: pd.DataFrame,
        positions: dict[str, float],
        lookback: int = 168,  # 1 week of hourly bars
    ) -> list[str]:
        """Check for excessive correlation between held positions.

        Args:
            returns_df: DataFrame of returns (columns = symbols)
            positions: Current positions
            lookback: Bars to look back

        Returns:
            List of warnings
        """
        warnings = []
        symbols = [s for s, p in positions.items() if p != 0]

        if len(symbols) < 2:
            return warnings

        recent = returns_df[symbols].tail(lookback)
        if len(recent) < 20:
            return warnings

        corr = recent.corr()

        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                c = corr.loc[s1, s2]
                p1_sign = np.sign(positions[s1])
                p2_sign = np.sign(positions[s2])

                if abs(c) > self.max_correlation and p1_sign == p2_sign:
                    warnings.append(
                        f"High corr {c:.2f} between {s1} and {s2} (same direction)"
                    )

        return warnings
