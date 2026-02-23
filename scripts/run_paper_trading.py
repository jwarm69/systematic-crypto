#!/usr/bin/env python3
"""Run paper trading on Hyperliquid testnet with production scheduler."""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.system.scheduler import Scheduler, ScheduleConfig
from src.system.trading_system import TradingSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/paper_trading.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("--config", default=None, help="Path to configs directory")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between runs")
    args = parser.parse_args()

    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)

    system = TradingSystem(config_path=args.config)
    system.load_state()

    schedule_cfg = ScheduleConfig(
        forecast_interval_hours=args.interval / 3600,
    )
    scheduler = Scheduler(config=schedule_cfg)

    logger.info("=" * 60)
    logger.info("SYSTEMATIC CRYPTO TRADING SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Capital: ${system.capital:,.0f} | Vol target: {system.vol_target:.0%}")
    logger.info(f"Max leverage: {system.max_leverage:.1f}x | IDM: {system.idm:.3f}")
    logger.info(f"Rules: {list(system.rules.keys())}")
    instruments = system.paper_cfg.get("trading", {}).get("instruments", ["BTC"])
    logger.info(f"Instruments: {instruments}")
    logger.info(f"Mode: {system.paper_cfg.get('trading', {}).get('mode', 'paper')}")
    logger.info("=" * 60)

    if args.once:
        results = system.run()
        _print_results(results)
        scheduler.mark_forecast_done()
        scheduler.mark_run_complete(success=True)
        _print_status(scheduler, system)
        return

    # Continuous loop
    while True:
        if scheduler.should_backoff():
            backoff = scheduler.backoff_seconds()
            logger.warning(
                f"Backing off {backoff}s after {scheduler.state.consecutive_errors} "
                f"consecutive errors"
            )
            time.sleep(backoff)

        try:
            logger.info(f"\n--- Run #{scheduler.state.run_count + 1} ---")
            results = system.run()
            _print_results(results)
            scheduler.mark_forecast_done()
            scheduler.mark_run_complete(success=True)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
            scheduler.mark_run_complete(success=False)

        _print_status(scheduler, system)

        sleep_time = max(10, scheduler.seconds_until_next_run())
        logger.info(f"Sleeping {sleep_time:.0f}s until next run...")
        try:
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break


def _print_results(results: dict):
    """Pretty-print trading loop results."""
    for instrument, data in results.items():
        side = "LONG" if data["buffered_position"] > 0 else "SHORT" if data["buffered_position"] < 0 else "FLAT"
        logger.info(
            f"  {instrument}: {side} forecast={data['forecast']:+.1f} "
            f"vol={data['volatility']:.1%} "
            f"pos={data['buffered_position']:.6f} "
            f"price=${data['price']:,.2f} "
            f"trade={data['trade_size']:.6f}"
        )
        if data.get("rule_forecasts"):
            for rule, fc_val in data["rule_forecasts"].items():
                logger.debug(f"    {rule}: {fc_val:+.1f}")
        if data.get("risk_reasons"):
            logger.warning(f"  Risk: {data['risk_reasons']}")


def _print_status(scheduler: Scheduler, system: TradingSystem):
    """Print scheduler and system status."""
    status = scheduler.get_status()
    logger.info(
        f"  Status: runs={status['run_count']} errors={status['error_count']} "
        f"kill_switch={'ON' if system.risk_manager.kill_switch_active else 'off'}"
    )


if __name__ == "__main__":
    main()
