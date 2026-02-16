#!/usr/bin/env python3
"""Run paper trading on Hyperliquid testnet."""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.system.trading_system import TradingSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("--config", default=None, help="Path to configs directory")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between runs")
    args = parser.parse_args()

    system = TradingSystem(config_path=args.config)
    system.load_state()

    logger.info("Starting paper trading system")
    logger.info(f"Capital: ${system.capital:,.0f} | Vol target: {system.vol_target:.0%}")
    logger.info(f"Rules: {list(system.rules.keys())}")

    if args.once:
        results = system.run()
        _print_results(results)
        return

    # Continuous loop
    while True:
        try:
            results = system.run()
            _print_results(results)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)

        logger.info(f"Sleeping {args.interval}s until next run...")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break


def _print_results(results: dict):
    """Pretty-print trading loop results."""
    for instrument, data in results.items():
        logger.info(
            f"{instrument}: forecast={data['forecast']:.1f} "
            f"vol={data['volatility']:.1%} "
            f"pos={data['buffered_position']:.6f} "
            f"price=${data['price']:.2f} "
            f"trade={data['trade_size']:.6f}"
        )
        if data.get("risk_reasons"):
            logger.warning(f"  Risk: {data['risk_reasons']}")


if __name__ == "__main__":
    main()
