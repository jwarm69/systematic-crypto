#!/usr/bin/env python3
"""Fetch and cache OHLCV data for instruments."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import fetch_and_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


INSTRUMENT_SYMBOLS = {
    "BTC": "BTC/USDC:USDC",
    "ETH": "ETH/USDC:USDC",
    "SOL": "SOL/USDC:USDC",
}


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data")
    parser.add_argument("--instrument", default="BTC", choices=list(INSTRUMENT_SYMBOLS.keys()))
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--exchange", default="hyperliquid")
    parser.add_argument("--force", action="store_true", help="Force refresh from exchange")
    args = parser.parse_args()

    symbol = INSTRUMENT_SYMBOLS[args.instrument]
    df = fetch_and_cache(
        symbol=symbol,
        timeframe=args.timeframe,
        limit=args.bars,
        exchange_id=args.exchange,
        force_refresh=args.force,
    )

    if df.empty:
        print("No data returned")
        sys.exit(1)

    print(f"\nFetched {len(df)} bars for {args.instrument} ({args.timeframe})")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"\nLast 5 bars:")
    print(df.tail())


if __name__ == "__main__":
    main()
