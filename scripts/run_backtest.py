#!/usr/bin/env python3
"""Run backtest for a single instrument with EWMAC."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_and_cache
from src.rules.ewmac import EWMACRule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


INSTRUMENT_SYMBOLS = {
    "BTC": "BTC/USDC:USDC",
    "ETH": "ETH/USDC:USDC",
    "SOL": "SOL/USDC:USDC",
}


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--instrument", default="BTC", choices=list(INSTRUMENT_SYMBOLS.keys()))
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--capital", type=float, default=5000)
    parser.add_argument("--vol-target", type=float, default=0.12)
    parser.add_argument("--fast-span", type=int, default=16)
    parser.add_argument("--slow-span", type=int, default=64)
    parser.add_argument("--exchange", default="hyperliquid")
    parser.add_argument(
        "--funding-bps-per-8h",
        type=float,
        default=0.0,
        help="Constant funding assumption in bps per 8h (0 disables)",
    )
    args = parser.parse_args()

    symbol = INSTRUMENT_SYMBOLS[args.instrument]

    # Fetch data
    print(f"Fetching {args.bars} bars of {args.instrument} {args.timeframe} data...")
    df = fetch_and_cache(symbol, args.timeframe, args.bars, args.exchange)
    if df.empty:
        print("No data available")
        sys.exit(1)

    prices = df["close"]
    print(f"Data: {len(prices)} bars from {prices.index[0]} to {prices.index[-1]}")

    # Generate forecast
    rule = EWMACRule(fast_span=args.fast_span, slow_span=args.slow_span)
    forecast = rule.forecast(prices)
    print(f"\nForecast stats:")
    print(f"  Mean: {forecast.mean():.2f}")
    print(f"  Avg abs: {forecast.abs().mean():.2f} (target: 10)")
    print(f"  Min: {forecast.min():.2f}, Max: {forecast.max():.2f}")
    print(f"  Current: {forecast.iloc[-1]:.2f}")

    # Run backtest
    engine = BacktestEngine(
        capital=args.capital,
        vol_target=args.vol_target,
        timeframe=args.timeframe,
        funding_bps_per_8h=args.funding_bps_per_8h,
    )
    results = engine.run(prices, forecast)

    # Print metrics
    m = results["metrics"]
    print(f"\n{'='*50}")
    print(f"BACKTEST RESULTS: {args.instrument} EWMAC({args.fast_span},{args.slow_span})")
    print(f"{'='*50}")
    print(f"Capital:           ${args.capital:,.0f}")
    print(f"Timeframe:         {args.timeframe} ({m['bars_per_year']:.0f} bars/year)")
    print(f"Final Value:       ${m['final_value']:,.2f}")
    print(f"Total Return:      {m['total_return']:.1%}")
    print(f"Annualized Return: {m['annualized_return']:.1%}")
    print(f"Annualized Vol:    {m['annualized_vol']:.1%}")
    print(f"Sharpe Ratio:      {m['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio:     {m['sortino_ratio']:.3f}")
    print(f"Calmar Ratio:      {m['calmar_ratio']:.3f}")
    print(f"Max Drawdown:      {m['max_drawdown']:.1%}")
    print(f"Win Rate:          {m['win_rate']:.1%}")
    print(f"Profit Factor:     {m['profit_factor']:.2f}")
    print(f"Total Costs:       ${m['total_costs']:.2f}")
    print(f"Funding PnL:       ${m['total_funding_pnl']:.2f}")
    print(f"Bars:              {m['n_bars']}")
    print(f"{'='*50}")

    gate = "PASS" if m["sharpe_ratio"] > 0.3 else "FAIL"
    print(f"\nPhase 1 Gate (Sharpe > 0.3): {gate}")


if __name__ == "__main__":
    main()
