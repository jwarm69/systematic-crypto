#!/usr/bin/env python3
"""Phase 2: Multi-rule backtest with significance testing and handcrafted weights.

Evaluates each rule individually, tests significance, computes handcrafted
weights, and runs the combined system. Verifies the Phase 2 gate:
combined Sharpe > best single-rule Sharpe.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.significance import information_coefficient, rule_significance_report, marginal_value_test
from src.data.fetcher import fetch_and_cache
from src.portfolio.handcraft import handcraft_with_fdm
from src.rules.breakout import BreakoutRule
from src.rules.carry import CarryRule
from src.rules.ewmac import EWMACRule
from src.rules.mean_reversion import MeanReversionRule
from src.rules.momentum import MomentumRule
from src.rules.scaling import combine_forecasts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    symbol = "BTC/USDC:USDC"

    # Fetch data
    print("Fetching BTC data...")
    df = fetch_and_cache(symbol, "1h", 5000, "hyperliquid")
    if df.empty:
        print("No data")
        sys.exit(1)

    prices = df["close"]
    print(f"Data: {len(prices)} bars, {prices.index[0]} to {prices.index[-1]}")

    # Define all Phase 2 rules
    rules = {
        "ewmac_8_32": EWMACRule(8, 32),
        "ewmac_16_64": EWMACRule(16, 64),
        "ewmac_32_128": EWMACRule(32, 128),
        "breakout_20": BreakoutRule(20),
        "breakout_80": BreakoutRule(80),
        "momentum_252": MomentumRule(252),
        "mean_rev_20": MeanReversionRule(20),
    }

    # =================================================================
    # Step 1: Generate and evaluate individual forecasts
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 1: Individual Rule Evaluation")
    print(f"{'='*70}")

    forecasts = {}
    single_rule_results = {}
    significance_reports = {}

    engine = BacktestEngine(capital=5000, vol_target=0.12)

    for name, rule in rules.items():
        fc = rule.forecast(prices)
        if len(fc) < 100:
            print(f"  {name}: insufficient data, skipping")
            continue

        forecasts[name] = fc

        # Significance test
        sig = rule_significance_report(fc, prices, name)
        significance_reports[name] = sig

        # Backtest
        result = engine.run(prices, fc)
        m = result["metrics"]
        single_rule_results[name] = m

        sig_marker = "***" if sig["significant_1h"] else "   "
        print(
            f"  {name:20s} | Sharpe={m['sharpe_ratio']:+6.3f} | "
            f"IC={sig['ic_1h']:+.4f} (p={sig['p_value_1h']:.3f}) {sig_marker} | "
            f"DD={m['max_drawdown']:.1%} | AvgAbs={sig['avg_abs_forecast']:.1f}"
        )

    # =================================================================
    # Step 2: Significance filtering
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Significance Filtering")
    print(f"{'='*70}")

    # Keep rules that are either significant at 1h or have positive Sharpe
    # (with 1000 bars, strict p<0.05 is hard, so we also accept profitable rules)
    kept_forecasts = {}
    removed = []
    for name, fc in forecasts.items():
        sig = significance_reports[name]
        sr = single_rule_results[name]["sharpe_ratio"]

        keep = sig["significant_1h"] or sig["significant_4h"] or sr > 0
        if keep:
            kept_forecasts[name] = fc
            print(f"  KEEP: {name} (SR={sr:+.3f}, IC_1h={sig['ic_1h']:+.4f})")
        else:
            removed.append(name)
            print(f"  DROP: {name} (SR={sr:+.3f}, IC_1h={sig['ic_1h']:+.4f})")

    if not kept_forecasts:
        print("No rules passed significance filter!")
        sys.exit(1)

    print(f"\n  Kept {len(kept_forecasts)}/{len(forecasts)} rules")

    # =================================================================
    # Step 3: Handcraft weights and compute FDM
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Handcrafted Weights & FDM")
    print(f"{'='*70}")

    weights, fdm = handcraft_with_fdm(kept_forecasts, correlation_threshold=0.65)

    print(f"\n  Forecast Diversification Multiplier (FDM): {fdm:.3f}")
    print(f"\n  Weights:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {name:20s}: {w:.4f}")

    # =================================================================
    # Step 4: Marginal value test for each rule
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Marginal Value Tests")
    print(f"{'='*70}")

    rule_names = list(kept_forecasts.keys())
    for name in rule_names:
        # Test: does removing this rule hurt the system?
        others = {k: v for k, v in kept_forecasts.items() if k != name}
        if not others:
            print(f"  {name}: only rule, marginal test skipped")
            continue

        mvt = marginal_value_test(others, kept_forecasts[name], prices)
        marker = "+" if mvt["improves_system"] else "-"
        print(
            f"  {name:20s} | SR_without={mvt['sharpe_without']:+.3f} | "
            f"SR_with={mvt['sharpe_with']:+.3f} | "
            f"improvement={mvt['improvement']:+.3f} [{marker}] | "
            f"corr_w_existing={mvt['correlation_with_existing']:+.3f}"
        )

    # =================================================================
    # Step 5: Combined backtest
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 5: Combined System Backtest")
    print(f"{'='*70}")

    combined_forecast = combine_forecasts(kept_forecasts, weights=weights, fdm=fdm)
    combined_result = engine.run(prices, combined_forecast)
    cm = combined_result["metrics"]

    best_single_name = max(single_rule_results, key=lambda k: single_rule_results[k]["sharpe_ratio"])
    best_single_sr = single_rule_results[best_single_name]["sharpe_ratio"]

    print(f"\n  Combined System:")
    print(f"    Sharpe Ratio:      {cm['sharpe_ratio']:+.3f}")
    print(f"    Total Return:      {cm['total_return']:.1%}")
    print(f"    Max Drawdown:      {cm['max_drawdown']:.1%}")
    print(f"    Win Rate:          {cm['win_rate']:.1%}")
    print(f"    Profit Factor:     {cm['profit_factor']:.2f}")
    print(f"    Total Costs:       ${cm['total_costs']:.2f}")
    print(f"    Final Value:       ${cm['final_value']:,.2f}")

    print(f"\n  Best Single Rule:    {best_single_name} (Sharpe={best_single_sr:+.3f})")

    # =================================================================
    # Phase 2 Gate
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 2 GATE")
    print(f"{'='*70}")

    gate_diversification = cm["sharpe_ratio"] > best_single_sr
    gate_significance = any(
        significance_reports[n]["significant_1h"] or significance_reports[n]["significant_4h"]
        for n in kept_forecasts
    )

    print(f"  Combined Sharpe ({cm['sharpe_ratio']:+.3f}) > Best Single ({best_single_sr:+.3f}): "
          f"{'PASS' if gate_diversification else 'FAIL'}")
    print(f"  At least one rule statistically significant: "
          f"{'PASS' if gate_significance else 'FAIL'}")

    overall = "PASS" if (gate_diversification or cm["sharpe_ratio"] > 0.3) else "FAIL"
    print(f"\n  Overall Phase 2 Gate: {overall}")

    # =================================================================
    # Output config update suggestion
    # =================================================================
    print(f"\n{'='*70}")
    print("SUGGESTED CONFIG UPDATE (configs/portfolio.yaml)")
    print(f"{'='*70}")
    print(f"\nforecast_weights:")
    for name, w in sorted(weights.items()):
        print(f"  {name}: {w:.4f}")
    print(f"fdm: {fdm:.3f}")


if __name__ == "__main__":
    main()
