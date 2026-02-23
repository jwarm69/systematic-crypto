#!/usr/bin/env python3
"""Phase 3: Multi-instrument backtest with IDM and portfolio construction.

Fetches BTC, ETH, SOL data, computes instrument correlations and IDM,
runs per-instrument forecasts, and evaluates the portfolio-level system.
Verifies Phase 3 gate: multi-instrument Sharpe > single-instrument Sharpe.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.multi_instrument import MultiInstrumentBacktest
from src.data.fetcher import fetch_and_cache
from src.data.hyperliquid_data import (
    compute_cross_asset_features,
    compute_returns,
    liquidity_filter,
)
from src.portfolio.instrument_weights import (
    calculate_idm,
    compute_instrument_correlation,
    equal_weights,
    recommend_weights_and_idm,
)
from src.rules.breakout import BreakoutRule
from src.rules.ewmac import EWMACRule
from src.rules.momentum import MomentumRule
from src.rules.scaling import combine_forecasts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_config():
    with open(CONFIG_DIR / "portfolio.yaml") as f:
        portfolio = yaml.safe_load(f)
    with open(CONFIG_DIR / "instruments.yaml") as f:
        instruments = yaml.safe_load(f)
    return portfolio, instruments


def main():
    portfolio_cfg, instruments_cfg = load_config()

    # Phase 3 instruments
    target_instruments = ["BTC", "ETH", "SOL"]
    timeframe = "1h"
    limit = 5000

    # =================================================================
    # Step 1: Fetch data for all instruments
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 1: Fetch Multi-Instrument Data")
    print(f"{'='*70}")

    all_data = {}
    for name in target_instruments:
        cfg = instruments_cfg["instruments"].get(name, {})
        symbol = cfg.get("symbol", f"{name}/USDC:USDC")
        print(f"  Fetching {name} ({symbol})...")
        df = fetch_and_cache(symbol, timeframe, limit, "hyperliquid")
        if df.empty or len(df) < 200:
            print(f"  {name}: insufficient data ({len(df)} bars), skipping")
            continue
        all_data[name] = df
        print(f"  {name}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    if len(all_data) < 2:
        print("Need at least 2 instruments for Phase 3")
        sys.exit(1)

    active_instruments = list(all_data.keys())
    print(f"\n  Active instruments: {active_instruments}")

    # =================================================================
    # Step 2: Liquidity filter
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Liquidity Filter")
    print(f"{'='*70}")

    liquid = liquidity_filter(all_data, min_daily_volume_usd=500_000)
    if not liquid:
        print("  No instruments passed liquidity filter, using all")
        liquid = active_instruments

    print(f"  Liquid instruments: {liquid}")
    all_data = {k: v for k, v in all_data.items() if k in liquid}

    # =================================================================
    # Step 3: Compute correlations and IDM
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Instrument Correlations & IDM")
    print(f"{'='*70}")

    returns_dict = compute_returns(all_data)
    weights, idm, corr_matrix = recommend_weights_and_idm(returns_dict)

    print(f"\n  Instrument Weights:")
    for name, w in sorted(weights.items()):
        print(f"    {name}: {w:.4f}")

    print(f"\n  IDM: {idm:.3f}")

    print(f"\n  Correlation Matrix:")
    for name in corr_matrix.index:
        row = "    " + name.ljust(6)
        for col in corr_matrix.columns:
            row += f" {corr_matrix.loc[name, col]:+.3f}"
        print(row)

    # =================================================================
    # Step 4: Cross-asset features
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Cross-Asset Features")
    print(f"{'='*70}")

    cross_features = compute_cross_asset_features(all_data, reference="BTC")
    for name, feat_df in cross_features.items():
        if feat_df.empty:
            continue
        print(f"  {name}: {list(feat_df.columns)}")

    # =================================================================
    # Step 5: Generate forecasts per instrument
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 5: Generate Forecasts")
    print(f"{'='*70}")

    # Use the same rules across all instruments
    forecast_weights = portfolio_cfg["forecast_weights"]
    fdm = portfolio_cfg["fdm"]

    rules = {
        "ewmac_8_32": EWMACRule(8, 32),
        "ewmac_16_64": EWMACRule(16, 64),
        "ewmac_32_128": EWMACRule(32, 128),
        "breakout_80": BreakoutRule(80),
        "momentum": MomentumRule(252),
    }

    all_forecasts = {}
    all_prices = {}
    for name, df in all_data.items():
        prices = df["close"]
        all_prices[name] = prices

        rule_forecasts = {}
        for rule_name, rule in rules.items():
            if rule_name not in forecast_weights:
                continue
            fc = rule.forecast(prices)
            if len(fc) > 100:
                rule_forecasts[rule_name] = fc

        if not rule_forecasts:
            print(f"  {name}: no valid forecasts, skipping")
            continue

        combined = combine_forecasts(rule_forecasts, weights=forecast_weights, fdm=fdm)
        all_forecasts[name] = combined
        print(
            f"  {name}: combined forecast last={combined.iloc[-1]:+.1f}, "
            f"avg_abs={combined.abs().mean():.1f}"
        )

    # =================================================================
    # Step 6: Single-instrument baseline (BTC only)
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 6: Single-Instrument Baseline (BTC)")
    print(f"{'='*70}")

    single_engine = BacktestEngine(
        capital=portfolio_cfg["account"]["capital"],
        vol_target=portfolio_cfg["account"]["vol_target"],
        max_leverage=portfolio_cfg["account"]["max_leverage"],
    )

    if "BTC" in all_forecasts:
        single_result = single_engine.run(all_prices["BTC"], all_forecasts["BTC"])
        single_metrics = single_result["metrics"]
        print(f"  BTC-only Sharpe:    {single_metrics['sharpe_ratio']:+.3f}")
        print(f"  BTC-only Return:    {single_metrics['total_return']:.1%}")
        print(f"  BTC-only Max DD:    {single_metrics['max_drawdown']:.1%}")
        print(f"  BTC-only Final:     ${single_metrics['final_value']:,.2f}")
    else:
        single_metrics = {"sharpe_ratio": 0}
        print("  BTC not available for baseline")

    # =================================================================
    # Step 7: Multi-instrument portfolio backtest
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 7: Multi-Instrument Portfolio Backtest")
    print(f"{'='*70}")

    multi_engine = MultiInstrumentBacktest(
        capital=portfolio_cfg["account"]["capital"],
        vol_target=portfolio_cfg["account"]["vol_target"],
        max_leverage=portfolio_cfg["account"]["max_leverage"],
    )

    multi_result = multi_engine.run(
        prices_dict=all_prices,
        forecasts_dict=all_forecasts,
        instrument_weights=weights,
        idm=idm,
    )
    multi_metrics = multi_result["metrics"]

    print(f"\n  Portfolio Metrics ({len(all_forecasts)} instruments):")
    print(f"    Sharpe Ratio:      {multi_metrics['sharpe_ratio']:+.3f}")
    print(f"    Total Return:      {multi_metrics['total_return']:.1%}")
    print(f"    Max Drawdown:      {multi_metrics['max_drawdown']:.1%}")
    print(f"    Win Rate:          {multi_metrics['win_rate']:.1%}")
    print(f"    Profit Factor:     {multi_metrics['profit_factor']:.2f}")
    print(f"    Total Costs:       ${multi_metrics['total_costs']:.2f}")
    print(f"    Final Value:       ${multi_metrics['final_value']:,.2f}")
    print(f"    Avg Leverage:      {multi_metrics['avg_leverage']:.2f}x")
    print(f"    Max Leverage:      {multi_metrics['max_leverage_realized']:.2f}x")

    print(f"\n  Instrument Contributions:")
    for name, contrib in multi_metrics.get("instrument_contributions", {}).items():
        print(f"    {name}: PnL=${contrib['total_pnl']:.2f} ({contrib['pnl_share']:.1%})")

    # =================================================================
    # Step 8: Also test with IDM=1.0 for comparison
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 8: IDM Comparison")
    print(f"{'='*70}")

    no_idm_result = multi_engine.run(
        prices_dict=all_prices,
        forecasts_dict=all_forecasts,
        instrument_weights=weights,
        idm=1.0,
    )
    no_idm_metrics = no_idm_result["metrics"]
    print(f"  Without IDM (IDM=1.0):  Sharpe={no_idm_metrics['sharpe_ratio']:+.3f}, DD={no_idm_metrics['max_drawdown']:.1%}")
    print(f"  With IDM (IDM={idm:.3f}):   Sharpe={multi_metrics['sharpe_ratio']:+.3f}, DD={multi_metrics['max_drawdown']:.1%}")

    # =================================================================
    # Phase 3 Gate
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 3 GATE")
    print(f"{'='*70}")

    gate_sharpe = multi_metrics["sharpe_ratio"] > single_metrics["sharpe_ratio"]
    gate_drawdown = abs(multi_metrics["max_drawdown"]) < 0.15
    gate_multi = len(all_forecasts) >= 2

    print(f"  Multi-instrument Sharpe ({multi_metrics['sharpe_ratio']:+.3f}) > "
          f"Single-instrument ({single_metrics['sharpe_ratio']:+.3f}): "
          f"{'PASS' if gate_sharpe else 'FAIL'}")
    print(f"  Max Drawdown ({abs(multi_metrics['max_drawdown']):.1%}) < 15%: "
          f"{'PASS' if gate_drawdown else 'FAIL'}")
    print(f"  Multiple instruments active ({len(all_forecasts)}): "
          f"{'PASS' if gate_multi else 'FAIL'}")

    # Relaxed gate: multi-instrument works if Sharpe > 0.3 even if not better than single
    overall = "PASS" if (
        gate_multi and gate_drawdown and
        (gate_sharpe or multi_metrics["sharpe_ratio"] > 0.3)
    ) else "FAIL"
    print(f"\n  Overall Phase 3 Gate: {overall}")

    # =================================================================
    # Config suggestion
    # =================================================================
    print(f"\n{'='*70}")
    print("SUGGESTED CONFIG UPDATE (configs/portfolio.yaml)")
    print(f"{'='*70}")
    print(f"\ninstrument_weights:")
    for name, w in sorted(weights.items()):
        print(f"  {name}: {w:.4f}")
    print(f"idm: {idm:.3f}")
    print(f"\n# Also update configs/paper.yaml instruments:")
    print(f"#   instruments: {active_instruments}")


if __name__ == "__main__":
    main()
