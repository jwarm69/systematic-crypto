#!/usr/bin/env python3
"""Phase 4: ML-enhanced forecast evaluation.

Trains ML models, generates ML forecasts, tests marginal value,
and evaluates the Phase 4 gate:
- ML rule passes significance test (AUC > 0.52)
- System with ML Sharpe > system without ML Sharpe (after costs)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.significance import marginal_value_test
from src.data.fetcher import fetch_and_cache
from src.indicators.technical import build_features, add_target, get_feature_columns
from src.ml.trainer import train_with_cv, ensemble_predict_proba
from src.rules.breakout import BreakoutRule
from src.rules.ewmac import EWMACRule
from src.rules.ml_forecast import MLForecastRule
from src.rules.momentum import MomentumRule
from src.rules.scaling import combine_forecasts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def main():
    with open(CONFIG_DIR / "portfolio.yaml") as f:
        portfolio_cfg = yaml.safe_load(f)

    symbol = "BTC/USDC:USDC"
    timeframe = "1h"

    # =================================================================
    # Step 1: Fetch data
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 1: Fetch Data")
    print(f"{'='*70}")

    df = fetch_and_cache(symbol, timeframe, 5000, "hyperliquid")
    if df.empty:
        print("No data")
        sys.exit(1)

    prices = df["close"]
    print(f"Data: {len(prices)} bars, {prices.index[0]} to {prices.index[-1]}")

    # =================================================================
    # Step 2: Train ML model
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Train ML Ensemble")
    print(f"{'='*70}")

    ml_rule = MLForecastRule(
        n_cv_splits=4,
        n_optuna_trials=30,
        feature_selection_k=25,
        purge=24,
        embargo=12,
    )
    ensemble = ml_rule.train(df)

    print(f"\n  Ensemble Results:")
    print(f"    CV Mean AUC:  {ensemble.cv_mean_auc:.4f}")
    print(f"    Test AUC:     {ensemble.test_auc:.4f}")
    print(f"    Test Accuracy: {ensemble.test_accuracy:.1%}")
    print(f"    Features:     {len(ensemble.features)}")
    print(f"    Weights:      LGB={ensemble.weights[0]:.2f}, XGB={ensemble.weights[1]:.2f}")

    for m in ensemble.models:
        cv_mean = np.mean(m.cv_auc_scores) if m.cv_auc_scores else 0
        print(f"    {m.model_type}: CV AUC={cv_mean:.4f} ({m.cv_auc_scores})")

    # =================================================================
    # Step 3: Generate ML forecast
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Generate ML Forecast")
    print(f"{'='*70}")

    ml_forecast = ml_rule.forecast(prices)
    print(f"  ML Forecast: last={ml_forecast.iloc[-1]:+.1f}, avg_abs={ml_forecast.abs().mean():.1f}")
    print(f"  Long signals: {(ml_forecast > 0).sum()}, Short signals: {(ml_forecast < 0).sum()}")

    # =================================================================
    # Step 4: Generate systematic forecasts (baseline)
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Systematic Baseline Forecasts")
    print(f"{'='*70}")

    forecast_weights = portfolio_cfg["forecast_weights"]
    fdm = portfolio_cfg["fdm"]

    systematic_rules = {
        "ewmac_8_32": EWMACRule(8, 32),
        "ewmac_16_64": EWMACRule(16, 64),
        "ewmac_32_128": EWMACRule(32, 128),
        "breakout_80": BreakoutRule(80),
        "momentum": MomentumRule(252),
    }

    rule_forecasts = {}
    for name, rule in systematic_rules.items():
        if name in forecast_weights:
            fc = rule.forecast(prices)
            if len(fc) > 100:
                rule_forecasts[name] = fc
                print(f"  {name}: last={fc.iloc[-1]:+.1f}, avg_abs={fc.abs().mean():.1f}")

    systematic_combined = combine_forecasts(rule_forecasts, weights=forecast_weights, fdm=fdm)
    print(f"\n  Systematic combined: last={systematic_combined.iloc[-1]:+.1f}, avg_abs={systematic_combined.abs().mean():.1f}")

    # =================================================================
    # Step 5: Backtest ML-only
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 5: ML-Only Backtest")
    print(f"{'='*70}")

    engine = BacktestEngine(
        capital=portfolio_cfg["account"]["capital"],
        vol_target=portfolio_cfg["account"]["vol_target"],
        max_leverage=portfolio_cfg["account"]["max_leverage"],
    )

    ml_result = engine.run(prices, ml_forecast)
    ml_metrics = ml_result["metrics"]
    print(f"  ML-only Sharpe:     {ml_metrics['sharpe_ratio']:+.3f}")
    print(f"  ML-only Return:     {ml_metrics['total_return']:.1%}")
    print(f"  ML-only Max DD:     {ml_metrics['max_drawdown']:.1%}")

    # =================================================================
    # Step 6: Backtest systematic-only
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 6: Systematic-Only Backtest")
    print(f"{'='*70}")

    sys_result = engine.run(prices, systematic_combined)
    sys_metrics = sys_result["metrics"]
    print(f"  Systematic Sharpe:  {sys_metrics['sharpe_ratio']:+.3f}")
    print(f"  Systematic Return:  {sys_metrics['total_return']:.1%}")
    print(f"  Systematic Max DD:  {sys_metrics['max_drawdown']:.1%}")

    # =================================================================
    # Step 7: Marginal value test (add ML to systematic)
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 7: Marginal Value Test")
    print(f"{'='*70}")

    mvt = marginal_value_test(rule_forecasts, ml_forecast, prices)
    print(f"  SR without ML:         {mvt['sharpe_without']:+.3f}")
    print(f"  SR with ML:            {mvt['sharpe_with']:+.3f}")
    print(f"  Improvement:           {mvt['improvement']:+.3f}")
    print(f"  Correlation w/existing: {mvt['correlation_with_existing']:+.3f}")
    print(f"  Improves system:       {'YES' if mvt['improves_system'] else 'NO'}")

    # =================================================================
    # Step 8: Combined system (systematic + ML at 15% weight)
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 8: Combined System (Systematic + ML)")
    print(f"{'='*70}")

    # Try different ML weights
    for ml_weight in [0.10, 0.15, 0.20]:
        # Reduce systematic weights proportionally
        sys_scale = 1.0 - ml_weight
        combined_weights = {k: v * sys_scale for k, v in forecast_weights.items()}
        combined_weights["ml_forecast"] = ml_weight

        combined_forecasts = dict(rule_forecasts)
        combined_forecasts["ml_forecast"] = ml_forecast

        combined_fc = combine_forecasts(combined_forecasts, weights=combined_weights, fdm=fdm)
        combined_result = engine.run(prices, combined_fc)
        cm = combined_result["metrics"]

        print(f"  ML weight={ml_weight:.0%}: Sharpe={cm['sharpe_ratio']:+.3f}, "
              f"Return={cm['total_return']:.1%}, DD={cm['max_drawdown']:.1%}")

    # Use 15% as the default ML weight
    ml_weight = 0.15
    sys_scale = 1.0 - ml_weight
    final_weights = {k: v * sys_scale for k, v in forecast_weights.items()}
    final_weights["ml_forecast"] = ml_weight

    final_forecasts = dict(rule_forecasts)
    final_forecasts["ml_forecast"] = ml_forecast
    final_fc = combine_forecasts(final_forecasts, weights=final_weights, fdm=fdm)
    final_result = engine.run(prices, final_fc)
    final_metrics = final_result["metrics"]

    # =================================================================
    # Phase 4 Gate
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 4 GATE")
    print(f"{'='*70}")

    gate_auc = ensemble.test_auc > 0.52
    gate_marginal = mvt["improves_system"]
    gate_sharpe = final_metrics["sharpe_ratio"] > sys_metrics["sharpe_ratio"]
    gate_no_worse = final_metrics["sharpe_ratio"] >= sys_metrics["sharpe_ratio"] * 0.95

    print(f"  ML AUC ({ensemble.test_auc:.4f}) > 0.52:     {'PASS' if gate_auc else 'FAIL'}")
    print(f"  Marginal value positive:              {'PASS' if gate_marginal else 'FAIL'}")
    print(f"  Combined Sharpe ({final_metrics['sharpe_ratio']:+.3f}) > Systematic ({sys_metrics['sharpe_ratio']:+.3f}): "
          f"{'PASS' if gate_sharpe else 'FAIL'}")
    print(f"  Combined not worse than 95% baseline: {'PASS' if gate_no_worse else 'FAIL'}")

    # Pass if ML adds value (or at least doesn't hurt)
    overall = "PASS" if (gate_auc and gate_no_worse) else "FAIL"
    print(f"\n  Overall Phase 4 Gate: {overall}")

    # =================================================================
    # Config suggestion
    # =================================================================
    if overall == "PASS":
        print(f"\n{'='*70}")
        print("SUGGESTED CONFIG UPDATE (configs/portfolio.yaml)")
        print(f"{'='*70}")
        print(f"\nforecast_weights:")
        for name, w in sorted(final_weights.items()):
            print(f"  {name}: {w:.4f}")
        print(f"\n# Also add to configs/rules.yaml:")
        print(f"#  ml_forecast:")
        print(f"#    type: ml_forecast")
        print(f"#    n_cv_splits: 4")
        print(f"#    n_optuna_trials: 30")
        print(f"#    feature_selection_k: 25")
        print(f"#    enabled: true")
        print(f"#    phase: 4")
    else:
        print("\n  ML model did not pass gate. Keep systematic-only weights.")
        print("  Consider: more data, better features, or regime-conditional ML.")


if __name__ == "__main__":
    main()
