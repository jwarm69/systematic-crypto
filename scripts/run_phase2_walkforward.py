#!/usr/bin/env python3
"""Run constrained walk-forward Phase 2 tuning for one or more instruments."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.phase2_walkforward import walk_forward_weight_selection
from src.data.fetcher import fetch_and_cache
from src.rules.breakout import BreakoutRule
from src.rules.ewmac import EWMACRule
from src.rules.momentum import MomentumRule
from src.rules.scaling import calculate_fdm, combine_forecasts


INSTRUMENT_SYMBOLS = {
    "BTC": "BTC/USDC:USDC",
    "ETH": "ETH/USDC:USDC",
    "SOL": "SOL/USDC:USDC",
    "DOGE": "DOGE/USDC:USDC",
    "SHIB": "KSHIB/USDC:USDC",
    "ZEC": "ZEC/USDC:USDC",
}


def _make_rules() -> dict:
    return {
        "ewmac_8_32": EWMACRule(8, 32),
        "ewmac_16_64": EWMACRule(16, 64),
        "ewmac_32_128": EWMACRule(32, 128),
        "breakout_80": BreakoutRule(80),
        "momentum": MomentumRule(252),
    }


def _parse_instruments(raw: str) -> list[str]:
    instruments = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not instruments:
        raise ValueError("No instruments provided")
    unknown = [token for token in instruments if token not in INSTRUMENT_SYMBOLS]
    if unknown:
        raise ValueError(f"Unknown instrument(s): {', '.join(unknown)}")
    return instruments


def _full_sample_compare(
    prices: pd.Series,
    rules: dict,
    engine: BacktestEngine,
    tuned_weights: dict[str, float],
    equal_weights: dict[str, float],
) -> dict:
    forecasts = {
        name: rule.forecast(prices).reindex(prices.index).fillna(0.0) for name, rule in rules.items()
    }
    corr = pd.DataFrame(forecasts).corr().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)

    tuned_fdm = calculate_fdm(corr, pd.Series(tuned_weights))
    equal_fdm = calculate_fdm(corr, pd.Series(equal_weights))

    tuned_fc = combine_forecasts(forecasts, weights=tuned_weights, fdm=tuned_fdm)
    equal_fc = combine_forecasts(forecasts, weights=equal_weights, fdm=equal_fdm)

    tuned_metrics = engine.run(prices, tuned_fc)["metrics"]
    equal_metrics = engine.run(prices, equal_fc)["metrics"]

    return {
        "tuned_metrics": tuned_metrics,
        "equal_metrics": equal_metrics,
        "tuned_fdm": float(tuned_fdm),
        "equal_fdm": float(equal_fdm),
    }


def _run_instrument(args, instrument: str) -> dict:
    symbol = INSTRUMENT_SYMBOLS[instrument]
    print(f"\n{'=' * 84}")
    print(f"{instrument} ({symbol})")
    print(f"{'=' * 84}")

    data = fetch_and_cache(
        symbol=symbol,
        timeframe=args.timeframe,
        limit=args.bars,
        exchange_id=args.exchange,
        force_refresh=args.force_refresh,
    )
    if data.empty:
        raise RuntimeError(f"No data for {instrument}")

    prices = data["close"].astype(float).sort_index()
    print(f"Data: {len(prices)} bars from {prices.index[0]} to {prices.index[-1]}")

    engine = BacktestEngine(
        capital=args.capital,
        vol_target=args.vol_target,
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        buffer_fraction=args.buffer_fraction,
        max_leverage=args.max_leverage,
        timeframe=args.timeframe,
        funding_bps_per_8h=args.funding_bps_per_8h,
    )
    rules = _make_rules()

    wf = walk_forward_weight_selection(
        prices=prices,
        rules=rules,
        engine=engine,
        n_splits=args.n_splits,
        test_size=args.test_size,
        purge=args.purge,
        embargo=args.embargo,
        search_trials=args.search_trials,
        max_weight=args.max_rule_weight,
        min_active_rules=args.min_active_rules,
        min_rule_weight=args.min_rule_weight,
        seed=args.seed,
    )

    fold_cols = [
        "fold",
        "train_bars",
        "test_bars",
        "candidate_test_sharpe",
        "baseline_test_sharpe",
        "test_sharpe_delta",
        "candidate_test_max_dd",
        "baseline_test_max_dd",
    ]
    print("\nWalk-forward folds:")
    print(wf["folds"][fold_cols].to_string(index=False))

    summary = wf["summary"]
    print("\nWalk-forward summary:")
    print(f"  Avg candidate test Sharpe: {summary['avg_candidate_test_sharpe']:+.3f}")
    print(f"  Avg baseline test Sharpe:  {summary['avg_baseline_test_sharpe']:+.3f}")
    print(f"  Mean test Sharpe delta:    {summary['mean_test_sharpe_delta']:+.3f}")
    print(f"  Outperform fraction:       {summary['outperform_fraction']:.1%}")
    print(f"  Max DD worsening:          {summary['max_drawdown_worsening']:.1%}")
    print(f"  Candidate OOS Sharpe:      {summary['candidate_oos_sharpe']:+.3f}")
    print(f"  Baseline OOS Sharpe:       {summary['baseline_oos_sharpe']:+.3f}")

    tuned_weights = wf["recommended_weights"]
    print("\nRecommended weights:")
    for name, weight in sorted(tuned_weights.items(), key=lambda x: -x[1]):
        print(f"  {name:15s}: {weight:.4f}")

    full = _full_sample_compare(
        prices=prices,
        rules=rules,
        engine=engine,
        tuned_weights=tuned_weights,
        equal_weights=wf["equal_weights"],
    )
    tuned = full["tuned_metrics"]
    baseline = full["equal_metrics"]
    print("\nFull-sample reference (not for model selection):")
    print(f"  Tuned Sharpe:    {tuned['sharpe_ratio']:+.3f} (FDM={full['tuned_fdm']:.3f})")
    print(f"  Baseline Sharpe: {baseline['sharpe_ratio']:+.3f} (FDM={full['equal_fdm']:.3f})")
    print(f"  Tuned Return:    {tuned['total_return']:.1%}")
    print(f"  Baseline Return: {baseline['total_return']:.1%}")

    return {
        "instrument": instrument,
        "symbol": symbol,
        "bars": int(len(prices)),
        "start": str(prices.index[0]),
        "end": str(prices.index[-1]),
        "walk_forward_summary": summary,
        "recommended_weights": tuned_weights,
        "equal_weights": wf["equal_weights"],
        "full_sample_reference": {
            "tuned_metrics": tuned,
            "baseline_metrics": baseline,
            "tuned_fdm": full["tuned_fdm"],
            "equal_fdm": full["equal_fdm"],
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Constrained walk-forward Phase 2 tuner")
    parser.add_argument("--instruments", default="BTC", help="Comma-separated list, e.g. BTC,ETH,SOL")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--exchange", default="hyperliquid")
    parser.add_argument("--force-refresh", action="store_true")

    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--vol-target", type=float, default=0.12)
    parser.add_argument("--max-leverage", type=float, default=5.0)
    parser.add_argument("--buffer-fraction", type=float, default=0.10)
    parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--funding-bps-per-8h", type=float, default=0.0)

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--purge", type=int, default=24)
    parser.add_argument("--embargo", type=int, default=0)
    parser.add_argument("--search-trials", type=int, default=800)
    parser.add_argument("--max-rule-weight", type=float, default=0.50)
    parser.add_argument("--min-active-rules", type=int, default=3)
    parser.add_argument("--min-rule-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write structured report JSON",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        instruments = _parse_instruments(args.instruments)
    except ValueError as exc:
        print(f"Argument error: {exc}")
        return 2

    reports = []
    for instrument in instruments:
        try:
            report = _run_instrument(args, instrument)
            reports.append(report)
        except Exception as exc:  # noqa: BLE001
            print(f"\n{instrument}: failed - {exc}")
            return 1

    if len(reports) > 1:
        print(f"\n{'=' * 84}")
        print("Cross-asset summary:")
        print(f"{'=' * 84}")
        rows = []
        for rep in reports:
            wf = rep["walk_forward_summary"]
            rows.append(
                {
                    "instrument": rep["instrument"],
                    "wf_oos_sharpe": wf["candidate_oos_sharpe"],
                    "wf_delta": wf["mean_test_sharpe_delta"],
                    "wf_outperform": wf["outperform_fraction"],
                    "full_ref_sharpe": rep["full_sample_reference"]["tuned_metrics"]["sharpe_ratio"],
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))

    if args.output_json:
        payload = {
            "instruments": instruments,
            "reports": reports,
            "config": {
                "timeframe": args.timeframe,
                "bars": args.bars,
                "n_splits": args.n_splits,
                "test_size": args.test_size,
                "purge": args.purge,
                "embargo": args.embargo,
                "search_trials": args.search_trials,
                "max_rule_weight": args.max_rule_weight,
                "min_active_rules": args.min_active_rules,
                "min_rule_weight": args.min_rule_weight,
                "seed": args.seed,
            },
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
