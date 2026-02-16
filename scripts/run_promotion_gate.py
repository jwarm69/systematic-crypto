#!/usr/bin/env python3
"""Run ML promotion gate against baseline forecasts."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.ml.guardrails import evaluate_candidate_vs_baseline, promotion_gate


GATE_PROFILES = {
    "lenient": {
        "min_mean_sharpe_delta": 0.03,
        "min_outperform_fraction": 0.70,
        "max_drawdown_worsening": 0.05,
    },
    "balanced": {
        "min_mean_sharpe_delta": 0.08,
        "min_outperform_fraction": 0.80,
        "max_drawdown_worsening": 0.03,
    },
    "strict": {
        "min_mean_sharpe_delta": 0.15,
        "min_outperform_fraction": 0.80,
        "max_drawdown_worsening": 0.02,
    },
}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format for {path} (expected csv or parquet)")


def _load_series(path: Path, value_column: str, timestamp_column: str | None) -> pd.Series:
    df = _read_table(path)

    if timestamp_column:
        if timestamp_column not in df.columns:
            raise ValueError(f"{path}: timestamp column '{timestamp_column}' not found")
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
        df = df.set_index(timestamp_column)
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    if value_column not in df.columns:
        raise ValueError(f"{path}: value column '{value_column}' not found")

    series = df[value_column].astype(float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run walk-forward ML promotion gate")
    parser.add_argument("--prices-file", required=True, help="CSV/parquet with timestamp + price column")
    parser.add_argument("--baseline-file", required=True, help="CSV/parquet with baseline forecast series")
    parser.add_argument("--candidate-file", required=True, help="CSV/parquet with candidate forecast series")
    parser.add_argument(
        "--funding-file",
        default=None,
        help="Optional CSV/parquet with per-bar funding rates",
    )

    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column name in inputs")
    parser.add_argument("--price-column", default="close", help="Price column name")
    parser.add_argument("--baseline-column", default="forecast", help="Baseline forecast column name")
    parser.add_argument("--candidate-column", default="forecast", help="Candidate forecast column name")
    parser.add_argument("--funding-column", default="funding_rate", help="Funding rate column name")

    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--vol-target", type=float, default=0.12)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--buffer-fraction", type=float, default=0.10)
    parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--funding-bps-per-8h", type=float, default=0.0)
    parser.add_argument("--timeframe", default="1h")

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--purge", type=int, default=0)
    parser.add_argument("--embargo", type=int, default=0)
    parser.add_argument(
        "--gate-profile",
        choices=sorted(GATE_PROFILES.keys()),
        default="balanced",
        help="Threshold preset for promotion decision",
    )

    parser.add_argument("--min-mean-sharpe-delta", type=float, default=None)
    parser.add_argument("--min-outperform-fraction", type=float, default=None)
    parser.add_argument("--max-drawdown-worsening", type=float, default=None)

    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write structured JSON report",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    profile = GATE_PROFILES[args.gate_profile]
    min_mean_sharpe_delta = (
        args.min_mean_sharpe_delta
        if args.min_mean_sharpe_delta is not None
        else profile["min_mean_sharpe_delta"]
    )
    min_outperform_fraction = (
        args.min_outperform_fraction
        if args.min_outperform_fraction is not None
        else profile["min_outperform_fraction"]
    )
    max_drawdown_worsening = (
        args.max_drawdown_worsening
        if args.max_drawdown_worsening is not None
        else profile["max_drawdown_worsening"]
    )

    prices = _load_series(Path(args.prices_file), args.price_column, args.timestamp_column)
    baseline = _load_series(Path(args.baseline_file), args.baseline_column, args.timestamp_column)
    candidate = _load_series(Path(args.candidate_file), args.candidate_column, args.timestamp_column)

    funding_rates = None
    if args.funding_file:
        funding_rates = _load_series(Path(args.funding_file), args.funding_column, args.timestamp_column)

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

    evaluation = evaluate_candidate_vs_baseline(
        prices=prices,
        baseline_forecasts=baseline,
        candidate_forecasts=candidate,
        engine=engine,
        n_splits=args.n_splits,
        test_size=args.test_size,
        purge=args.purge,
        embargo=args.embargo,
    )
    decision = promotion_gate(
        evaluation=evaluation,
        min_mean_sharpe_delta=min_mean_sharpe_delta,
        min_outperform_fraction=min_outperform_fraction,
        max_drawdown_worsening=max_drawdown_worsening,
    )

    print("\n=== Model Promotion Gate ===")
    print(
        f"Profile: {args.gate_profile} "
        f"(mean_delta>={min_mean_sharpe_delta:.3f}, "
        f"outperform>={min_outperform_fraction:.1%}, "
        f"dd_worsen<={max_drawdown_worsening:.1%})"
    )
    print(f"Approved: {decision.approved}")
    print(f"Mean Sharpe Delta:   {decision.summary['mean_sharpe_delta']:.3f}")
    print(f"Outperform Fraction: {decision.summary['outperform_fraction']:.1%}")
    print(f"Max DD Worsening:    {decision.summary['max_drawdown_worsening']:.1%}")
    print("Reasons:")
    for reason in decision.reasons:
        print(f"  - {reason}")

    splits = evaluation["splits"]
    print("\nSplit Summary:")
    print(
        splits[
            ["split", "baseline_sharpe", "candidate_sharpe", "sharpe_delta", "drawdown_delta"]
        ].to_string(index=False)
    )

    if args.output_json:
        payload = {
            "approved": decision.approved,
            "reasons": decision.reasons,
            "summary": decision.summary,
            "splits": splits.to_dict(orient="records"),
            "config": {
                "timeframe": args.timeframe,
                "n_splits": args.n_splits,
                "test_size": args.test_size,
                "purge": args.purge,
                "embargo": args.embargo,
                "gate_profile": args.gate_profile,
                "min_mean_sharpe_delta": min_mean_sharpe_delta,
                "min_outperform_fraction": min_outperform_fraction,
                "max_drawdown_worsening": max_drawdown_worsening,
            },
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote report: {output_path}")

    return 0 if decision.approved else 2


if __name__ == "__main__":
    raise SystemExit(main())
