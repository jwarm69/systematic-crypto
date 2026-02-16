"""Walk-forward and promotion guardrails for ML forecasts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestEngine


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    test_size: int | None = None,
    purge: int = 0,
    embargo: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding walk-forward splits with boundary purge/embargo.

    Train window is always strictly before the test window to avoid look-ahead.
    `purge` and `embargo` remove observations near the train/test boundary.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if n_splits <= 0:
        raise ValueError("n_splits must be > 0")

    split_test_size = test_size or max(1, n_samples // (n_splits + 1))
    initial_train = n_samples - (n_splits * split_test_size)
    if initial_train <= 0:
        raise ValueError("Not enough samples for requested splits/test_size")

    gap = max(0, purge) + max(0, embargo)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for split in range(n_splits):
        test_start = initial_train + (split * split_test_size)
        test_end = min(test_start + split_test_size, n_samples)
        if test_start >= n_samples:
            break

        train_end = max(0, test_start - gap)
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))

    if not splits:
        raise ValueError("No valid walk-forward splits produced")
    return splits


@dataclass
class PromotionDecision:
    """Model promotion decision relative to a baseline."""

    approved: bool
    reasons: list[str]
    summary: dict[str, float]


def evaluate_candidate_vs_baseline(
    prices: pd.Series,
    baseline_forecasts: pd.Series,
    candidate_forecasts: pd.Series,
    engine: BacktestEngine,
    n_splits: int = 5,
    test_size: int | None = None,
    purge: int = 0,
    embargo: int = 0,
) -> dict:
    """Evaluate candidate forecasts against baseline in walk-forward splits."""
    common_idx = prices.index.intersection(baseline_forecasts.index).intersection(candidate_forecasts.index)
    prices = prices.loc[common_idx]
    baseline_forecasts = baseline_forecasts.loc[common_idx]
    candidate_forecasts = candidate_forecasts.loc[common_idx]

    splits = purged_walk_forward_splits(
        n_samples=len(common_idx),
        n_splits=n_splits,
        test_size=test_size,
        purge=purge,
        embargo=embargo,
    )

    split_rows = []
    for i, (_, test_idx) in enumerate(splits, start=1):
        test_prices = prices.iloc[test_idx]
        base_fc = baseline_forecasts.iloc[test_idx]
        cand_fc = candidate_forecasts.iloc[test_idx]

        base_metrics = engine.run(test_prices, base_fc)["metrics"]
        cand_metrics = engine.run(test_prices, cand_fc)["metrics"]
        split_rows.append(
            {
                "split": i,
                "baseline_sharpe": float(base_metrics["sharpe_ratio"]),
                "candidate_sharpe": float(cand_metrics["sharpe_ratio"]),
                "baseline_max_drawdown": float(base_metrics["max_drawdown"]),
                "candidate_max_drawdown": float(cand_metrics["max_drawdown"]),
            }
        )

    split_df = pd.DataFrame(split_rows)
    split_df["sharpe_delta"] = split_df["candidate_sharpe"] - split_df["baseline_sharpe"]
    split_df["drawdown_delta"] = (
        split_df["candidate_max_drawdown"].abs() - split_df["baseline_max_drawdown"].abs()
    )

    return {
        "splits": split_df,
        "mean_sharpe_delta": float(split_df["sharpe_delta"].mean()),
        "median_sharpe_delta": float(split_df["sharpe_delta"].median()),
        "outperform_fraction": float((split_df["sharpe_delta"] > 0).mean()),
        "max_drawdown_worsening": float(split_df["drawdown_delta"].max()),
    }


def promotion_gate(
    evaluation: dict,
    min_mean_sharpe_delta: float = 0.05,
    min_outperform_fraction: float = 0.60,
    max_drawdown_worsening: float = 0.03,
) -> PromotionDecision:
    """Apply objective thresholds before promoting an ML model."""
    reasons: list[str] = []
    approved = True

    mean_delta = float(evaluation["mean_sharpe_delta"])
    outperform_fraction = float(evaluation["outperform_fraction"])
    dd_worsening = float(evaluation["max_drawdown_worsening"])

    if mean_delta < min_mean_sharpe_delta:
        approved = False
        reasons.append(
            f"Mean Sharpe delta {mean_delta:.3f} below threshold {min_mean_sharpe_delta:.3f}"
        )

    if outperform_fraction < min_outperform_fraction:
        approved = False
        reasons.append(
            f"Outperform fraction {outperform_fraction:.1%} below threshold {min_outperform_fraction:.1%}"
        )

    if dd_worsening > max_drawdown_worsening:
        approved = False
        reasons.append(
            f"Drawdown worsened by {dd_worsening:.1%} above limit {max_drawdown_worsening:.1%}"
        )

    if approved:
        reasons.append("Promotion criteria satisfied")

    summary = {
        "mean_sharpe_delta": mean_delta,
        "outperform_fraction": outperform_fraction,
        "max_drawdown_worsening": dd_worsening,
    }
    return PromotionDecision(approved=approved, reasons=reasons, summary=summary)

