"""Walk-forward weight selection for multi-rule Phase 2 systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ml.guardrails import purged_walk_forward_splits
from ..rules.base import AbstractTradingRule
from ..rules.scaling import calculate_fdm, combine_forecasts
from .engine import BacktestEngine


@dataclass
class WeightSearchResult:
    """Best in-sample weight vector found under constraints."""

    weights: dict[str, float]
    fdm: float
    score: float
    metrics: dict


def _apply_weight_cap(weights: pd.Series, max_weight: float) -> pd.Series:
    """Project non-negative weights back to simplex with an upper cap."""
    clipped = weights.clip(lower=0.0)
    if clipped.sum() <= 0:
        clipped[:] = 1.0 / len(clipped)
    clipped = clipped / clipped.sum()

    for _ in range(16):
        over_cap = clipped > max_weight
        if not over_cap.any():
            break
        capped = clipped.copy()
        capped[over_cap] = max_weight
        remainder = 1.0 - capped.sum()
        if remainder <= 1e-12:
            clipped = capped / capped.sum()
            break

        free = ~over_cap
        free_total = capped[free].sum()
        if free_total <= 0:
            capped[free] = remainder / free.sum()
        else:
            capped[free] = capped[free] + remainder * (capped[free] / free_total)
        clipped = capped

    clipped = clipped.clip(lower=0.0)
    if clipped.sum() <= 0:
        clipped[:] = 1.0 / len(clipped)
    return clipped / clipped.sum()


def _sample_weight_vector(
    rng: np.random.Generator,
    n_rules: int,
    max_weight: float,
    min_active_rules: int,
    min_rule_weight: float,
) -> np.ndarray | None:
    min_active_rules = int(max(1, min(min_active_rules, n_rules)))
    active_count = int(rng.integers(min_active_rules, n_rules + 1))

    if max_weight * active_count < 1.0:
        return None

    floor_budget = min_rule_weight * active_count
    if floor_budget >= 1.0:
        return None

    active_idx = rng.choice(n_rules, size=active_count, replace=False)
    draw = rng.dirichlet(np.ones(active_count))
    active_weights = min_rule_weight + (1.0 - floor_budget) * draw
    if active_weights.max() > max_weight:
        return None

    weights = np.zeros(n_rules, dtype=float)
    weights[active_idx] = active_weights
    return weights


def _evaluate_weights(
    prices: pd.Series,
    forecasts: dict[str, pd.Series],
    weights: dict[str, float],
    corr_matrix: pd.DataFrame,
    engine: BacktestEngine,
) -> tuple[float, float, dict, pd.Series]:
    fdm = calculate_fdm(corr_matrix, pd.Series(weights))
    combined = combine_forecasts(forecasts, weights=weights, fdm=fdm)
    result = engine.run(prices, combined)
    metrics = result["metrics"]
    score = float(metrics["sharpe_ratio"] - 0.15 * abs(metrics["max_drawdown"]))
    return score, fdm, metrics, result["portfolio_value"]


def constrained_weight_search(
    train_prices: pd.Series,
    train_forecasts: dict[str, pd.Series],
    engine: BacktestEngine,
    n_trials: int = 800,
    max_weight: float = 0.50,
    min_active_rules: int = 3,
    min_rule_weight: float = 0.05,
    seed: int = 42,
) -> WeightSearchResult:
    """Search for robust forecast weights in a bounded simplex."""
    if not train_forecasts:
        raise ValueError("train_forecasts is empty")
    if max_weight <= 0:
        raise ValueError("max_weight must be > 0")
    if min_rule_weight < 0:
        raise ValueError("min_rule_weight must be >= 0")

    rule_names = list(train_forecasts.keys())
    n_rules = len(rule_names)
    if max_weight * min_active_rules < 1.0:
        raise ValueError(
            "Infeasible constraints: max_weight * min_active_rules must be >= 1.0"
        )

    aligned = {
        name: fc.reindex(train_prices.index).fillna(0.0).astype(float)
        for name, fc in train_forecasts.items()
    }
    corr = pd.DataFrame(aligned).corr().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)

    rng = np.random.default_rng(seed)
    equal = np.array([1.0 / n_rules] * n_rules, dtype=float)
    equal_weights = dict(zip(rule_names, equal, strict=True))
    best_score, best_fdm, best_metrics, _ = _evaluate_weights(
        train_prices, aligned, equal_weights, corr, engine
    )
    best_weights = equal_weights

    max_attempts = max(1_000, n_trials * 8)
    attempts = 0
    accepted = 0

    while accepted < n_trials and attempts < max_attempts:
        attempts += 1
        sampled = _sample_weight_vector(
            rng=rng,
            n_rules=n_rules,
            max_weight=max_weight,
            min_active_rules=min_active_rules,
            min_rule_weight=min_rule_weight,
        )
        if sampled is None:
            continue

        weights = dict(zip(rule_names, sampled, strict=True))
        score, fdm, metrics, _ = _evaluate_weights(train_prices, aligned, weights, corr, engine)
        accepted += 1
        if score > best_score:
            best_score = score
            best_weights = weights
            best_fdm = fdm
            best_metrics = metrics

    return WeightSearchResult(
        weights=best_weights,
        fdm=float(best_fdm),
        score=float(best_score),
        metrics=best_metrics,
    )


def walk_forward_weight_selection(
    prices: pd.Series,
    rules: dict[str, AbstractTradingRule],
    engine: BacktestEngine,
    n_splits: int = 5,
    test_size: int | None = None,
    purge: int = 24,
    embargo: int = 0,
    search_trials: int = 800,
    max_weight: float = 0.50,
    min_active_rules: int = 3,
    min_rule_weight: float = 0.05,
    seed: int = 42,
) -> dict:
    """Select and evaluate constrained weights via expanding walk-forward CV."""
    if prices.empty:
        raise ValueError("prices is empty")
    if not rules:
        raise ValueError("rules is empty")

    aligned_prices = prices.dropna().astype(float).sort_index()
    rule_names = list(rules.keys())
    equal_weights = {name: 1.0 / len(rule_names) for name in rule_names}

    splits = purged_walk_forward_splits(
        n_samples=len(aligned_prices),
        n_splits=n_splits,
        test_size=test_size,
        purge=purge,
        embargo=embargo,
    )

    fold_rows: list[dict] = []
    chosen_weights: list[dict[str, float]] = []
    candidate_returns: list[pd.Series] = []
    baseline_returns: list[pd.Series] = []

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        train_prices = aligned_prices.iloc[train_idx]
        test_prices = aligned_prices.iloc[test_idx]
        if len(train_prices) < 200 or len(test_prices) < 50:
            continue

        history_end = int(test_idx[-1]) + 1
        history_prices = aligned_prices.iloc[:history_end]
        history_forecasts = {
            name: rule.forecast(history_prices).reindex(history_prices.index).fillna(0.0)
            for name, rule in rules.items()
        }
        train_forecasts = {
            name: fc.reindex(train_prices.index).fillna(0.0) for name, fc in history_forecasts.items()
        }
        test_forecasts = {
            name: fc.reindex(test_prices.index).fillna(0.0) for name, fc in history_forecasts.items()
        }

        optimized = constrained_weight_search(
            train_prices=train_prices,
            train_forecasts=train_forecasts,
            engine=engine,
            n_trials=search_trials,
            max_weight=max_weight,
            min_active_rules=min_active_rules,
            min_rule_weight=min_rule_weight,
            seed=seed + fold_num,
        )

        train_corr = pd.DataFrame(train_forecasts).corr().fillna(0.0)
        np.fill_diagonal(train_corr.values, 1.0)
        _, baseline_train_fdm, baseline_train_metrics, _ = _evaluate_weights(
            train_prices, train_forecasts, equal_weights, train_corr, engine
        )

        _, _, candidate_test_metrics, candidate_value = _evaluate_weights(
            test_prices, test_forecasts, optimized.weights, train_corr, engine
        )
        _, _, baseline_test_metrics, baseline_value = _evaluate_weights(
            test_prices, test_forecasts, equal_weights, train_corr, engine
        )

        candidate_returns.append(candidate_value.pct_change().dropna())
        baseline_returns.append(baseline_value.pct_change().dropna())
        chosen_weights.append(optimized.weights)

        fold_rows.append(
            {
                "fold": fold_num,
                "train_bars": len(train_prices),
                "test_bars": len(test_prices),
                "candidate_train_sharpe": float(optimized.metrics["sharpe_ratio"]),
                "baseline_train_sharpe": float(baseline_train_metrics["sharpe_ratio"]),
                "candidate_test_sharpe": float(candidate_test_metrics["sharpe_ratio"]),
                "baseline_test_sharpe": float(baseline_test_metrics["sharpe_ratio"]),
                "candidate_test_max_dd": float(candidate_test_metrics["max_drawdown"]),
                "baseline_test_max_dd": float(baseline_test_metrics["max_drawdown"]),
                "candidate_fdm": float(optimized.fdm),
                "baseline_fdm": float(baseline_train_fdm),
            }
        )

    if not fold_rows:
        raise ValueError("No valid walk-forward folds produced")

    folds_df = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    folds_df["test_sharpe_delta"] = (
        folds_df["candidate_test_sharpe"] - folds_df["baseline_test_sharpe"]
    )
    folds_df["test_drawdown_delta"] = (
        folds_df["candidate_test_max_dd"].abs() - folds_df["baseline_test_max_dd"].abs()
    )

    candidate_oos = pd.concat(candidate_returns).dropna()
    baseline_oos = pd.concat(baseline_returns).dropna()
    ann = float(np.sqrt(engine.bars_per_year))
    candidate_oos_sharpe = (
        float(candidate_oos.mean() / candidate_oos.std() * ann) if candidate_oos.std() > 0 else 0.0
    )
    baseline_oos_sharpe = (
        float(baseline_oos.mean() / baseline_oos.std() * ann) if baseline_oos.std() > 0 else 0.0
    )

    mean_weights = pd.DataFrame(chosen_weights).fillna(0.0).mean().reindex(rule_names, fill_value=0.0)
    mean_weights = _apply_weight_cap(mean_weights, max_weight=max_weight)
    if int((mean_weights > 0).sum()) < min_active_rules:
        top = mean_weights.sort_values(ascending=False).index[:min_active_rules]
        mean_weights = mean_weights.where(mean_weights.index.isin(top), 0.0)
        if mean_weights.sum() <= 0:
            mean_weights.loc[top] = 1.0
        mean_weights = _apply_weight_cap(mean_weights, max_weight=max_weight)

    summary = {
        "n_folds": int(len(folds_df)),
        "avg_candidate_test_sharpe": float(folds_df["candidate_test_sharpe"].mean()),
        "avg_baseline_test_sharpe": float(folds_df["baseline_test_sharpe"].mean()),
        "mean_test_sharpe_delta": float(folds_df["test_sharpe_delta"].mean()),
        "outperform_fraction": float((folds_df["test_sharpe_delta"] > 0).mean()),
        "max_drawdown_worsening": float(folds_df["test_drawdown_delta"].max()),
        "candidate_oos_sharpe": candidate_oos_sharpe,
        "baseline_oos_sharpe": baseline_oos_sharpe,
    }

    return {
        "folds": folds_df,
        "summary": summary,
        "recommended_weights": {k: float(v) for k, v in mean_weights.items()},
        "equal_weights": equal_weights,
    }
