"""Walk-forward cross-validation with purging.

Prevents look-ahead bias by using strict temporal ordering:
train on past data, skip a purge window, test on future data.

This is the gold standard for evaluating trading strategies.
Reference: Lopez de Prado, "Advances in Financial Machine Learning", Ch 7.
"""

import logging

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestEngine
from ..rules.base import AbstractTradingRule

logger = logging.getLogger(__name__)


def purged_walk_forward_cv(
    prices: pd.Series,
    rule: AbstractTradingRule,
    n_splits: int = 5,
    train_fraction: float = 0.6,
    purge_bars: int = 24,  # 1 day for hourly data
    capital: float = 5000,
    vol_target: float = 0.12,
    **rule_kwargs,
) -> dict:
    """Run purged walk-forward cross-validation.

    Splits data into n temporal folds. For each fold:
    1. Train: compute forecast scaling on training data
    2. Purge: skip purge_bars to prevent leakage
    3. Test: evaluate on out-of-sample data

    Args:
        prices: Full price series
        rule: Trading rule to evaluate
        n_splits: Number of CV folds
        train_fraction: Fraction of each fold used for training
        purge_bars: Bars to skip between train and test
        capital: Backtest capital
        vol_target: Backtest vol target
        **rule_kwargs: Additional kwargs passed to rule.forecast()

    Returns:
        Dict with per-fold and aggregate metrics
    """
    n = len(prices)
    fold_size = n // n_splits
    train_size = int(fold_size * train_fraction)

    fold_results = []
    all_oos_returns = []

    for fold in range(n_splits):
        fold_start = fold * fold_size
        fold_end = min(fold_start + fold_size, n)

        train_end = fold_start + train_size
        test_start = train_end + purge_bars
        test_end = fold_end

        if test_start >= test_end or test_end - test_start < 50:
            continue

        train_prices = prices.iloc[fold_start:train_end]
        test_prices = prices.iloc[test_start:test_end]

        # Generate forecast on training data (for scaling calibration)
        train_forecast = rule.forecast(train_prices, **rule_kwargs)

        # Generate forecast on test data
        # Use full history up to test point for better indicators,
        # but only evaluate performance on test period
        full_prices_to_test = prices.iloc[fold_start:test_end]
        full_forecast = rule.forecast(full_prices_to_test, **rule_kwargs)

        # Extract only the test period forecast
        test_forecast = full_forecast.loc[test_prices.index[0]:test_prices.index[-1]]

        if len(test_forecast) < 50:
            continue

        # Run backtest on test period
        engine = BacktestEngine(capital=capital, vol_target=vol_target)
        result = engine.run(test_prices, test_forecast)

        fold_metrics = result["metrics"]
        fold_metrics["fold"] = fold
        fold_metrics["train_bars"] = len(train_prices)
        fold_metrics["test_bars"] = len(test_prices)
        fold_results.append(fold_metrics)

        # Collect OOS returns for aggregate stats
        oos_returns = result["portfolio_value"].pct_change().dropna()
        all_oos_returns.append(oos_returns)

        logger.info(
            f"Fold {fold}: Sharpe={fold_metrics['sharpe_ratio']:.3f} "
            f"Return={fold_metrics['total_return']:.1%} "
            f"DD={fold_metrics['max_drawdown']:.1%}"
        )

    if not fold_results:
        return {"error": "No valid folds", "folds": []}

    # Aggregate metrics
    all_oos = pd.concat(all_oos_returns)
    ann_factor = np.sqrt(365 * 24)

    aggregate = {
        "rule_name": rule.name,
        "n_folds": len(fold_results),
        "avg_sharpe": np.mean([f["sharpe_ratio"] for f in fold_results]),
        "std_sharpe": np.std([f["sharpe_ratio"] for f in fold_results]),
        "min_sharpe": np.min([f["sharpe_ratio"] for f in fold_results]),
        "max_sharpe": np.max([f["sharpe_ratio"] for f in fold_results]),
        "avg_return": np.mean([f["total_return"] for f in fold_results]),
        "avg_max_dd": np.mean([f["max_drawdown"] for f in fold_results]),
        "worst_dd": np.min([f["max_drawdown"] for f in fold_results]),
        "oos_sharpe": (all_oos.mean() / all_oos.std() * ann_factor) if all_oos.std() > 0 else 0,
        "total_oos_bars": len(all_oos),
        "folds": fold_results,
    }

    # Is the rule consistently profitable?
    profitable_folds = sum(1 for f in fold_results if f["sharpe_ratio"] > 0)
    aggregate["profitable_folds"] = profitable_folds
    aggregate["consistency"] = profitable_folds / len(fold_results)

    return aggregate


def compare_rules_cv(
    prices: pd.Series,
    rules: dict[str, AbstractTradingRule],
    n_splits: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """Compare multiple rules using walk-forward CV.

    Args:
        prices: Price series
        rules: Dict of rule_name -> rule instance
        n_splits: CV folds
        **kwargs: Passed to purged_walk_forward_cv

    Returns:
        DataFrame comparing rules across metrics
    """
    results = []
    for name, rule in rules.items():
        logger.info(f"\n{'='*40}\nEvaluating: {name}\n{'='*40}")
        cv_result = purged_walk_forward_cv(prices, rule, n_splits=n_splits, **kwargs)
        if "error" not in cv_result:
            results.append({
                "rule": name,
                "avg_sharpe": cv_result["avg_sharpe"],
                "std_sharpe": cv_result["std_sharpe"],
                "oos_sharpe": cv_result["oos_sharpe"],
                "avg_return": cv_result["avg_return"],
                "worst_dd": cv_result["worst_dd"],
                "consistency": cv_result["consistency"],
                "n_folds": cv_result["n_folds"],
            })

    return pd.DataFrame(results).sort_values("oos_sharpe", ascending=False)
