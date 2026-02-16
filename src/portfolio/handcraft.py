"""Carver's handcrafting method for forecast weights.

Instead of optimizing weights (which overfits with few data points),
Carver recommends "handcrafting":
1. Group correlated rules together
2. Equal weight within each group
3. Equal weight across groups

This is more robust than optimization for small accounts and
short histories.
"""

import numpy as np
import pandas as pd

from ..rules.scaling import calculate_fdm


def handcraft_weights(
    forecast_correlation: pd.DataFrame,
    correlation_threshold: float = 0.65,
) -> dict[str, float]:
    """Calculate handcrafted forecast weights.

    Algorithm:
    1. Build groups of correlated rules (corr > threshold)
    2. Equal weight within each group
    3. Equal weight across groups
    4. Normalize to sum to 1.0

    Args:
        forecast_correlation: Correlation matrix of forecasts
        correlation_threshold: Threshold for grouping rules

    Returns:
        Dict of rule_name -> weight
    """
    rules = list(forecast_correlation.columns)
    n_rules = len(rules)

    if n_rules == 0:
        return {}
    if n_rules == 1:
        return {rules[0]: 1.0}

    # Group correlated rules using simple clustering
    groups = _cluster_rules(forecast_correlation, correlation_threshold)

    # Equal weight across groups, equal weight within groups
    n_groups = len(groups)
    weights = {}
    for group in groups:
        group_weight = 1.0 / n_groups
        rule_weight = group_weight / len(group)
        for rule in group:
            weights[rule] = rule_weight

    return weights


def _cluster_rules(
    corr: pd.DataFrame,
    threshold: float,
) -> list[list[str]]:
    """Simple greedy clustering of correlated rules.

    Args:
        corr: Correlation matrix
        threshold: Grouping threshold

    Returns:
        List of groups (each group is a list of rule names)
    """
    rules = list(corr.columns)
    assigned = set()
    groups = []

    for rule in rules:
        if rule in assigned:
            continue

        # Start new group with this rule
        group = [rule]
        assigned.add(rule)

        # Find other rules correlated with this one
        for other in rules:
            if other in assigned:
                continue
            if abs(corr.loc[rule, other]) > threshold:
                group.append(other)
                assigned.add(other)

        groups.append(group)

    return groups


def compute_forecast_correlation(
    forecasts: dict[str, pd.Series],
    min_periods: int = 100,
) -> pd.DataFrame:
    """Compute correlation matrix between forecasts.

    Args:
        forecasts: Dict of rule_name -> forecast series
        min_periods: Minimum overlapping periods for valid correlation

    Returns:
        Correlation matrix DataFrame
    """
    df = pd.DataFrame(forecasts)
    return df.corr(min_periods=min_periods)


def handcraft_with_fdm(
    forecasts: dict[str, pd.Series],
    correlation_threshold: float = 0.65,
) -> tuple[dict[str, float], float]:
    """Compute handcrafted weights AND the corresponding FDM.

    This is the complete workflow for Phase 2 forecast combination:
    1. Compute forecast correlations
    2. Handcraft weights
    3. Calculate FDM from weights and correlations

    Args:
        forecasts: Dict of rule_name -> forecast series
        correlation_threshold: For grouping rules

    Returns:
        (weights dict, fdm float)
    """
    corr = compute_forecast_correlation(forecasts)
    weights = handcraft_weights(corr, correlation_threshold)

    # Calculate FDM
    weight_series = pd.Series(weights)
    # Align correlation matrix to weight series
    aligned_corr = corr.loc[weight_series.index, weight_series.index]
    fdm = calculate_fdm(aligned_corr, weight_series)

    return weights, fdm
