"""Statistical significance testing for trading rules.

Simons insisted that every signal must pass statistical significance
before entering the system. No signal on intuition alone.

Key tests:
- Information Coefficient (IC): correlation between forecast and future returns
- t-statistic and p-value for IC
- Marginal value test: does adding this rule improve the system?
"""

import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(
    forecasts: pd.Series,
    returns: pd.Series,
    lag: int = 1,
) -> tuple[float, float, float]:
    """Calculate Information Coefficient (rank correlation between forecast and return).

    IC measures how well forecasts predict future returns.
    Rank correlation (Spearman) is more robust than Pearson.

    Args:
        forecasts: Forecast series
        returns: Return series
        lag: Number of bars to lag returns (1 = next-bar return)

    Returns:
        (ic, t_stat, p_value)
    """
    # Align: forecast at time t predicts return from t to t+lag
    shifted_returns = returns.shift(-lag)

    # Drop NaN
    valid = pd.DataFrame({"forecast": forecasts, "return": shifted_returns}).dropna()

    if len(valid) < 30:
        return 0.0, 0.0, 1.0

    # Spearman rank correlation
    ic, p_value = stats.spearmanr(valid["forecast"], valid["return"])

    # t-statistic
    n = len(valid)
    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 1 else 0

    return float(ic), float(t_stat), float(p_value)


def rule_significance_report(
    forecasts: pd.Series,
    prices: pd.Series,
    name: str = "",
) -> dict:
    """Generate comprehensive significance report for a single rule.

    Args:
        forecasts: Scaled forecast series [-20, +20]
        prices: Close price series
        name: Rule name

    Returns:
        Dict with significance metrics
    """
    returns = prices.pct_change()

    # IC at different horizons
    ic_1, t_1, p_1 = information_coefficient(forecasts, returns, lag=1)
    ic_4, t_4, p_4 = information_coefficient(forecasts, returns, lag=4)
    ic_24, t_24, p_24 = information_coefficient(forecasts, returns, lag=24)

    # Forecast-weighted returns (simple backtest proxy)
    weighted_returns = forecasts.shift(1) * returns / 10.0  # Normalize by avg forecast
    weighted_returns = weighted_returns.dropna()

    # Sharpe of forecast-weighted returns
    if len(weighted_returns) > 0 and weighted_returns.std() > 0:
        sr = weighted_returns.mean() / weighted_returns.std() * np.sqrt(365 * 24)
    else:
        sr = 0.0

    # Forecast statistics
    avg_abs = forecasts.abs().mean()
    forecast_autocorr = forecasts.autocorr(lag=1) if len(forecasts) > 10 else 0

    return {
        "name": name,
        "ic_1h": ic_1,
        "ic_4h": ic_4,
        "ic_24h": ic_24,
        "t_stat_1h": t_1,
        "p_value_1h": p_1,
        "t_stat_4h": t_4,
        "p_value_4h": p_4,
        "sharpe_ratio": sr,
        "avg_abs_forecast": avg_abs,
        "forecast_autocorr": forecast_autocorr,
        "n_observations": len(forecasts),
        "significant_1h": p_1 < 0.05,
        "significant_4h": p_4 < 0.05,
    }


def marginal_value_test(
    existing_forecasts: dict[str, pd.Series],
    new_forecast: pd.Series,
    prices: pd.Series,
    existing_weights: dict[str, float] | None = None,
) -> dict:
    """Test if adding a new rule improves the combined system.

    Computes:
    1. Sharpe of combined system WITHOUT new rule
    2. Sharpe of combined system WITH new rule (equal-weight the new rule in)
    3. Improvement and significance

    Args:
        existing_forecasts: Current rules (name -> forecast)
        new_forecast: Candidate rule's forecast
        prices: Price series
        existing_weights: Current weights (None = equal weight)

    Returns:
        Dict with improvement metrics
    """
    returns = prices.pct_change()

    # Current system forecast (without new rule)
    if existing_weights is None:
        existing_weights = {k: 1.0 / len(existing_forecasts) for k in existing_forecasts}

    index = returns.index
    current_combined = pd.Series(0.0, index=index)
    for name, fc in existing_forecasts.items():
        w = existing_weights.get(name, 0)
        current_combined += w * fc.reindex(index, fill_value=0)

    # New system forecast (with new rule added at equal weight)
    n_rules = len(existing_forecasts) + 1
    new_weights = {k: 1.0 / n_rules for k in existing_forecasts}
    new_weight_for_candidate = 1.0 / n_rules

    new_combined = pd.Series(0.0, index=index)
    for name, fc in existing_forecasts.items():
        new_combined += new_weights[name] * fc.reindex(index, fill_value=0)
    new_combined += new_weight_for_candidate * new_forecast.reindex(index, fill_value=0)

    # Compare Sharpe ratios
    def _sharpe(fc):
        wr = (fc.shift(1) * returns / 10.0).dropna()
        if len(wr) == 0 or wr.std() == 0:
            return 0.0
        return wr.mean() / wr.std() * np.sqrt(365 * 24)

    sr_without = _sharpe(current_combined)
    sr_with = _sharpe(new_combined)

    # Correlation of new rule with existing combined
    valid = pd.DataFrame({"existing": current_combined, "new": new_forecast}).dropna()
    if len(valid) > 30:
        corr_with_existing = valid["existing"].corr(valid["new"])
    else:
        corr_with_existing = 0.0

    improvement = sr_with - sr_without

    return {
        "sharpe_without": sr_without,
        "sharpe_with": sr_with,
        "improvement": improvement,
        "improves_system": improvement > 0,
        "correlation_with_existing": corr_with_existing,
    }
