"""Instrument weight allocation and Instrument Diversification Multiplier (IDM).

Carver's approach:
- Equal weights for small portfolios (<5 instruments)
- IDM = 1 / sqrt(w' * C * w) where C is the instrument correlation matrix
- IDM boosts position size when instruments are uncorrelated (diversification benefit)
- Single instrument: IDM = 1.0
- 3 uncorrelated instruments: IDM ~1.7
- 3 highly correlated: IDM ~1.0-1.2
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equal_weights(instruments: list[str]) -> dict[str, float]:
    """Equal weight allocation across instruments."""
    n = len(instruments)
    if n == 0:
        return {}
    w = 1.0 / n
    return {inst: w for inst in instruments}


def compute_instrument_correlation(
    returns_dict: dict[str, pd.Series],
    min_overlap: int = 100,
) -> pd.DataFrame:
    """Compute correlation matrix from instrument return series.

    Args:
        returns_dict: instrument_name -> return series
        min_overlap: Minimum overlapping observations required

    Returns:
        Correlation matrix as DataFrame
    """
    instruments = list(returns_dict.keys())
    n = len(instruments)

    if n <= 1:
        return pd.DataFrame(1.0, index=instruments, columns=instruments)

    # Align all return series
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna(how="all")

    # Compute pairwise correlation (handles missing data properly)
    corr = returns_df.corr(min_periods=min_overlap)

    # Fill NaN correlations with a conservative estimate (high positive)
    corr = corr.fillna(0.5)

    # Ensure diagonal is 1
    np.fill_diagonal(corr.values, 1.0)

    # Ensure positive semi-definite (nearest PSD if needed)
    corr = _nearest_psd(corr)

    return corr


def calculate_idm(
    weights: dict[str, float],
    correlation_matrix: pd.DataFrame,
    cap: float = 2.5,
) -> float:
    """Calculate Instrument Diversification Multiplier.

    IDM = 1 / sqrt(w' * C * w)

    The IDM scales up position sizes to account for the diversification
    benefit of holding multiple imperfectly correlated instruments.

    Args:
        weights: Instrument weights (must sum to 1.0)
        correlation_matrix: Instrument correlation matrix
        cap: Maximum IDM (Carver recommends 2.5)

    Returns:
        IDM value (>= 1.0, capped at `cap`)
    """
    instruments = list(weights.keys())
    n = len(instruments)

    if n <= 1:
        return 1.0

    w = np.array([weights[inst] for inst in instruments])

    # Extract correlation matrix in same order as weights
    C = correlation_matrix.loc[instruments, instruments].values

    # w' * C * w
    portfolio_variance = w @ C @ w

    if portfolio_variance <= 0:
        logger.warning("Non-positive portfolio variance in IDM calculation, returning 1.0")
        return 1.0

    idm = 1.0 / np.sqrt(portfolio_variance)

    # Cap IDM
    idm = min(idm, cap)

    logger.info(
        f"IDM = {idm:.3f} (portfolio_var = {portfolio_variance:.4f}, "
        f"{n} instruments)"
    )

    return float(idm)


def recommend_weights_and_idm(
    returns_dict: dict[str, pd.Series],
    min_overlap: int = 100,
    idm_cap: float = 2.5,
) -> tuple[dict[str, float], float, pd.DataFrame]:
    """Compute instrument weights, IDM, and correlation matrix.

    Uses equal weights (Carver's recommendation for <5 instruments).

    Args:
        returns_dict: instrument_name -> return series
        min_overlap: Min overlapping periods for correlation
        idm_cap: Maximum IDM

    Returns:
        (weights, idm, correlation_matrix)
    """
    instruments = list(returns_dict.keys())
    weights = equal_weights(instruments)
    corr = compute_instrument_correlation(returns_dict, min_overlap=min_overlap)
    idm = calculate_idm(weights, corr, cap=idm_cap)

    return weights, idm, corr


def _nearest_psd(corr: pd.DataFrame) -> pd.DataFrame:
    """Project a correlation matrix to the nearest positive semi-definite matrix."""
    A = corr.values
    eigenvalues = np.linalg.eigvalsh(A)

    if eigenvalues.min() >= -1e-10:
        return corr  # Already PSD

    # Higham's algorithm (simplified)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-8)
    A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Re-normalize to correlation matrix
    D = np.sqrt(np.diag(A_psd))
    A_corr = A_psd / np.outer(D, D)
    np.fill_diagonal(A_corr, 1.0)

    logger.warning("Projected correlation matrix to nearest PSD")
    return pd.DataFrame(A_corr, index=corr.index, columns=corr.columns)
