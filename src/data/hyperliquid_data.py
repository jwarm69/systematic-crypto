"""Multi-instrument data fetching for Hyperliquid.

Fetches OHLCV, funding rates, open interest, and volume
for multiple instruments. Provides cross-asset feature computation.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .fetcher import fetch_and_cache

logger = logging.getLogger(__name__)


def fetch_multi_instrument(
    instruments: dict[str, dict],
    timeframe: str = "1h",
    limit: int = 5000,
    exchange_id: str = "hyperliquid",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple instruments.

    Args:
        instruments: instrument_name -> config dict (must have 'symbol' key)
        timeframe: Candle timeframe
        limit: Max bars per instrument
        exchange_id: Exchange
        force_refresh: Force re-fetch

    Returns:
        instrument_name -> OHLCV DataFrame
    """
    data = {}
    for name, cfg in instruments.items():
        symbol = cfg.get("symbol", f"{name}/USDC:USDC")
        logger.info(f"Fetching {name} ({symbol})...")
        try:
            df = fetch_and_cache(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                exchange_id=exchange_id,
                force_refresh=force_refresh,
            )
            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {name}: {len(df)} bars")
                continue
            data[name] = df
            logger.info(f"  {name}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
        except Exception as exc:
            logger.error(f"Failed to fetch {name}: {exc}")
    return data


def compute_returns(
    data: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """Compute price returns for each instrument.

    Args:
        data: instrument_name -> OHLCV DataFrame

    Returns:
        instrument_name -> returns Series
    """
    returns = {}
    for name, df in data.items():
        ret = df["close"].pct_change().dropna()
        returns[name] = ret
    return returns


def align_price_series(
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align close prices across instruments into a single DataFrame.

    Args:
        data: instrument_name -> OHLCV DataFrame

    Returns:
        DataFrame with columns = instrument names, index = aligned timestamps
    """
    prices = {}
    for name, df in data.items():
        prices[name] = df["close"]

    aligned = pd.DataFrame(prices)
    aligned = aligned.dropna(how="all")
    return aligned


def compute_cross_asset_features(
    data: dict[str, pd.DataFrame],
    reference: str = "BTC",
    lookbacks: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute cross-asset features for each instrument.

    Features:
    - BTC momentum (reference asset momentum as signal for alts)
    - Relative strength (instrument return vs reference return)
    - Cross-correlation (rolling correlation with reference)

    Args:
        data: instrument_name -> OHLCV DataFrame
        reference: Reference instrument (usually BTC)
        lookbacks: Lookback periods for features

    Returns:
        instrument_name -> DataFrame of cross-asset features
    """
    if lookbacks is None:
        lookbacks = [24, 72, 168]  # 1d, 3d, 1w in hourly bars

    if reference not in data:
        logger.warning(f"Reference instrument {reference} not in data")
        return {}

    ref_prices = data[reference]["close"]
    ref_returns = ref_prices.pct_change()

    features = {}
    for name, df in data.items():
        inst_prices = df["close"]
        inst_returns = inst_prices.pct_change()

        feat_dict = {}
        for lb in lookbacks:
            # Reference (BTC) momentum
            feat_dict[f"ref_momentum_{lb}"] = ref_prices.pct_change(lb)

            # Relative strength: instrument vs reference
            if name != reference:
                inst_mom = inst_prices.pct_change(lb)
                ref_mom = ref_prices.pct_change(lb)
                # Align indices
                common = inst_mom.index.intersection(ref_mom.index)
                if len(common) > 0:
                    feat_dict[f"relative_strength_{lb}"] = (
                        inst_mom.loc[common] - ref_mom.loc[common]
                    )

                # Rolling correlation with reference
                common_ret = pd.DataFrame({
                    "inst": inst_returns,
                    "ref": ref_returns,
                }).dropna()
                if len(common_ret) > lb:
                    feat_dict[f"corr_with_ref_{lb}"] = (
                        common_ret["inst"].rolling(lb).corr(common_ret["ref"])
                    )

        features[name] = pd.DataFrame(feat_dict)

    return features


def liquidity_filter(
    data: dict[str, pd.DataFrame],
    min_daily_volume_usd: float = 1_000_000,
    lookback_days: int = 7,
) -> list[str]:
    """Filter instruments by minimum liquidity.

    Args:
        data: instrument_name -> OHLCV DataFrame
        min_daily_volume_usd: Minimum average daily volume in USD
        lookback_days: Days to average over

    Returns:
        List of instrument names that pass the filter
    """
    passed = []
    lookback_bars = lookback_days * 24  # Assuming hourly bars

    for name, df in data.items():
        recent = df.tail(lookback_bars)
        if len(recent) < 24:
            logger.warning(f"{name}: insufficient data for liquidity filter")
            continue

        # Volume is in base units, multiply by price for USD volume
        daily_volume_usd = (recent["volume"] * recent["close"]).sum() / lookback_days

        if daily_volume_usd >= min_daily_volume_usd:
            passed.append(name)
            logger.info(f"  {name}: avg daily volume ${daily_volume_usd:,.0f} - PASS")
        else:
            logger.info(f"  {name}: avg daily volume ${daily_volume_usd:,.0f} - FAIL (min ${min_daily_volume_usd:,.0f})")

    return passed
