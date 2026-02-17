"""OHLCV data fetching via CCXT with parquet caching and gap-fill cleaning."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert a candle timeframe string to milliseconds."""
    mapping = {
        "1m": 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }
    if timeframe in mapping:
        return mapping[timeframe]

    if len(timeframe) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"Invalid timeframe: {timeframe}") from exc

    unit_ms = {"m": 60_000, "h": 60 * 60_000, "d": 24 * 60 * 60_000}
    if unit not in unit_ms:
        raise ValueError(f"Unsupported timeframe unit '{unit}' in {timeframe}")
    return value * unit_ms[unit]


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 5000,
    exchange_id: str = "hyperliquid",
    since: datetime | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data via CCXT synchronously.

    Args:
        symbol: CCXT unified symbol (e.g. "BTC/USDC:USDC")
        timeframe: Candle timeframe ("1h", "4h", "1d")
        limit: Max bars to fetch
        exchange_id: CCXT exchange id
        since: Fetch bars since this datetime (UTC)

    Returns:
        DataFrame with columns: open, high, low, close, volume, indexed by datetime
    """
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    exchange.load_markets()

    since_ms = int(since.timestamp() * 1000) if since else None

    all_bars = []
    fetched = 0
    max_batch_size = 1000  # Most exchanges cap at 1000 per request

    # If since isn't given and user asks for deep history, estimate a historical
    # starting point so pagination can walk forward and actually collect >1k bars.
    if since_ms is None and limit > max_batch_size:
        tf_ms = timeframe_to_milliseconds(timeframe)
        now_ms = (
            int(exchange.milliseconds())
            if hasattr(exchange, "milliseconds")
            else int(datetime.now(timezone.utc).timestamp() * 1000)
        )
        lookback_bars = limit + max_batch_size
        since_ms = now_ms - (lookback_bars * tf_ms)
        logger.info(
            "Estimated since for deep fetch: %s (%d bars lookback)",
            pd.to_datetime(since_ms, unit="ms", utc=True),
            lookback_bars,
        )

    while fetched < limit:
        batch_size = min(max_batch_size, limit - fetched)
        bars = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=batch_size
        )
        if not bars:
            break

        all_bars.extend(bars)
        fetched += len(bars)

        # Without an explicit since, exchanges usually return only "latest".
        # In that mode there is no reliable backward pagination.
        if since_ms is None:
            break

        if len(bars) < batch_size:
            break  # No more data

        # Move since forward for pagination
        next_since_ms = bars[-1][0] + 1
        if next_since_ms <= since_ms:
            break
        since_ms = next_since_ms

    if not all_bars:
        logger.warning(f"No data returned for {symbol} {timeframe}")
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    if len(df) > limit:
        df = df.tail(limit)

    logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe}")
    return df


def clean_ohlcv(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """Clean OHLCV data: remove duplicates, fill gaps, handle outliers.

    Args:
        df: Raw OHLCV DataFrame
        timeframe: Expected candle frequency for gap detection

    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df

    # Remove exact duplicates
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Reindex to fill gaps with forward-fill (max 5 bars)
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    freq = freq_map.get(timeframe, "1h")
    full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq, tz="UTC")
    n_gaps = len(full_index) - len(df)
    if n_gaps > 0:
        logger.info(f"Filling {n_gaps} gaps in {timeframe} data")
        df = df.reindex(full_index)
        # Forward-fill OHLC (use last close), zero-fill volume
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill(limit=5)
        df["volume"] = df["volume"].fillna(0)

    # Drop any remaining NaN rows (start of series)
    df = df.dropna(subset=["close"])

    # Remove zero/negative prices
    df = df[df["close"] > 0]

    # Flag extreme returns (>50% in one bar) but don't remove
    returns = df["close"].pct_change()
    extreme = returns.abs() > 0.5
    if extreme.any():
        logger.warning(f"Found {extreme.sum()} bars with >50% returns - possible data issues")

    return df


def save_parquet(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    """Save DataFrame to parquet cache.

    Args:
        df: OHLCV DataFrame
        symbol: Instrument symbol (used in filename)
        timeframe: Timeframe (used in filename)

    Returns:
        Path to saved file
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Clean symbol for filename: BTC/USDC:USDC -> BTC_USDC
    clean_name = symbol.replace("/", "_").replace(":", "_")
    path = DATA_DIR / f"{clean_name}_{timeframe}.parquet"
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} bars to {path}")
    return path


def load_parquet(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load DataFrame from parquet cache.

    Args:
        symbol: Instrument symbol
        timeframe: Timeframe

    Returns:
        DataFrame if cache exists, None otherwise
    """
    clean_name = symbol.replace("/", "_").replace(":", "_")
    path = DATA_DIR / f"{clean_name}_{timeframe}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} bars from cache: {path}")
        return df
    return None


def fetch_and_cache(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 5000,
    exchange_id: str = "hyperliquid",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch data with caching: load from parquet if available, else fetch and save.

    Args:
        symbol: CCXT unified symbol
        timeframe: Candle timeframe
        limit: Max bars
        exchange_id: Exchange
        force_refresh: If True, always fetch fresh data

    Returns:
        Cleaned OHLCV DataFrame
    """
    cached = None
    if not force_refresh:
        cached = load_parquet(symbol, timeframe)
        if cached is not None and len(cached) > 0:
            # Check if cache is stale (last bar > 2x timeframe ago)
            last_bar = cached.index[-1]
            now = pd.Timestamp.now(tz="UTC")
            tf_hours = {"1m": 1 / 60, "5m": 5 / 60, "1h": 1, "4h": 4, "1d": 24}
            max_age_hours = tf_hours.get(timeframe, 1) * 2

            if (now - last_bar).total_seconds() / 3600 < max_age_hours:
                return cached

            # Fetch only new data since last cached bar
            logger.info(f"Cache stale, fetching new data since {last_bar}")
            try:
                new_data = fetch_ohlcv(
                    symbol,
                    timeframe,
                    limit=500,
                    exchange_id=exchange_id,
                    since=last_bar.to_pydatetime(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed incremental refresh for %s %s: %s", symbol, timeframe, exc)
                return cached
            if not new_data.empty:
                combined = pd.concat([cached, new_data])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                combined = clean_ohlcv(combined, timeframe)
                save_parquet(combined, symbol, timeframe)
                return combined
            return cached

    # Full fetch
    try:
        df = fetch_ohlcv(symbol, timeframe, limit, exchange_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed full fetch for %s %s: %s", symbol, timeframe, exc)
        if cached is not None and len(cached) > 0:
            return cached
        return pd.DataFrame()
    df = clean_ohlcv(df, timeframe)
    if not df.empty:
        save_parquet(df, symbol, timeframe)
    return df
