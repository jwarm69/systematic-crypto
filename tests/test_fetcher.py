"""Tests for OHLCV fetching/pagination helpers."""

from __future__ import annotations

import sys
import types

from src.data.fetcher import fetch_ohlcv, timeframe_to_milliseconds


def test_timeframe_to_milliseconds():
    assert timeframe_to_milliseconds("1m") == 60_000
    assert timeframe_to_milliseconds("1h") == 3_600_000
    assert timeframe_to_milliseconds("4h") == 14_400_000
    assert timeframe_to_milliseconds("1d") == 86_400_000


def test_fetch_ohlcv_auto_backfills_when_limit_exceeds_exchange_cap(monkeypatch):
    tf_ms = 3_600_000
    n_total = 2_600
    base_ts = 1_700_000_000_000

    candles = []
    for i in range(n_total):
        ts = base_ts + i * tf_ms
        price = 100.0 + i * 0.1
        candles.append([ts, price, price + 1, price - 1, price + 0.25, 1_000 + i])

    class FakeExchange:
        last_instance = None

        def __init__(self):
            self.calls = []
            FakeExchange.last_instance = self

        def load_markets(self):
            return None

        def milliseconds(self):
            return base_ts + (n_total + 2) * tf_ms

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
            self.calls.append({"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit})
            if since is None:
                start = max(0, len(candles) - limit)
            else:
                start = 0
                while start < len(candles) and candles[start][0] < since:
                    start += 1
            return candles[start:start + limit]

    fake_ccxt = types.SimpleNamespace(hyperliquid=FakeExchange)
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)

    df = fetch_ohlcv(
        symbol="BTC/USDC:USDC",
        timeframe="1h",
        limit=2_200,
        exchange_id="hyperliquid",
    )

    assert len(df) == 2_200
    exchange_instance = FakeExchange.last_instance
    assert exchange_instance is not None
    assert len(exchange_instance.calls) >= 3
    assert exchange_instance.calls[0]["since"] is not None


def test_fetch_ohlcv_small_limit_keeps_latest_fetch_mode(monkeypatch):
    tf_ms = 3_600_000
    n_total = 1_500
    base_ts = 1_700_000_000_000

    candles = []
    for i in range(n_total):
        ts = base_ts + i * tf_ms
        price = 100.0 + i * 0.1
        candles.append([ts, price, price + 1, price - 1, price + 0.25, 1_000 + i])

    class FakeExchange:
        last_instance = None

        def __init__(self):
            self.calls = []
            FakeExchange.last_instance = self

        def load_markets(self):
            return None

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
            self.calls.append({"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit})
            start = max(0, len(candles) - limit)
            return candles[start:start + limit]

    fake_ccxt = types.SimpleNamespace(hyperliquid=FakeExchange)
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)

    df = fetch_ohlcv(
        symbol="ETH/USDC:USDC",
        timeframe="1h",
        limit=300,
        exchange_id="hyperliquid",
    )

    assert len(df) == 300
    exchange_instance = FakeExchange.last_instance
    assert exchange_instance is not None
    assert len(exchange_instance.calls) == 1
    assert exchange_instance.calls[0]["since"] is None
