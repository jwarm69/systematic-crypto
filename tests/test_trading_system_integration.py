"""Integration-style tests for TradingSystem orchestration."""

from datetime import timedelta

import pandas as pd
import pytest
import yaml

from src.execution.exchange_base import OrderResult, Position
from src.system.trading_system import TradingSystem


class DummyRule:
    """Deterministic forecast rule for integration tests."""

    def forecast(self, prices: pd.Series, **kwargs) -> pd.Series:
        return pd.Series(10.0, index=prices.index)


class FakeExchange:
    """Simple in-memory exchange used by TradingSystem tests."""

    def __init__(self, balance: float = 10000.0, partial_fill: float | None = None):
        self.paper_mode = True
        self._balance = balance
        self._positions: dict[str, float] = {}
        self._partial_fill = partial_fill

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def get_balance(self) -> float:
        return self._balance

    async def get_positions(self) -> list[Position]:
        positions: list[Position] = []
        for symbol, signed_size in self._positions.items():
            if signed_size == 0:
                continue
            positions.append(
                Position(
                    symbol=symbol,
                    side="long" if signed_size > 0 else "short",
                    size=abs(signed_size),
                    entry_price=100.0,
                )
            )
        return positions

    async def get_position(self, symbol: str) -> Position | None:
        signed_size = self._positions.get(symbol, 0.0)
        if signed_size == 0:
            return None
        return Position(
            symbol=symbol,
            side="long" if signed_size > 0 else "short",
            size=abs(signed_size),
            entry_price=100.0,
        )

    async def place_order(self, order) -> OrderResult:
        fill_size = self._partial_fill if self._partial_fill is not None else order.size
        signed_fill = fill_size if order.side == "buy" else -fill_size
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + signed_fill
        if abs(self._positions[order.symbol]) < 1e-12:
            self._positions.pop(order.symbol, None)
        return OrderResult(
            success=True,
            order_id="test_order",
            filled_size=fill_size,
            filled_price=100.0,
        )

    def update_paper_price(self, symbol: str, price: float) -> None:
        return None


def _write_test_configs(base_dir, max_leverage: float = 2.0) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    with open(base_dir / "portfolio.yaml", "w") as f:
        yaml.safe_dump(
            {
                "account": {"capital": 5000, "vol_target": 0.12, "max_leverage": max_leverage},
                "risk": {
                    "max_drawdown": 0.50,
                    "daily_loss_limit": 0.03,
                    "kill_switch_ratio": 1.3,
                },
                "position": {"buffer_fraction": 0.10},
                "instrument_weights": {"BTC": 1.0},
                "forecast_weights": {"dummy": 1.0},
                "idm": 1.0,
                "fdm": 1.0,
            },
            f,
            sort_keys=False,
        )

    with open(base_dir / "instruments.yaml", "w") as f:
        yaml.safe_dump(
            {
                "instruments": {
                    "BTC": {
                        "symbol": "BTC/USDC:USDC",
                        "min_order_size": 0.001,
                    }
                }
            },
            f,
            sort_keys=False,
        )

    with open(base_dir / "rules.yaml", "w") as f:
        yaml.safe_dump(
            {
                "rules": {
                    "dummy": {
                        "type": "ewmac",
                        "fast_span": 16,
                        "slow_span": 64,
                        "enabled": False,
                    }
                }
            },
            f,
            sort_keys=False,
        )

    with open(base_dir / "paper.yaml", "w") as f:
        yaml.safe_dump(
            {
                "exchange": {"name": "fake"},
                "trading": {"mode": "paper", "instruments": ["BTC"], "timeframe": "1h"},
                "schedule": {"data_lookback_bars": 200},
                "logging": {"state_file": str(base_dir / "state.json")},
            },
            f,
            sort_keys=False,
        )


def _price_frame(n: int = 200, close: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )


def _build_system(tmp_path, max_leverage: float = 2.0) -> TradingSystem:
    config_dir = tmp_path / "configs"
    _write_test_configs(config_dir, max_leverage=max_leverage)
    system = TradingSystem(config_path=config_dir)
    system.rules = {"dummy": DummyRule()}
    return system


@pytest.mark.asyncio
async def test_run_once_resets_daily_baseline(monkeypatch, tmp_path):
    system = _build_system(tmp_path)
    fake_exchange = FakeExchange(balance=10000.0)
    system.exchange = fake_exchange

    monkeypatch.setattr("src.system.trading_system.fetch_and_cache", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr("src.system.trading_system.current_volatility", lambda *args, **kwargs: 0.5)

    await system.run_once()
    first_reset = system.risk_manager._last_daily_reset

    assert system.risk_manager.daily_start_equity == 10000.0
    assert first_reset is not None

    fake_exchange._balance = 9500.0
    await system.run_once()

    assert system.risk_manager.daily_start_equity == 10000.0
    assert system.risk_manager._last_daily_reset == first_reset

    system.risk_manager._last_daily_reset = first_reset - timedelta(days=1)
    fake_exchange._balance = 9000.0
    await system.run_once()
    assert system.risk_manager.daily_start_equity == 9000.0


@pytest.mark.asyncio
async def test_run_once_flattens_when_leverage_breached(monkeypatch, tmp_path):
    system = _build_system(tmp_path, max_leverage=1.0)
    fake_exchange = FakeExchange(balance=1000.0)
    system.exchange = fake_exchange

    monkeypatch.setattr("src.system.trading_system.fetch_and_cache", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr("src.system.trading_system.current_volatility", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.system.trading_system.carver_position_size", lambda *args, **kwargs: 20.0)

    results = await system.run_once()
    btc = results["BTC"]

    assert btc["buffered_position"] == 0.0
    assert any("Leverage" in reason for reason in btc["risk_reasons"])


@pytest.mark.asyncio
async def test_run_once_reconciles_position_after_partial_fill(monkeypatch, tmp_path):
    system = _build_system(tmp_path, max_leverage=10.0)
    fake_exchange = FakeExchange(balance=10000.0, partial_fill=0.4)
    system.exchange = fake_exchange

    monkeypatch.setattr("src.system.trading_system.fetch_and_cache", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr("src.system.trading_system.current_volatility", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("src.system.trading_system.carver_position_size", lambda *args, **kwargs: 5.0)

    results = await system.run_once()
    symbol = "BTC/USDC:USDC"

    assert results["BTC"]["trade_result"] is not None
    assert results["BTC"]["trade_result"].filled_size == 0.4
    assert system.current_positions[symbol] == pytest.approx(0.4)
    assert system.current_positions[symbol] != pytest.approx(results["BTC"]["buffered_position"])
