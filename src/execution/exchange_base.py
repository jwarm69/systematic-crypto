"""Abstract base class for exchange adapters.

Adapted from astro-trader/src/exchange/base.py with astro-trader deps removed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Order:
    """Order to place on exchange."""
    symbol: str             # e.g. "BTC/USDC:USDC"
    side: str               # "buy" or "sell"
    size: float             # In instrument units (e.g. BTC)
    order_type: str = "market"  # "market" or "limit"
    price: float | None = None  # Required for limit orders
    reduce_only: bool = False
    client_order_id: str | None = None


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: str | None = None
    filled_size: float = 0.0
    filled_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: str | None = None


@dataclass
class Position:
    """Current position on exchange."""
    symbol: str
    side: str               # "long" or "short"
    size: float             # Absolute size
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    liquidation_price: float | None = None
    leverage: float = 1.0


class ExchangeAdapter(ABC):
    """Abstract exchange adapter providing unified interface."""

    def __init__(self, config: dict, paper_mode: bool = True):
        self.config = config
        self.paper_mode = paper_mode
        self._connected = False

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Adapter", "")

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def get_balance(self) -> float: ...

    @abstractmethod
    async def get_positions(self) -> list[Position]: ...

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None: ...

    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult: ...

    @abstractmethod
    async def close_position(self, symbol: str) -> OrderResult: ...

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float: ...

    @abstractmethod
    async def cancel_all_orders(self, symbol: str | None = None) -> int: ...

    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 500
    ) -> list: ...

    @abstractmethod
    async def fetch_funding_rate(self, symbol: str) -> float: ...

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
