"""Hyperliquid exchange adapter using CCXT.

Adapted from astro-trader/src/exchange/hyperliquid.py with additional
methods for funding rates and OHLCV fetching.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from .exchange_base import ExchangeAdapter, Order, OrderResult, Position

logger = logging.getLogger(__name__)


class HyperliquidAdapter(ExchangeAdapter):
    """Hyperliquid perpetual futures adapter via CCXT.

    Supports both live trading and paper trading modes.
    Paper mode simulates execution with configurable slippage.
    """

    def __init__(self, config: dict, paper_mode: bool = True):
        super().__init__(config, paper_mode)
        self.cca = None  # CCXT async client

        # Paper trading state
        self._paper_balance = config.get("paper_balance", 10000.0)
        self._paper_positions: dict[str, Position] = {}
        self._paper_prices: dict[str, float] = {}
        self._paper_funding_rates: dict[str, float] = {}

    async def connect(self) -> None:
        """Connect to Hyperliquid (or initialize paper mode)."""
        if self.paper_mode:
            logger.info("Hyperliquid adapter connected in PAPER mode")
            self._connected = True
            return

        try:
            import ccxt.async_support as ccxt_async

            params = {}
            if self.config.get("wallet_address"):
                params["walletAddress"] = self.config["wallet_address"]
            if self.config.get("private_key"):
                params["privateKey"] = self.config["private_key"]

            self.cca = ccxt_async.hyperliquid(params)

            if self.config.get("testnet", False):
                self.cca.set_sandbox_mode(True)

            await self.cca.load_markets()
            self._connected = True
            logger.info("Hyperliquid adapter connected (LIVE)")

        except ImportError:
            raise ImportError("ccxt required: pip install ccxt")

    async def disconnect(self) -> None:
        if self.cca:
            await self.cca.close()
        self._connected = False

    async def get_balance(self) -> float:
        if self.paper_mode:
            return self._paper_balance
        info = await self.cca.fetch_balance()
        return float(info["info"]["marginSummary"]["accountValue"])

    async def get_positions(self) -> list[Position]:
        if self.paper_mode:
            return list(self._paper_positions.values())

        info = await self.cca.fetch_balance()
        positions = []
        for asset_pos in info["info"].get("assetPositions", []):
            pos = asset_pos.get("position", {})
            size = float(pos.get("szi", 0))
            if size != 0:
                coin = pos.get("coin", "")
                symbol = f"{coin}/USDC:USDC"
                positions.append(Position(
                    symbol=symbol,
                    side="long" if size > 0 else "short",
                    size=abs(size),
                    entry_price=float(pos.get("entryPx", 0)),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                    liquidation_price=float(pos.get("liquidationPx", 0)) if pos.get("liquidationPx") else None,
                    leverage=float(pos.get("leverage", {}).get("value", 1)),
                ))
        return positions

    async def get_position(self, symbol: str) -> Position | None:
        if self.paper_mode:
            return self._paper_positions.get(symbol)
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def place_order(self, order: Order) -> OrderResult:
        if self.paper_mode:
            return self._paper_order(order)

        try:
            params = {"reduceOnly": order.reduce_only, "timeInForce": "Gtc"}
            if order.client_order_id:
                params["clientOrderId"] = order.client_order_id

            response = await self.cca.create_order(
                symbol=order.symbol,
                type=order.order_type,
                side=order.side,
                amount=order.size,
                price=order.price,
                params=params,
            )
            return OrderResult(
                success=True,
                order_id=response.get("id"),
                filled_size=float(response.get("filled", order.size)),
                filled_price=float(response.get("average", 0)),
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return OrderResult(success=False, error_message=str(e), timestamp=datetime.now())

    def _paper_order(self, order: Order) -> OrderResult:
        """Simulate order in paper mode."""
        price = self._paper_prices.get(order.symbol, 0)
        if price == 0:
            return OrderResult(success=False, error_message=f"No price for {order.symbol}")

        slippage_bps = self.config.get("slippage_bps", 5)
        slippage = price * (slippage_bps / 10000)
        fill_price = price + slippage if order.side == "buy" else price - slippage

        current_pos = self._paper_positions.get(order.symbol)

        if order.reduce_only and current_pos:
            pnl = self._calc_pnl(current_pos, fill_price)
            self._paper_balance += pnl
            del self._paper_positions[order.symbol]
        elif not order.reduce_only:
            side = "long" if order.side == "buy" else "short"

            if current_pos and current_pos.side == side:
                # Add to position
                total_cost = (current_pos.size * current_pos.entry_price) + (order.size * fill_price)
                total_size = current_pos.size + order.size
                self._paper_positions[order.symbol] = Position(
                    symbol=order.symbol, side=side,
                    size=total_size, entry_price=total_cost / total_size,
                )
            elif current_pos and current_pos.side != side:
                # Close and flip
                pnl = self._calc_pnl(current_pos, fill_price)
                self._paper_balance += pnl
                if order.size > current_pos.size:
                    remaining = order.size - current_pos.size
                    self._paper_positions[order.symbol] = Position(
                        symbol=order.symbol, side=side,
                        size=remaining, entry_price=fill_price,
                    )
                else:
                    self._paper_positions.pop(order.symbol, None)
            else:
                self._paper_positions[order.symbol] = Position(
                    symbol=order.symbol, side=side,
                    size=order.size, entry_price=fill_price,
                )

        return OrderResult(
            success=True,
            order_id=f"paper_{uuid.uuid4().hex[:8]}",
            filled_size=order.size,
            filled_price=fill_price,
            timestamp=datetime.now(),
        )

    def _calc_pnl(self, pos: Position, exit_price: float) -> float:
        if pos.side == "long":
            return (exit_price - pos.entry_price) * pos.size
        return (pos.entry_price - exit_price) * pos.size

    async def close_position(self, symbol: str) -> OrderResult:
        pos = await self.get_position(symbol)
        if not pos:
            return OrderResult(success=True, filled_size=0, error_message="No position")
        side = "sell" if pos.side == "long" else "buy"
        order = Order(symbol=symbol, side=side, size=pos.size, reduce_only=True)
        return await self.place_order(order)

    async def get_current_price(self, symbol: str) -> float:
        if self.paper_mode:
            return self._paper_prices.get(symbol, 0)
        ticker = await self.cca.fetch_ticker(symbol)
        price = float(ticker.get("last", 0))
        return price

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        if self.paper_mode:
            return 0
        response = await self.cca.cancel_all_orders(symbol)
        return len(response) if isinstance(response, list) else 1

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> list:
        """Fetch OHLCV bars from Hyperliquid."""
        if self.paper_mode:
            return []
        return await self.cca.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    async def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate for a symbol."""
        if self.paper_mode:
            return self._paper_funding_rates.get(symbol, 0.0001)
        try:
            funding = await self.cca.fetch_funding_rate(symbol)
            return float(funding.get("fundingRate", 0))
        except Exception as e:
            logger.warning(f"Could not fetch funding rate for {symbol}: {e}")
            return 0.0

    def update_paper_price(self, symbol: str, price: float) -> None:
        """Update simulated price for paper trading."""
        self._paper_prices[symbol] = price

    def update_paper_funding(self, symbol: str, rate: float) -> None:
        """Update simulated funding rate for paper trading."""
        self._paper_funding_rates[symbol] = rate
