"""Main trading system loop.

Orchestrates: data fetch -> forecast -> position sizing -> buffering -> risk check -> execute.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data.fetcher import fetch_and_cache
from ..execution.exchange_base import Order
from ..execution.hyperliquid import HyperliquidAdapter
from ..indicators.volatility import current_volatility
from ..portfolio.buffering import apply_buffer
from ..portfolio.position_sizing import carver_position_size
from ..risk.risk_manager import RiskManager
from ..rules.ewmac import EWMACRule
from ..rules.scaling import combine_forecasts

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


class TradingSystem:
    """Main trading system orchestrator.

    Loads config, runs the forecast->size->execute pipeline,
    and manages state persistence.
    """

    def __init__(self, config_path: str | Path | None = None):
        # Load configs
        config_path = Path(config_path) if config_path else CONFIG_DIR
        self.portfolio_cfg = self._load_yaml(config_path / "portfolio.yaml")
        self.instruments_cfg = self._load_yaml(config_path / "instruments.yaml")
        self.rules_cfg = self._load_yaml(config_path / "rules.yaml")
        self.paper_cfg = self._load_yaml(config_path / "paper.yaml")

        # Core parameters
        self.capital = self.portfolio_cfg["account"]["capital"]
        self.vol_target = self.portfolio_cfg["account"]["vol_target"]
        self.max_leverage = self.portfolio_cfg["account"]["max_leverage"]
        self.buffer_fraction = self.portfolio_cfg["position"]["buffer_fraction"]
        self.forecast_weights = self.portfolio_cfg["forecast_weights"]
        self.fdm = self.portfolio_cfg["fdm"]
        self.idm = self.portfolio_cfg["idm"]

        # Initialize trading rules
        self.rules = self._init_rules()

        # Initialize risk manager
        risk_cfg = self.portfolio_cfg["risk"]
        self.risk_manager = RiskManager(
            max_drawdown=risk_cfg["max_drawdown"],
            daily_loss_limit=risk_cfg["daily_loss_limit"],
            kill_switch_ratio=risk_cfg.get("kill_switch_ratio", 1.3),
            max_leverage=self.max_leverage,
        )

        # Exchange adapter
        exchange_cfg = self.paper_cfg.get("exchange", {})
        self.exchange = HyperliquidAdapter(
            config=exchange_cfg,
            paper_mode=self.paper_cfg.get("trading", {}).get("mode", "paper") == "paper",
        )

        # State
        self.current_positions: dict[str, float] = {}
        self.state_file = Path(self.paper_cfg.get("logging", {}).get("state_file", "data/system_state.json"))

    def _load_yaml(self, path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _init_rules(self) -> dict[str, EWMACRule]:
        """Initialize enabled trading rules from config."""
        rules = {}
        for rule_name, cfg in self.rules_cfg["rules"].items():
            if not cfg.get("enabled", False):
                continue
            if cfg["type"] == "ewmac":
                rules[rule_name] = EWMACRule(
                    fast_span=cfg["fast_span"],
                    slow_span=cfg["slow_span"],
                )
        return rules

    async def run_once(self) -> dict:
        """Run one iteration of the trading loop.

        Returns:
            Dict with forecasts, positions, trades executed
        """
        results = {}
        await self.exchange.connect()

        try:
            # Get active instruments
            instruments = self.paper_cfg.get("trading", {}).get("instruments", ["BTC"])
            timeframe = self.paper_cfg.get("trading", {}).get("timeframe", "1h")

            for instrument in instruments:
                inst_cfg = self.instruments_cfg["instruments"].get(instrument, {})
                symbol = inst_cfg.get("symbol", f"{instrument}/USDC:USDC")

                # 1. Fetch data
                logger.info(f"Fetching data for {instrument}...")
                df = fetch_and_cache(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=self.paper_cfg.get("schedule", {}).get("data_lookback_bars", 5000),
                )

                if df.empty or len(df) < 100:
                    logger.warning(f"Insufficient data for {instrument}, skipping")
                    continue

                prices = df["close"]

                # 2. Calculate forecasts from each rule
                rule_forecasts = {}
                for rule_name, rule in self.rules.items():
                    fc = rule.forecast(prices)
                    rule_forecasts[rule_name] = fc
                    logger.info(
                        f"  {rule_name}: forecast={fc.iloc[-1]:.1f} "
                        f"(avg_abs={fc.abs().mean():.1f})"
                    )

                if not rule_forecasts:
                    logger.warning(f"No forecasts for {instrument}")
                    continue

                # 3. Combine forecasts
                combined = combine_forecasts(
                    rule_forecasts,
                    weights=self.forecast_weights,
                    fdm=self.fdm,
                )
                current_forecast = combined.iloc[-1]

                # 4. Calculate volatility
                vol = current_volatility(prices)

                # 5. Calculate target position
                inst_weight = self.portfolio_cfg.get("instrument_weights", {}).get(instrument, 1.0)
                target_pos = carver_position_size(
                    capital=self.capital,
                    price=prices.iloc[-1],
                    instrument_vol=vol,
                    forecast=current_forecast,
                    vol_target=self.vol_target,
                    idm=self.idm,
                    instrument_weight=inst_weight,
                    max_leverage=self.max_leverage,
                )

                # 6. Apply buffer
                current_pos = self.current_positions.get(symbol, 0.0)
                buffered_pos = apply_buffer(
                    current_position=current_pos,
                    target_position=target_pos,
                    capital=self.capital,
                    price=prices.iloc[-1],
                    instrument_vol=vol,
                    vol_target=self.vol_target,
                    buffer_fraction=self.buffer_fraction,
                )

                # 7. Risk check
                equity = await self.exchange.get_balance()
                self.risk_manager.update_equity(equity)
                should_reduce, risk_reasons = self.risk_manager.check_all(equity)

                if should_reduce:
                    logger.warning(f"Risk triggered: {risk_reasons}")
                    buffered_pos = 0.0  # Flatten

                # 8. Execute trade if needed
                trade_size = buffered_pos - current_pos
                trade_result = None

                min_size = inst_cfg.get("min_order_size", 0.001)
                if abs(trade_size) >= min_size:
                    side = "buy" if trade_size > 0 else "sell"
                    # Update paper price before executing
                    if self.exchange.paper_mode:
                        self.exchange.update_paper_price(symbol, prices.iloc[-1])

                    order = Order(
                        symbol=symbol,
                        side=side,
                        size=abs(trade_size),
                    )
                    trade_result = await self.exchange.place_order(order)
                    if trade_result.success:
                        self.current_positions[symbol] = buffered_pos
                        logger.info(
                            f"TRADE: {side} {abs(trade_size):.6f} {instrument} "
                            f"@ {trade_result.filled_price:.2f}"
                        )
                    else:
                        logger.error(f"Trade failed: {trade_result.error_message}")
                else:
                    logger.info(f"No trade needed for {instrument} (change {trade_size:.6f} < min {min_size})")
                    self.current_positions[symbol] = current_pos

                results[instrument] = {
                    "forecast": current_forecast,
                    "volatility": vol,
                    "target_position": target_pos,
                    "buffered_position": buffered_pos,
                    "current_position": current_pos,
                    "trade_size": trade_size if abs(trade_size) >= min_size else 0,
                    "trade_result": trade_result,
                    "price": prices.iloc[-1],
                    "risk_reasons": risk_reasons if should_reduce else [],
                }

            # Save state
            self._save_state()

        finally:
            await self.exchange.disconnect()

        return results

    def _save_state(self) -> None:
        """Persist system state to JSON."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": self.current_positions,
            "capital": self.capital,
            "kill_switch": self.risk_manager.kill_switch_active,
            "peak_equity": self.risk_manager.peak_equity,
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self) -> None:
        """Load persisted state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            self.current_positions = state.get("positions", {})
            self.risk_manager.peak_equity = state.get("peak_equity", 0)
            self.risk_manager.kill_switch_active = state.get("kill_switch", False)
            logger.info(f"Loaded state: {len(self.current_positions)} positions")

    def run(self) -> dict:
        """Synchronous entry point."""
        return asyncio.run(self.run_once())
