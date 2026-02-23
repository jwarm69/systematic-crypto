"""Main trading system loop.

Orchestrates: data fetch -> forecast -> position sizing -> buffering -> risk check -> execute.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..data.fetcher import fetch_and_cache
from ..execution.exchange_base import Order
from ..execution.hyperliquid import HyperliquidAdapter
from ..indicators.volatility import current_volatility
from ..portfolio.buffering import apply_buffer
from ..portfolio.position_sizing import carver_position_size
from ..risk.risk_manager import RiskManager
from ..rules.base import AbstractTradingRule
from ..rules.breakout import BreakoutRule
from ..rules.carry import CarryRule
from ..rules.ewmac import EWMACRule
from ..rules.mean_reversion import MeanReversionRule
from ..rules.momentum import MomentumRule
from ..rules.scaling import combine_forecasts
from .trade_logger import TradeLogger

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


class TradingSystem:
    """Main trading system orchestrator.

    Loads config, runs the forecast->size->execute pipeline,
    and manages state persistence with trade logging.
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

        # Trade logger
        log_cfg = self.paper_cfg.get("logging", {})
        self.trade_logger = TradeLogger(
            trade_log_path=log_cfg.get("trade_log", "logs/trades.csv"),
            pnl_history_path=log_cfg.get("pnl_history", "data/pnl_history.json"),
            run_log_path=log_cfg.get("run_log", "data/run_history.json"),
        )

        # State
        self.current_positions: dict[str, float] = {}
        self.state_file = Path(log_cfg.get("state_file", "data/system_state.json"))
        self.last_forecasts: dict[str, dict[str, float]] = {}  # instrument -> {rule: value}
        self.last_run_ts: datetime | None = None

    def _load_yaml(self, path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _symbol_for_instrument(self, instrument: str) -> str:
        inst_cfg = self.instruments_cfg["instruments"].get(instrument, {})
        return inst_cfg.get("symbol", f"{instrument}/USDC:USDC")

    @staticmethod
    def _signed_position_size(position) -> float:
        if position is None or position.size == 0:
            return 0.0
        return float(position.size if position.side == "long" else -position.size)

    def _init_rules(self) -> dict[str, AbstractTradingRule]:
        """Initialize enabled trading rules from config."""
        rules = {}
        for rule_name, cfg in self.rules_cfg["rules"].items():
            if not cfg.get("enabled", False):
                continue
            rule_type = cfg["type"]
            if rule_type == "ewmac":
                rules[rule_name] = EWMACRule(
                    fast_span=cfg["fast_span"],
                    slow_span=cfg["slow_span"],
                )
            elif rule_type == "carry":
                rules[rule_name] = CarryRule(smooth_days=cfg.get("smooth_days", 90))
            elif rule_type == "breakout":
                rules[rule_name] = BreakoutRule(lookback=cfg["lookback"])
            elif rule_type == "momentum":
                rules[rule_name] = MomentumRule(lookback=cfg["lookback"])
            elif rule_type == "mean_reversion":
                rules[rule_name] = MeanReversionRule(
                    lookback=cfg.get("lookback", 20),
                    z_cap=cfg.get("z_cap", 3.0),
                )
            elif rule_type == "ml_forecast":
                from ..rules.ml_forecast import MLForecastRule
                rules[rule_name] = MLForecastRule(
                    n_cv_splits=cfg.get("n_cv_splits", 4),
                    n_optuna_trials=cfg.get("n_optuna_trials", 30),
                    feature_selection_k=cfg.get("feature_selection_k", 25),
                )
        return rules

    async def _refresh_positions_from_exchange(self, instruments: list[str]) -> None:
        """Refresh local position cache from exchange truth for tracked instruments."""
        try:
            open_positions = await self.exchange.get_positions()
        except Exception as exc:
            logger.warning(f"Could not refresh positions from exchange: {exc}")
            return

        symbol_to_pos = {p.symbol: self._signed_position_size(p) for p in open_positions}
        for instrument in instruments:
            symbol = self._symbol_for_instrument(instrument)
            self.current_positions[symbol] = symbol_to_pos.get(symbol, 0.0)

    async def _sync_position_from_exchange(self, symbol: str, fallback: float) -> float:
        """Read back a symbol position from exchange after order placement."""
        try:
            position = await self.exchange.get_position(symbol)
        except Exception as exc:
            logger.warning(f"Could not sync position for {symbol}: {exc}")
            return fallback
        return self._signed_position_size(position)

    async def run_once(self) -> dict:
        """Run one iteration of the trading loop.

        Returns:
            Dict with forecasts, positions, trades executed
        """
        results = {}
        run_start = datetime.now(timezone.utc)
        await self.exchange.connect()

        try:
            # Establish run-level equity and daily baseline before placing any orders.
            run_ts = datetime.now(timezone.utc)
            run_equity = await self.exchange.get_balance()
            if self.risk_manager.maybe_reset_daily(run_equity, now=run_ts):
                logger.info(f"Daily risk baseline reset at equity ${run_equity:,.2f}")

                # Log daily PnL on reset
                if self.last_run_ts is not None:
                    prev_equity = self.risk_manager.daily_start_equity
                    self.trade_logger.log_daily_pnl(
                        date=run_ts.strftime("%Y-%m-%d"),
                        equity=run_equity,
                        pnl=run_equity - prev_equity if prev_equity > 0 else 0,
                        positions=dict(self.current_positions),
                        prices={},
                    )

            # Get active instruments
            instruments = self.paper_cfg.get("trading", {}).get("instruments", ["BTC"])
            timeframe = self.paper_cfg.get("trading", {}).get("timeframe", "1h")
            await self._refresh_positions_from_exchange(instruments)
            proposed_positions = dict(self.current_positions)
            latest_prices: dict[str, float] = {}

            for instrument in instruments:
                inst_cfg = self.instruments_cfg["instruments"].get(instrument, {})
                symbol = self._symbol_for_instrument(instrument)

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
                inst_forecast_values = {}
                for rule_name, rule in self.rules.items():
                    # For ML rules, set OHLCV data
                    if hasattr(rule, "set_ohlcv"):
                        rule.set_ohlcv(df)
                    fc = rule.forecast(prices)
                    rule_forecasts[rule_name] = fc
                    inst_forecast_values[rule_name] = float(fc.iloc[-1])
                    logger.info(
                        f"  {rule_name}: forecast={fc.iloc[-1]:.1f} "
                        f"(avg_abs={fc.abs().mean():.1f})"
                    )

                self.last_forecasts[instrument] = inst_forecast_values

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
                latest_price = float(prices.iloc[-1])
                latest_prices[symbol] = latest_price

                # 5. Calculate target position
                inst_weight = self.portfolio_cfg.get("instrument_weights", {}).get(instrument, 1.0)
                target_pos = carver_position_size(
                    capital=self.capital,
                    price=latest_price,
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
                    price=latest_price,
                    instrument_vol=vol,
                    vol_target=self.vol_target,
                    buffer_fraction=self.buffer_fraction,
                )

                # 7. Risk check
                risk_positions = dict(proposed_positions)
                risk_positions[symbol] = buffered_pos

                equity = await self.exchange.get_balance()
                should_reduce, risk_reasons = self.risk_manager.check_all(
                    equity,
                    positions=risk_positions,
                    prices=latest_prices,
                )

                if should_reduce:
                    logger.warning(f"Risk triggered: {risk_reasons}")
                    buffered_pos = 0.0  # Flatten
                    risk_positions[symbol] = 0.0

                # 8. Execute trade if needed
                trade_size = buffered_pos - current_pos
                trade_result = None

                min_size = inst_cfg.get("min_order_size", 0.001)
                if abs(trade_size) >= min_size:
                    side = "buy" if trade_size > 0 else "sell"
                    # Update paper price before executing
                    if self.exchange.paper_mode:
                        self.exchange.update_paper_price(symbol, latest_price)

                    order = Order(
                        symbol=symbol,
                        side=side,
                        size=abs(trade_size),
                    )
                    trade_result = await self.exchange.place_order(order)
                    if trade_result.success:
                        synced_position = await self._sync_position_from_exchange(
                            symbol=symbol,
                            fallback=buffered_pos,
                        )
                        self.current_positions[symbol] = synced_position
                        proposed_positions[symbol] = synced_position
                        logger.info(
                            f"TRADE: {side} {abs(trade_size):.6f} {instrument} "
                            f"@ {trade_result.filled_price:.2f}"
                        )

                        # Log trade
                        self.trade_logger.log_trade(
                            instrument=instrument,
                            symbol=symbol,
                            side=side,
                            size=abs(trade_size),
                            price=latest_price,
                            filled_price=trade_result.filled_price,
                            forecast=current_forecast,
                            volatility=vol,
                            target_pos=target_pos,
                            buffered_pos=buffered_pos,
                            success=True,
                        )
                    else:
                        logger.error(f"Trade failed: {trade_result.error_message}")
                        self.trade_logger.log_trade(
                            instrument=instrument,
                            symbol=symbol,
                            side=side,
                            size=abs(trade_size),
                            price=latest_price,
                            filled_price=0,
                            forecast=current_forecast,
                            volatility=vol,
                            target_pos=target_pos,
                            buffered_pos=buffered_pos,
                            success=False,
                            error=trade_result.error_message or "unknown",
                        )
                else:
                    logger.info(f"No trade needed for {instrument} (change {trade_size:.6f} < min {min_size})")
                    self.current_positions[symbol] = current_pos
                    proposed_positions[symbol] = current_pos

                results[instrument] = {
                    "forecast": current_forecast,
                    "volatility": vol,
                    "target_position": target_pos,
                    "buffered_position": buffered_pos,
                    "current_position": current_pos,
                    "trade_size": trade_size if abs(trade_size) >= min_size else 0,
                    "trade_result": trade_result,
                    "price": latest_price,
                    "risk_reasons": risk_reasons if should_reduce else [],
                    "rule_forecasts": inst_forecast_values,
                }

            # Log run summary
            run_duration = (datetime.now(timezone.utc) - run_start).total_seconds()
            self.trade_logger.log_run({
                "equity": run_equity,
                "positions": dict(self.current_positions),
                "prices": {k: v for k, v in latest_prices.items()},
                "forecasts": self.last_forecasts,
                "kill_switch": self.risk_manager.kill_switch_active,
                "duration_seconds": run_duration,
                "instruments_traded": list(results.keys()),
            })

            # Save state
            self._save_state(equity=run_equity, prices=latest_prices)
            self.last_run_ts = run_ts

        finally:
            await self.exchange.disconnect()

        return results

    def _save_state(self, equity: float = 0, prices: dict | None = None) -> None:
        """Persist system state to JSON."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": self.current_positions,
            "capital": self.capital,
            "equity": equity,
            "kill_switch": self.risk_manager.kill_switch_active,
            "peak_equity": self.risk_manager.peak_equity,
            "daily_start_equity": self.risk_manager.daily_start_equity,
            "prices": prices or {},
            "last_forecasts": self.last_forecasts,
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
            self.last_forecasts = state.get("last_forecasts", {})
            logger.info(f"Loaded state: {len(self.current_positions)} positions")

    def run(self) -> dict:
        """Synchronous entry point."""
        return asyncio.run(self.run_once())
