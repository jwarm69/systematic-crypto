"""Trade and PnL logging for production system.

Logs every trade to CSV and maintains daily PnL history in JSON.
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeLogger:
    """Persistent trade and PnL logger."""

    def __init__(
        self,
        trade_log_path: str | Path = "logs/trades.csv",
        pnl_history_path: str | Path = "data/pnl_history.json",
        run_log_path: str | Path = "data/run_history.json",
    ):
        self.trade_log_path = Path(trade_log_path)
        self.pnl_history_path = Path(pnl_history_path)
        self.run_log_path = Path(run_log_path)

        # Ensure directories exist
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.pnl_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_trade_csv()

    def _ensure_trade_csv(self) -> None:
        """Create CSV header if file doesn't exist."""
        if not self.trade_log_path.exists():
            with open(self.trade_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "instrument", "symbol", "side", "size",
                    "price", "filled_price", "notional", "forecast",
                    "volatility", "target_pos", "buffered_pos", "success",
                    "error",
                ])

    def log_trade(
        self,
        instrument: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        filled_price: float,
        forecast: float,
        volatility: float,
        target_pos: float,
        buffered_pos: float,
        success: bool,
        error: str = "",
    ) -> None:
        """Append a trade to the CSV log."""
        ts = datetime.now(timezone.utc).isoformat()
        notional = size * filled_price if filled_price > 0 else size * price
        with open(self.trade_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts, instrument, symbol, side, f"{size:.8f}",
                f"{price:.2f}", f"{filled_price:.2f}", f"{notional:.2f}",
                f"{forecast:.2f}", f"{volatility:.4f}",
                f"{target_pos:.8f}", f"{buffered_pos:.8f}",
                success, error,
            ])
        logger.info(f"Logged trade: {side} {size:.6f} {instrument} @ {filled_price:.2f}")

    def log_run(self, run_data: dict) -> None:
        """Append a run summary to the run history."""
        history = self._load_json(self.run_log_path, default=[])
        run_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **run_data,
        }
        history.append(run_entry)
        # Keep last 2000 entries
        if len(history) > 2000:
            history = history[-2000:]
        self._save_json(self.run_log_path, history)

    def log_daily_pnl(
        self,
        date: str,
        equity: float,
        pnl: float,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> None:
        """Log daily PnL snapshot."""
        history = self._load_json(self.pnl_history_path, default=[])

        # Check if we already have an entry for this date
        for entry in history:
            if entry.get("date") == date:
                # Update existing entry
                entry["equity"] = equity
                entry["pnl"] = pnl
                entry["positions"] = positions
                entry["prices"] = prices
                self._save_json(self.pnl_history_path, history)
                return

        history.append({
            "date": date,
            "equity": equity,
            "pnl": pnl,
            "positions": positions,
            "prices": prices,
        })
        self._save_json(self.pnl_history_path, history)

    def get_trade_history(self, n: int = 100) -> list[dict]:
        """Read last N trades from CSV."""
        if not self.trade_log_path.exists():
            return []
        trades = []
        with open(self.trade_log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
        return trades[-n:]

    def get_pnl_history(self) -> list[dict]:
        """Read full PnL history."""
        return self._load_json(self.pnl_history_path, default=[])

    def get_run_history(self, n: int = 100) -> list[dict]:
        """Read last N run summaries."""
        history = self._load_json(self.run_log_path, default=[])
        return history[-n:]

    @staticmethod
    def _load_json(path: Path, default=None):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default if default is not None else {}

    @staticmethod
    def _save_json(path: Path, data) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
