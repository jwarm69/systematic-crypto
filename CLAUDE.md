# Systematic Crypto Trading

## Architecture

Carver-style systematic trading system for crypto perpetual futures on Hyperliquid.

```
Data → Indicators → Trading Rules → Portfolio Construction → Execution
```

**Core principle:** Every signal is a "forecast" scaled to [-20, +20] (avg abs = 10).
Position size = (capital × vol_target × forecast) / (instrument_vol × price × 10).

## Project Structure

- `src/data/` - OHLCV fetching via CCXT, parquet caching, gap-fill
- `src/indicators/` - Volatility (EWMA), technical indicators
- `src/rules/` - Trading rules (EWMAC, carry, breakout, momentum, ML). Each outputs a forecast [-20, +20]
- `src/portfolio/` - Forecast combination, position sizing, buffering, cost model
- `src/risk/` - Kill switch, drawdown limits, correlation monitoring
- `src/execution/` - Hyperliquid adapter, paper trading mode
- `src/backtest/` - Vectorized engine, walk-forward CV, significance tests
- `src/system/` - Main trading loop, scheduler
- `src/dashboard/` - Streamlit monitoring UI
- `configs/` - YAML configuration (instruments, rules, portfolio, paper trading)

## Key Commands

```bash
# Install
pip install -e ".[dev]"

# Fetch data
python scripts/fetch_data.py --instrument BTC --bars 5000

# Run backtest
python scripts/run_backtest.py --config configs/portfolio.yaml

# Paper trade (Hyperliquid testnet)
python scripts/run_paper_trading.py --config configs/paper.yaml

# Dashboard
streamlit run src/dashboard/app.py
```

## Configuration

All config in `configs/` as YAML. Key parameters:
- `portfolio.yaml` - Capital, vol target, risk limits, forecast caps
- `instruments.yaml` - Instrument definitions (symbol, exchange, multiplier)
- `rules.yaml` - Rule parameters (EWMAC spans, breakout windows)
- `paper.yaml` - Hyperliquid testnet credentials

## Conventions

- Forecasts always scaled to [-20, +20], target avg abs = 10
- Volatility = annualized, using EWMA with 25-day span
- Returns = percentage returns (not log returns) for position sizing
- All prices in USD terms
- Timestamps in UTC
