# SBT — Strategy Backtesting Tool

Modular crypto perpetual futures backtesting framework built on [Nautilus Trader](https://nautilustrader.io) + [CCXT](https://ccxt.readthedocs.io/). Supports Optuna multi-objective optimization and interactive tearsheet reporting.

---

## Installation

Requires Python >=3.14.

```bash
# Clone the repo
git clone https://github.com/sbOogway/sbt.git && cd sbt

# Install with uv (recommended)
uv sync

# Or install with pip
pip install .
```

After install, the `sbt` command is available globally. You can also run commands via `python -m sbt <command>`.

---

## TL;DR Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Set up git hooks (required for worktree auto-symlinking)
pre-commit install --hook-type post-checkout
git config core.mainRepo "$(pwd)"

# 3. Download historical market data
sbt data --exchange binance --symbol BTC/USDT --interval 5m --start 2024-01-01 --type ohlcv

# 4. Run a direct single backtest
sbt backtest --config config.toml --strategy overnight_drift
```

---

## Examples & Usage

### 1. Download Data (`sbt data`)

Fetch OHLCV candles or funding rate history saved directly to Apache Feather format (`data/`):

```bash
# OHLCV candlestick data
sbt data --exchange binance --symbol BTC/USDT --interval 5m --start 2024-01-01 --type ohlcv

# Funding rate history (interval not required)
sbt data --exchange binance --symbol BTC/USDT --start 2024-01-01 --type funding
```

---

### 2. Standalone Backtest (`sbt backtest`)

Execute a single strategy directly and open the interactive HTML tearsheet:

```bash
# Run strategy from config.toml
sbt backtest --config config.toml --strategy overnight_drift

# Override parameters on the fly
sbt backtest --strategy orb --leverage 2.0 --start 2023-01-01
```

Available strategies: `overnight_drift`, `bitcoin_intraday_momentum`, `glucksmann`, `orb`.

---

### 3. Optuna Multi-Objective Optimization

Find the optimal strategy parameters by simultaneously maximizing **Sharpe Ratio**, **Total Trades**, and **Net PnL** (producing an interactive 3D Pareto frontier):

```bash
sbt optimize --config config.toml --strategy overnight_drift --trials 50
```

Parameter syntax:
- `name=int(min,max)`
- `name=float(min,max)`
- `name=cat(val1,val2,val3)`

Generates and opens `reports/pareto_report.html` with an interactive Plotly 3D scatter visualization.

---

## Configuration (`config.toml`)

```toml
[run]
exchange = "BINANCE"
symbol = "BTC/USDT"
interval = "5m"
capital = 1000
leverage = 3.0
start = "2020-01-01"
maker_fee = 0.0
taker_fee = 0.00055
settle_currency = "USDT"
slippage_ticks = 2
tick_size = 0.01
```

Strategy parameters are **not** configured via `config.toml` — each
strategy file (`sbt/strategies/...`) is the single source of truth for
its parameters and defaults. Per-run overrides happen through the
optimizer (`--param`).
