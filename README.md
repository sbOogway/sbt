# SBT — Strategy Backtesting Tool

Modular crypto perpetual futures backtesting framework built on [Nautilus Trader](https://nautilustrader.io) + [CCXT](https://ccxt.readthedocs.io/). Supports concurrent execution with git worktree isolation, Optuna multi-objective optimization, and interactive tearsheet reporting.

---

## TL;DR Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Download historical market data
uv run python3 -m sbt.data --exchange binance --symbol BTC/USDT --interval 5m --start 2024-01-01 --type ohlcv

# 3. Run a direct single backtest
uv run python3 -m sbt --config config.toml --strategy overnight_drift
```

---

## Examples & Usage

### 1. Download Data (`sbt.data`)

Fetch OHLCV candles or funding rate history saved directly to Apache Feather format (`data/`):

```bash
# OHLCV candlestick data
uv run python3 -m sbt.data --exchange binance --symbol BTC/USDT --interval 5m --start 2024-01-01 --type ohlcv

# Funding rate history (interval not required)
uv run python3 -m sbt.data --exchange binance --symbol BTC/USDT --start 2024-01-01 --type funding
```

---

### 2. Standalone Backtest (`sbt`)

Execute a single strategy directly and open the interactive HTML tearsheet:

```bash
# Run strategy from config.toml
uv run python3 -m sbt --config config.toml --strategy overnight_drift

# Override parameters on the fly
uv run python3 -m sbt --strategy orb --leverage 2.0 --start 2023-01-01
```

Available strategies: `overnight_drift`, `bitcoin_intraday_momentum`, `glucksmann`, `orb`.

---

### 3. Concurrent Client-Server Backtesting

Run multiple backtests in parallel across isolated git worktrees (`.worktrees/`):

#### Start the Server Daemon
```bash
# Launch scheduler with 4 isolated workers
uv run python3 -m sbt.server --workers 4 --port 5555 --db sbt.db
```

#### Submit Jobs via Client CLI
```bash
# Submit a single job
uv run python3 -m sbt.client submit --config config.toml --strategy overnight_drift

# Submit all strategies defined in config.toml concurrently and wait for completion
uv run python3 -m sbt.client submit --config config.toml --all-strategies --wait

# Check worker pool and job status
uv run python3 -m sbt.client status

# Inspect full statistics for a completed job
uv run python3 -m sbt.client results --job <job_id>
```

---

### 4. Optuna Multi-Objective Optimization (`sbt.client optimize`)

Find the optimal strategy parameters by simultaneously maximizing **Sharpe Ratio**, **Total Trades**, and **Net PnL** (producing an interactive 3D Pareto frontier):

```bash
uv run python3 -m sbt.client optimize \
  --config config.toml \
  --strategy overnight_drift \
  --trials 50 \
  --param "rv_lookback=int(3,30)" \
  --param "vol_max_scale=float(1.0,4.0)" \
  --param "entry_time=cat(18:00,19:00,20:00,21:00)"
```

Parameter syntax:
- `name=int(min,max)`
- `name=float(min,max)`
- `name=cat(val1,val2,val3)`

Generates and opens `reports/pareto_report.html` with an interactive Plotly 3D scatter visualization.

---

### 5. Multi-Strategy Comparison Dashboard (`sbt.client compare`)

Compare performance metrics and equity returns across multiple runs side-by-side:

```bash
# Compare specific completed job IDs
uv run python3 -m sbt.client compare --jobs a3f2c1,b7e4d9,c1a8f3

# Output saved to reports/compare.html and auto-opened in browser
```

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

[strategy.overnight_drift]
entry_time = "20:00"
exit_time = "14:00"
vol_scaling = true
rv_lookback = 5
vol_max_scale = 2
weekdays_only = true
funding_enabled = false
```
