# SBT — Strategy Backtesting Tool

Backtesting framework on [nautilus-trader](https://nautilustrader.io) + [ccxt](https://ccxt.readthedocs.io/) for crypto perpetual futures.

## Commands

### Single Backtest (Backward compatible)
```bash
uv run python3 -m sbt --config config.toml --strategy overnight_drift
```

### Data Downloader
```bash
uv run python3 -m sbt.data --exchange hyperliquid --symbol XYZ-SP500/USDC:USDC --interval 1h --start 2026-03-18 --type ohlcv
uv run python3 -m sbt.data --exchange hyperliquid --symbol XYZ-SP500/USDC:USDC --start 2026-03-18 --type funding
```

### Server (Scheduler Daemon + Git Worktree Isolation)
```bash
uv run python3 -m sbt.server --workers 4 --port 5555 --worker-port 5556 --db sbt.db
```

### Client CLI
```bash
# Submit single strategy or batch of all strategies
uv run python3 -m sbt.client submit --config config.toml --strategy overnight_drift
uv run python3 -m sbt.client submit --config config.toml --all-strategies --wait

# Status and results
uv run python3 -m sbt.client status
uv run python3 -m sbt.client results --job <job_id>

# Multi-strategy comparison dashboard
uv run python3 -m sbt.client compare --jobs <id1,id2,id3>

# Optuna multi-objective hyperparameter optimization (Sharpe + Trades + PnL Pareto front)
uv run python3 -m sbt.client optimize --config config.toml --strategy overnight_drift --trials 50 \
  --param "rv_lookback=int(3,30)" \
  --param "vol_max_scale=float(1.0,4.0)"
```

- `--strategy` defaults to `bitcoin_intraday_momentum` if omitted.
- `--type funding` ignores `--interval` (no interval param).
- No tests, linting, typechecking, or CI exist. No verification step to run.

## Key Conventions

- **Times UTC** throughout; bars at UTC hour boundaries.
- **USDC settlement** for Hyperliquid perps.
- **Instrument factory**: `make_perpetual()` in `sbt/utils.py` (`price_precision=1`, `size_precision=3`).
- **Slippage in ticks**: `slippage_ticks * tick_size / ref_price * 10000` bps added to `taker_fee`.
- **Funding rates** tracked as metadata side-channel; does not flow through engine PnL.
- **Vol scaling** is rolling (daily), not monthly — `add_return()` updates weight each call.
- **Position sizing**: `risk_percent * current_equity * leverage * vol_weight` (compounding).
- **FillModel**: `FillModel(prob_slippage=1.0)` when `slippage_ticks > 0` (1 tick nautilus slippage).
- **Data files** (`.feather`) auto-detected by `{exchange}_{symbol}_{interval}_*.feather` pattern in `data/` or `./`. Funding files matched by `*funding*` in path.
- **Tearsheets** saved to `reports/` and auto-opened via `webbrowser`.
- `data/`, `reports/`, `.worktrees/`, `*.db` are gitignored.

## Adding a Strategy

1. Create `sbt/strategies/<name>.py` with `<Name>Config(StrategyConfig, frozen=True)` and `<Name>(Strategy)`.
2. Register in `sbt/utils.py` `get_strategy_class()`.
3. Add `[strategy.<name>]` section in `config.toml`.
4. Run: `uv run python3 -m sbt --strategy <name>` or submit via `sbt.client`.

## Structure

```
sbt/
├── __main__.py             CLI entry point (thin, delegates to core.runner)
├── data.py                 Data downloader (ccxt → feather)
├── report.py               HTML tearsheet + TradingView chart
├── stats.py                Custom portfolio statistics
├── utils.py                Strategy loader, instrument factory
├── volatility.py           Moreira & Muir rolling vol scaling
├── core/                   Extracted backtest primitives
│   ├── config.py           RunConfig dataclass (TOML + CLI → config)
│   ├── runner.py           BacktestRunner (engine setup + execution)
│   ├── job.py              BacktestJob / BacktestResult models
│   └── db.py               SQLite result store
├── server/                 Scheduler daemon + git worktree supervisor
│   ├── scheduler.py        ZMQ ROUTER scheduler & dispatcher
│   ├── worker.py           ZMQ DEALER worker with isolated worktrees
│   └── __main__.py         python -m sbt.server CLI
├── client/                 Client CLI
│   ├── client.py           ZMQ SbtClient helper
│   ├── cli.py              Subcommand handlers
│   └── __main__.py         python -m sbt.client CLI
├── optimize/               Optuna multi-objective optimization
│   ├── param_parser.py     Parameter space specification parser
│   ├── study.py            Optuna study coordinator (Sharpe + Trades + PnL)
│   └── report.py           Plotly 3D Pareto frontier HTML report
├── compare/                Multi-strategy comparison dashboard
│   └── dashboard.py        Side-by-side metric tables + comparison charts
└── strategies/
    ├── overnight_drift.py
    ├── bitcoin_intraday_momentum.py
    ├── glucksmann.py
    └── orb.py
papers/                     Reference PDFs (not code)
data/                       .feather files (gitignored)
reports/                    Generated HTML tearsheets (gitignored)
.worktrees/                 Worker isolated checkouts (gitignored)
sbt.db                      SQLite results & Optuna study database (gitignored)
config.toml                 Run + strategy parameters
```

## Stale Artifacts

- `sbt/strategies/orb.py.bak` — backup file, ignore it.

## Dependencies

- Python >=3.14 (`.python-version`)
- `nautilus-trader[visualization]`, `ccxt`, `pyzmq`, `optuna` — managed via `uv`
