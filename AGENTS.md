# SBT — Strategy Backtesting Tool

> **Agents:** before modifying code, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) —
> module map, execution pipeline, plugin contracts, server protocol,
> persistence schema, and gotchas. This file stays the quick ops manual.

Backtesting framework on [nautilus-trader](https://nautilustrader.io) + [ccxt](https://ccxt.readthedocs.io/) for crypto perpetual futures.

## Commands

### Single Backtest (Backward compatible)
```bash
uv run python3 -m sbt --config config.toml --strategy overnight_drift
# --no-open skips auto-opening the tearsheet in a browser
```

### Data Downloader
```bash
# Incremental by default: re-running with the same --output extends the file
# from its newest timestamp and renames it so the encoded date range matches
# the actual contents (--no-resume refetches; --page-limit overrides rows/call)
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

# Optuna hyperparameter optimization
#   --objective sharpe (default): 3-objective Pareto front (Sharpe + Trades + PnL)
#   --objective sqn: pure single-objective maximization of Van Tharp's System Quality Number
uv run python3 -m sbt.client optimize --config config.toml --strategy overnight_drift --trials 50 \
  --objective sqn \
  --param "rv_lookback=int(3,30)" \
  --param "vol_max_scale=float(1.0,4.0)"

# Train/validation holdout split (70% in-sample / 30% out-of-sample)
# Runs both windows; top-level result metrics = out-of-sample; per-window
# stats as in_sample_* / out_of_sample_* columns and one tearsheet per window.
uv run python3 -m sbt --config config.toml --strategy key_breakout --train-val-split 0.7
```

- `--strategy` defaults to `bitcoin_intraday_momentum` if omitted.
- `--type funding` ignores `--interval` (no interval param).
- Tests: `uv run pytest` — suite in `tests/` runs headless on synthetic bars via the runner's explicit `bars=`/`funding=` seam (no data files needed). No linting, typechecking, or CI exists.

## Agents

- **@paper-quant** (`.opencode/agent/paper-quant.md`) — autonomous agent: finds a quant research paper online, downloads the PDF into `papers/` (Sci-Hub fallback for paywalls), implements it as a new strategy following the conventions below, backtests it and reports honestly. Invoke via `@paper-quant <topic/request>`.

## Key Conventions

- **Times UTC** throughout; bars at UTC hour boundaries.
- **USDC settlement** for Hyperliquid perps.
- **Instrument factory**: `make_perpetual()` in `sbt/utils.py` (`price_precision=1`, `size_precision=3`).
- **Slippage in ticks**: `slippage_ticks * tick_size / ref_price * 10000` bps added to `taker_fee`. Single mechanism only — no `FillModel`.
- **Funding rates** tracked as metadata side-channel; does not flow through engine PnL.
- **Strategy base class**: bar-driven strategies subclass `SBTStrategy` (`sbt/strategies/base.py`) — compounding `equity()` sizing via the canonical `open_position(side, price)` (stop-distance strategies use `risk_quantity`), `FundingTracker` (base implements `on_funding_rate`; opt in via `subscribe_funding: bool = True` on the config), order gating via `active_from` (bars before it still feed indicators/plugins). Subclasses implement `on_trading_bar(bar)`. L2 strategies subclass `L2EventStrategy` (`sbt/strategies/l2/base.py` — shared book maintenance, sampling grid, sizing; no bar stream).
- **Plugin validation**: plugins declare `required_config_fields`; `PluginHost.from_config()` raises listing missing fields instead of silently defaulting.
- **Plugins**: strategies opt in via the flat `plugins: tuple[str, ...]` field on their config (e.g. `("vol_scaling",)`); plugin params stay flat on the config so optimizer specs keep working. Registry in `sbt/plugins/__init__.py`.
- **Vol scaling** is a plugin (`VolScalingPlugin`, Moreira & Muir rolling RV). Feeding modes: automatic daily close-to-close tracking (`vol_track_daily=True`) or manual `plugin.add_return()` when sampling follows a specific time/timezone. `vol_rebalance_freq` = `"daily"` (default) or `"monthly"` (weight refresh on month starts).
- **Position sizing**: `risk_percent * current_equity * leverage * plugins.size_multiplier()` (compounding).
- **FillModel**: removed. Slippage is fee-bps only (see above).
- **Data files** (`.feather`) auto-detected by `{exchange}_{symbol}_{tag}_*.feather` pattern in `data/` or `./` (`tag` = bar interval, or `funding`). The whole naming contract is owned by `sbt/core/feather.py` — never hand-build or hand-parse these filenames. Unprefixed files are used only when unique. Multiple matches: best coverage of `[start, end]` wins (then overlap, then newest); chosen file is printed. Funding files matched by tag `funding`.
- **Runner data seam**: `run(bars=…, funding=…)` accepts explicit frames — headless and deterministic, zero disk reads (this is what tests use). Default `None` keeps feather auto-discovery; explicit bars without a funding frame run without funding rather than searching disk.
- **Tearsheets** saved to `reports/` and auto-opened via `webbrowser` unless `--no-open` (or config `open_report: false`).
- `data/`, `reports/`, `.worktrees/`, `*.db` are gitignored.

## Plugins

Strategy-level plugins (`sbt/plugins/base.py`) receive forwarded lifecycle events from a `PluginHost` and may implement `SizingPlugin.size_multiplier()`. Runner-level plugins expand one job into windows.

Adding a strategy-level plugin:
1. Create `sbt/plugins/<name>.py` with `<Name>Plugin(StrategyPlugin)` (or `SizingPlugin`) and a unique `name` ClassVar. Params are read off the host strategy's config via `getattr` defaults.
2. Register in `sbt/plugins/__init__.py` `_PLUGIN_REGISTRY`.
3. Strategies enable it by adding the name to their `plugins` tuple; document any new config fields on each adopting strategy's `<Name>Config`.

## Adding a Strategy

1. Create `sbt/strategies/ohlc/<name>.py` (bar-driven) or `sbt/strategies/l2/<name>.py` (order-book-driven). Bar-driven: `<Name>Config(SBTBarStrategyConfig, kw_only=True, frozen=True)`; L2: `<Name>Config(SBTStrategyConfig, kw_only=True, frozen=True)` — the tiers already carry the shared fields (`instrument_id`, `bar_type` [bar tier only], `capital`, `leverage`, `backtest_start_date`, `active_from`, `risk_percent`, `subscribe_funding`), so declare only signal parameters and their defaults. Strategy class: `<Name>(SBTStrategy)` for bar mode, `<Name>(L2EventStrategy)` for L2. Set `plugins` defaults in the config (e.g. `plugins: tuple[str, ...] = ("vol_scaling",)`), instantiate `self.plugins = PluginHost.from_config(config)`, forward `self.plugins.on_bar(self, bar)` from `on_trading_bar`, and size via `open_position(side, price)` (stop-distance: `risk_quantity`). Note: `kw_only=True` is required on every config subclass — msgspec does not inherit it, and overriding an inherited field without it breaks struct construction.
2. Register in `sbt/utils.py` `_STRATEGY_REGISTRY` with the module path under the strategy's folder.
3. Run: `uv run python3 -m sbt --strategy <name>` or submit via `sbt.client`.

Strategy parameters are **not** configured via `config.toml` — the `[strategy.*]` sections were removed; the strategy file is the single source of truth. Per-run overrides only happen through the optimizer (`--param`) / server (`with_overrides`).

## Structure

```
sbt/
├── __main__.py             CLI entry point (thin, delegates to core.runner)
├── data.py                 Data downloader (ccxt → feather)
├── report.py               HTML tearsheet + TradingView chart
├── stats.py                Custom portfolio statistics
├── utils.py                Strategy loader, instrument factory
├── plugins/                Plugin system
│   ├── base.py             SBTStrategyConfig, StrategyPlugin/SizingPlugin ABCs, PluginHost
│   ├── vol_scaling.py      VolScalingPlugin (Moreira & Muir rolling RV)
│   └── train_val_split.py  Runner-level IS/OOS holdout split
├── core/                   Extracted backtest primitives
│   ├── config.py           RunConfig dataclass (TOML + CLI → config)
│   ├── runner.py           BacktestRunner (engine setup + execution, window splitting)
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
└── strategies/
    ├── base.py               SBTStrategy base class + FundingTracker
    ├── ohlc/                 Bar-driven strategies (SBTStrategy subclasses)
    │   ├── overnight_drift.py
    │   ├── bitcoin_intraday_momentum.py
    │   ├── glucksmann.py
    │   ├── key_breakout.py
    │   └── orb.py
    └── l2/                   Order-book-driven strategies (L2EventStrategy subclasses)
        └── order_imbalance.py  (registry name: l2_order_imbalance)
papers/                     Reference PDFs organized by strategy kind (not code)
├── ohlc/                   Papers behind bar-driven strategies
├── l2/                     Order-book / microstructure papers
└── plugins/                Papers behind plugins (e.g. vol scaling)
data/                       .feather files (gitignored)
reports/                    Generated HTML tearsheets (gitignored)
.worktrees/                 Worker isolated checkouts (gitignored)
sbt.db                      SQLite results & Optuna study database (gitignored)
config.toml                 Run + strategy parameters
```

## Gotchas

- `BarDataWrangler` (nautilus 1.230.0) raises "buffer source array is read-only" for every input shape — use `load_bars()` in `sbt/core/runner.py` instead.
- Server workers run from `.worktrees/worker-N` checkouts; until changes are committed, re-seed them after edits: `cp -a sbt/. .worktrees/worker-0/sbt/`.

## Dependencies

- Python >=3.14 (`.python-version`)
- `nautilus-trader[visualization]`, `ccxt`, `pyzmq`, `optuna` — managed via `uv`

## Agent skills

### Issue tracker

Issues live as GitHub Issues on `sbOogway/sbt`, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.
