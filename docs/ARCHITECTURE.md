# SBT Architecture Reference

Single-file reference for coding agents. Read this instead of scanning `sbt/`.
`AGENTS.md` covers commands and recipes ("how do I run X"); this document
covers mechanics ("how does X work", "what breaks if I touch Y"). Anchors are
`file.py` + symbol names — grep those, never line numbers.

## 1. System overview

Crypto perpetual-futures backtesting on nautilus-trader + ccxt data
ingestion, an optional distributed scheduler, and Optuna hyperparameter
optimization.

```
 python -m sbt           single run ─┐
 python -m sbt.client    submit/optimize/status ─┐
                                 ZMQ (DEALER)    │
                                                 ▼
                                      Scheduler (ROUTER :5555)
                                                 │ JOB dispatch on :5556
                                                 ▼
                       Workers ×N (DEALER, isolated git worktrees)
                                                 │
                             BacktestRunner ──► BacktestEngine (nautilus)
                                                 │
                                   BacktestResult (JSON-safe dict)
                          ┌──────────────────────┼──────────────────┐
                    tearsheets             ResultStore          RESULT msg
                   reports/*.html         sqlite sbt.db      back to scheduler
```

Every execution path — CLI, server job, optimizer trial — converges on
`core/runner.py::BacktestRunner.run()`. There is exactly one engine-setup
code path per data mode (`bar`, `l2`) inside `_run_window`.

**Data flow (bar mode):**
`data/*.feather` → `find_feather()` → `pd.read_feather` → date filter →
(optional runner-plugin window slicing incl. warm-up bars) →
`load_bars()` → `engine.add_data(bars)` (+ optional funding updates) →
`engine.run()` → analyzer stats + positions/fills reports → `BacktestResult`.

## 2. Module map

| Path | Responsibility | Key symbols |
|---|---|---|
| `sbt/__main__.py` | CLI entry: parse config, run, print report | `RunConfig.parse_cli`, `BacktestRunner.run` |
| `sbt/core/config.py` | Run configuration; TOML/CLI/JSON codec | `RunConfig`, `_coerce` |
| `sbt/core/job.py` | Job/result models + JSON codecs | `BacktestJob`, `BacktestResult`, `JobStatus` |
| `sbt/core/runner.py` | Engine setup + execution + windowing | `BacktestRunner`, `run/_run_windows/_run_window`, `load_bars`, `resolve_runner_plugin`, `INLINE_ROW_BUDGET` |
| `sbt/core/db.py` | SQLite store (jobs/results), migrations | `ResultStore`, `_SCHEMA_VERSION` |
| `sbt/core/l2.py` | Nautilus catalog L2 loaders | `load_order_book_deltas`, `load_trade_ticks`, `list_l2_instruments` |
| `sbt/core/feather.py` | Feather filename contract: naming, range parsing, discovery/ranking, resume healing | `feather_path`, `parse_range`, `find_feather`, `actual_range_name`, `to_utc_ts` |
| `sbt/utils.py` | Strategy registry, instrument factory, intervals | `_STRATEGY_REGISTRY`, `get_strategy_class`, `make_perpetual`, `parse_interval`, `interval_delta` |
| `sbt/stats.py` | Custom portfolio statistics | `system_quality_number`, `CalmarRatio`, `AnnualizedReturn`, `RunConfigStatistic` |
| `sbt/plugins/base.py` | Plugin contracts + host + config tiers + Window | `SBTStrategyConfig`, `SBTBarStrategyConfig`, `StrategyPlugin`, `SizingPlugin`, `RunnerPlugin`, `Window`, `PluginHost` |
| `sbt/plugins/vol_scaling.py` | Moreira–Muir realized-vol sizing | `VolScalingPlugin` |
| `sbt/plugins/train_val_split.py` | Runner-level IS/OOS holdout | `TrainValSplit`, `IN_SAMPLE`, `OUT_OF_SAMPLE` |
| `sbt/plugins/__init__.py` | Plugin registries | `_PLUGIN_REGISTRY`, `_RUNNER_PLUGIN_REGISTRY`, `get_plugin_class`, `get_runner_plugin_class` |
| `sbt/strategies/base.py` | Shared strategy plumbing | `SBTStrategy`, `FundingTracker` |
| `sbt/strategies/ohlc/*.py`, `sbt/strategies/l2/*.py` | Concrete strategies (see §5) | registry names |
| `sbt/data.py` | ccxt → feather downloader | `fetch_ohlcv`, `fetch_funding_rates`, `_paginate`, `main` |
| `sbt/report.py` | HTML tearsheet + TV chart | `print_report` |
| `sbt/server/scheduler.py` | ZMQ ROUTER scheduler/dispatcher | `Scheduler` |
| `sbt/server/worker.py` | DEALER worker in git worktree | `Worker`, `ensure_worktree` |
| `sbt/client/client.py` | DEALER client helper | `SbtClient._request` |
| `sbt/client/cli.py` / `__main__.py` | Client subcommands | `cmd_submit/cmd_status/cmd_results/cmd_optimize` |
| `sbt/optimize/study.py` | Optuna orchestration + executors | `run_optuna_study`, `LocalExecutor`, `SchedulerExecutor` |
| `sbt/optimize/param_parser.py` | Param spec grammar | `parse_param_spec`, `suggest_params` |
| `sbt/optimize/report.py` | Pareto/SQN HTML reports | `generate_pareto_report`, `generate_sqn_report` |
**Import graph rule:** `plugins.base` imports `core.job` only;
`core.runner` imports `utils` (which lazily imports strategies) and
`plugins`; strategies import `plugins` + `strategies.base`. Deferred imports
inside functions (`TrainValSplit.expand` pulling `utils.interval_delta`,
`PluginHost.from_config` pulling the registry) exist to break import cycles —
keep new shared code out of these cycles or defer it the same way.

## 3. Configuration (`RunConfig`)

One flat dataclass; strategy parameters are NOT part of it — they live as
fields on each `<Strategy>Config` and travel inside
`RunConfig.strategy_params` (dict). Single source of truth is the strategy
file; per-run overrides happen only via optimizer specs or server overrides.

Precedence (lowest → highest): `config.toml [run]` table → CLI flags →
optimizer/server overrides (`with_overrides(params)` merges into
`strategy_params` only).

Codec: `to_dict`/`from_dict` derive from `dataclasses.fields` +
type hints; `_coerce` handles Decimal/int/float/bool/Optional so JSON round
trips (ZMQ, DB) keep types. Adding a field needs no codec change.
`from_toml` maps `[run]` keys 1:1 onto fields (CLI key `"feather"` aliases
to `feather_path`). L2 mode inference: symbol contains `-LINEAR.` or
`data_type == "l2"`.

Notable fields: `train_val_split` (float fraction or None),
`warmup_bars` (bars preloaded before a window's trading start),
`open_report` (browser auto-open; CLI `--no-open` sets False).

## 4. Execution pipeline

### 4.1 Dispatch

`BacktestRunner.run(job_id, bars=None, funding=None)`:
1. `resolve_runner_plugin(cfg)` — returns `TrainValSplit(cfg.train_val_split)`
   when set, else None.
2. With plugin → `_run_windows`: load+filter bar frame once (skipped for
   L2 and when explicit `bars` are given), `plugin.expand(cfg, df)` →
   `{key: Window}`, run each window through
   `_run_window(f"{job_id}:{key}", ..., bars=win.df)`; engines collected
   into `runner.window_engines[key]`; merge via `plugin.combine(...)`,
   print via `plugin.summarize(...)`.
3. Without plugin → single `_run_window(job_id, cfg.start, cfg.end)`.
   `expand()` raising `ValueError` becomes a FAILED result.

**Data-source seam**: explicit `bars`/`funding` frames bypass all file
discovery — the frame is used as-is (caller owns slicing/warm-up) and no
feather is read, so runs are headless and deterministic (this is what the
`tests/` suite runs on). Explicit bars without a funding frame run without
funding instead of searching disk. `None` (default) preserves the feather
convention exactly.

### 4.2 `_run_window` — bar mode

1. Frame source: explicit `bars` used as-is; else resolve feather
   (`cfg.feather_path` or `find_feather`), read, require columns
   `[timestamp, open, high, low, close, volume]`.
2. Discovery path slices to `[start, end]`; explicit frames are trusted
   as-is. <2 rows → FAILED.
3. Slippage: `slippage_bps = slippage_ticks * tick_size / ref_price * 10000`
   where `ref_price = first close`; effective
   `taker_fee = cfg.taker_fee + Decimal(slippage_bps) / 10000` (fee as a
   notional fraction). This is the ONLY slippage mechanism — no FillModel.
4. Venue setup: `OmsType.NETTING`, `AccountType.MARGIN`, settle currency
   from `cfg.settle_currency` (USDT/USDC map else synthesized), starting
   balance = `cfg.capital`, leverage = `cfg.leverage`.
5. Instrument: `make_perpetual()` (`utils.py`) — price_precision=1,
   size_precision=3, id `{SYMBOL}-PERP`.
6. Bar type: `{instrument_id}-{parse_interval(interval)}-LAST-EXTERNAL`.
7. Strategy config built by the runner: always passes
   `instrument_id, bar_type, capital, leverage, backtest_start_date,
   active_from=start.isoformat(), **cfg.strategy_params`.
8. Funding side-channel: an explicit `funding` frame wins (sliced to the
   window); else `find_feather(..., "funding")` matches
   `{exchange}_{symbol}_funding_*.feather`; loaded rows become
   `FundingRateUpdate`s (also date-filtered). Missing file just logs.
9. Register custom stats (`CalmarRatio`, `AnnualizedReturn`,
   `RunConfigStatistic`), run engine, collect:
   - objectives: `PnL (total)`, `Sharpe Ratio (252 days)`,
     `num_trades = len(positions_df)`, `sqn` from closed positions'
     `realized_return` column via `system_quality_number`
   - `funding_pnl = strategy.funding.total_paid` when the strategy exposes
     a typed `funding` tracker
   - positions/fills reports packaged by `_spill_artifacts`.

### 4.3 `_run_window` — L2 mode

Symbol matched against `list_l2_instruments(catalog_dir)` (substring
normalization); instrument from `load_l2_instrument`; venue/book from the
instrument; deltas + trades loaded via
`load_order_book_deltas/load_trade_ticks` which expect **plain date
strings** — hence `start_str = str(to_utc_ts(start))` normalization
(`pd.Timestamp(x, tz="UTC")` raises on tz-aware input). Strategy kwargs
mirror bar mode minus `capital` semantics differences; `bar_type` added
only when the ConfigClass declares one.

### 4.4 Data loading

`load_bars(df, bar_type, instrument)` converts via int64-nanos timestamps
(`as_unit("ns")` → numpy columns → `Bar(...)` per row).
Do NOT use nautilus' `BarDataWrangler` — broken on nautilus 1.230.0
("buffer source array is read-only" for every input shape).

`core/feather.py` owns the whole filename contract; the runner only
calls into it. `find_feather(exchange, symbol, tag, dirs, start, end)`:
pattern `{exchange.lower()}_{symbol_without_slash}_{tag}_*.feather`
(*tag* = bar interval for OHLCV, `funding` for funding rates); bare
(unprefixed) files count only when unique; ranking among matches =
covers [start,end] > max overlap > newest range end; searches
`[cfg.data_dir, "."]` in order; prints its choice when ambiguous.
Range suffix `_YYYYMMDD[_YYYYMMDD].feather` parsed by `parse_range`;
the downloader names files via `feather_path()` and heals stale
suffixes on resume via `actual_range_name()`, so a conventional
filename always states the range of its contents.

## 5. Strategy framework

Bar strategies subclass `SBTStrategy` (`strategies/base.py`); the only
required override is `on_trading_bar(bar)`.

Lifecycle: `SBTStrategy.on_start` forwards `plugins.on_start` and
subscribes `config.bar_type`. `on_bar(bar)` records `bar.ts_event` and
the latest close (`_latest_price`), then forwards to plugins, then calls
`on_trading_bar(bar)` — **every** bar arrives (pre-`active_from` ones
too) so indicators/plugins warm up.

Window gating: `trading_active` is False while
`_current_ts_ns < _active_from_ns` (ISO string on config). Orders go
through `submit_market(side, qty)`, which is a no-op during warm-up.
`enter_market/exit_market` maintain `position_side` / `_open_qty`;
`in_position` reports the tracked state; `exit_market` settles funding
accrual for the closing position.

Sizing: `equity()` reads the live account total balance each time →
compounding by construction. The canonical formula is owned once by
`SBTStrategy.open_position(side, price)`:
`notional = equity() * config.risk_percent * leverage
* plugins.size_multiplier()`; quantity = `sized_quantity(notional /
price)` (rounds to 3dp, None when <= 0). Full-notional strategies call
`open_position`; `risk_percent` lives on the shared `SBTStrategyConfig`
tier. Stop-based strategies use the shared `risk_quantity(stop_distance,
risk_fraction)`: risk amount = equity × leverage × risk_fraction ×
plugin multiplier, quantity = amount / stop_distance.

`FundingTracker`: signed from holder's perspective — a long paying a
positive rate accrues cost; `total_paid > 0` means the strategy paid.
The base class implements `on_funding_rate(FundingRateUpdate)` (accrues
against the open position at `_latest_price`); strategies opt in via
`subscribe_funding: bool = True` on their config (the base subscribes in
`on_start`). Funding does NOT flow through engine PnL — it is metadata
reported as `BacktestResult.funding_pnl`.

Registry: `utils._STRATEGY_REGISTRY`, name → (module, strategy class,
config class). Bar-driven strategies live in `strategies/ohlc/`:
`bitcoin_intraday_momentum`, `glucksmann`,
`key_breakout`, `orb`, `overnight_drift`, plus the reversal/seasonality
pair (all SBTStrategy subclasses). Order-book strategies live in
`strategies/l2/`, all subclasses of `L2EventStrategy` (`strategies/l2/
base.py`: maintains the L2 book from deltas, samples signals on a time
grid via `_sample_due`, executes entries/exits with equity-fraction
sizing; shared EWMA helpers `clamped_dt_s`/`ewma_alpha`) — including
`l2_order_imbalance`, whose blended composite signal is pure subclass
code on that base.

### Config hierarchy & runner-injected fields

`SBTStrategyConfig` (plugins/base.py) carries every field the runner
injects at construction: `instrument_id` (required), `capital`,
`leverage`, `backtest_start_date`, `active_from` — concrete strategies
declare only signal parameters. `SBTBarStrategyConfig(SBTStrategyConfig)`
adds `bar_type` for bar-driven strategies; L2 configs subclass the base
directly and never receive one. Construction goes through
`core.runner._build_strategy_config`, which raises (→ FAILED result)
listing unknown keys + valid fields, so optimizer/server typos fail
loudly. Tests pin this per registered strategy
(`tests/test_strategy_configs.py`).

### msgspec gotcha (breaks construction)

nautilus `StrategyConfig` is a msgspec Struct. Subclasses must declare:

```python
class XConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
```

`kw_only=True` is NOT inherited, and re-declaring an inherited field (e.g.
`plugins`) in a child without it raises
`TypeError: Required field 'instrument_id' cannot follow optional fields`.

## 6. Plugin system

Two plugin kinds, both registered in `plugins/__init__.py`:
strategy-level (`_PLUGIN_REGISTRY`: `vol_scaling`) and runner-level
(`_RUNNER_PLUGIN_REGISTRY`: `train_val_split`).

### Strategy-level

- `StrategyPlugin(ABC)`: constructed with the full strategy config; hooks
  `on_start/on_bar/on_stop(strategy, ...)`. ClassVars: `name`,
  `required_config_fields` (enforced), `optional_config_fields`
  (documentation only).
- `SizingPlugin(StrategyPlugin)`: adds `size_multiplier() -> float`.
- `PluginHost.from_config(config)`: instantiates each name in
  `config.plugins`; validates every required field exists on the host
  config (msgspec `__struct_fields__` or dataclass fields) and raises
  listing missing names — silent getattr defaults are disabled so optimizer
  typos fail loudly.
- Host usage pattern (strategies own one host):
  `self.plugins = PluginHost.from_config(config)` in `__init__`; forward
  `self.plugins.on_bar(self, bar)`; multiply notional by
  `self.plugins.size_multiplier()`.
- Params stay flat fields on the strategy config (getattr with defaults)
  so optimizer specs like `rv_lookback=int(3,30)` work unchanged.

`VolScalingPlugin`: weight = `min(vol_max_scale, C/RV)` with RV = sum of
squared returns over trailing `rv_lookback` returns, C = running mean of RV
history; capped at `vol_max_scale` when RV=0. Feeding modes: automatic
(`vol_track_daily=True`; close-to-close across date rollovers of the
forwarded bar stream) or manual (`plugin.add_return(ret)` — used when
sampling follows a specific clock/timezone, e.g. overnight_drift feeds at
its configurable entry hour). Rebalance: `"daily"` updates the weight per
fed return; `"monthly"` stages into `_pending_weight` and applies it on
the first day-completion whose incoming month differs (weight constant
within month; requires automatic tracking so month boundaries are known).

### Runner-level

`RunnerPlugin(ABC)` contract:
- `expand(cfg, df) -> dict[str, Window]` — derive named windows; raise
  ValueError on bad config/data (becomes a FAILED result).
- `combine(job_id, results, windows) -> BacktestResult` — merge results.
- `summarize(results)` — optional console summary.

`Window` NamedTuple `(label, start, end, df=None)`: `start/end` are the
TRADING bounds (enforced via strategy `active_from`), while `df` may
include warm-up rows before `start` (the runner uses plugin frames
as-is; slicing is the plugin's job).

`TrainValSplit(fraction)`:
- `split_ts = first_ts + span * fraction`; IS ends `split_ts - one bar`
  (the boundary bar belongs to OOS only — no duplicate bar).
- OOS slice preloads `cfg.warmup_bars * interval_delta(interval)` before
  split_ts, clamped to range start.
- With `df=None` (L2 mode) requires explicit `cfg.end`.
- `combine` promotes OOS metrics to top-level result fields (optimizers
  therefore validate out-of-sample); per-window metrics stored as
  first-class columns (`in_sample_*` / `out_of_sample_*`); durations
  summed; positions/fills/stats come from OOS.

## 7. Server architecture

Topology: scheduler binds ROUTER on the client endpoint (:5555 default)
and ROUTER on the worker endpoint (:5556). Workers connect DEALER with
identity `worker-N`. Clients use DEALER sending `[b"", payload]` and read
the LAST frame of the reply. (History: REQ never worked here — its empty
delimiter frame broke the scheduler's `[identity, payload]` framing; the
scheduler now tolerates both by taking `msg_parts[2] if len >= 3 else
msg_parts[1]`.)

Client actions (`Scheduler._handle_client_req`): `submit`, `submit_batch`,
`status`, `get_result`, `list_results`, `ping`. Replies are JSON with
`status` of `ok` / `pending` / `not_found` / `error`.

Worker protocol (`Scheduler._handle_worker_msg`): sends `READY`, then per
job `ACK{job_id}` immediately on receipt (so the ACK timer measures
delivery, not execution), then `RESULT{job_id, result}`; answers `PING`
with `PONG`; receives `JOB{job}`, `PING`, `SHUTDOWN`.

Durability model — every sweep runs each poll tick (500 ms):

| Sweep | Env knob (default) | Behavior |
|---|---|---|
| Startup reconcile | — | stale RUNNING→PENDING; enqueue pending FIFO by submitted_at |
| `_check_acks` | SBT_ACK_TIMEOUT (30s) | no ACK in time → kill worker BEFORE requeue, respawn |
| `_heartbeat` | SBT_HEARTBEAT_INTERVAL (10s), SBT_MAX_MISSED_PONGS (3) | PING busy workers; reap after missed pongs |
| `_check_job_timeouts` | job.timeout_seconds (3600) | past budget → treat as worker death |
| `_check_child_procs` | — | exited Popen → death handling |

Death handling `_handle_worker_death`: kill proc → locate its job via
busy/awaiting_ack maps → `_requeue_or_fail`: requeue at queue FRONT while
`attempts < MAX_ATTEMPTS` (env SBT_MAX_ATTEMPTS=2), else FAILED.
RESULT ownership check: results from a non-owner
(`active_owner(job_id) != sender`) are dropped — this is what makes
requeue races safe.

Worktree isolation (`ensure_worktree`, worker.py): creates
`.worktrees/{worker_id}` via `git worktree add --detach HEAD`, symlinks
repo `data/`, creates `reports/`. Returns True only for a real checkout;
mkdir fallback logs a DEGRADED-isolation warning. Workers spawn with
`cwd=worktree` and PYTHONPATH prefixed with the worktree, so
`python -m sbt.server.worker` imports worktree code. Consequence:
uncommitted changes are invisible to workers until committed or re-seeded
(`cp -a sbt/. .worktrees/worker-0/sbt/`). Worker test hooks: env
SBT_JOB_DELAY_S sleeps before running (deterministic interruption tests);
during execution it chdirs into the worktree and points data_dir at its
symlink.

## 8. Persistence

SQLite `sbt.db` via `ResultStore`; same file doubles as Optuna storage
(`sqlite:///...`). WAL journal + busy_timeout=5000 for concurrent readers.
Versioned idempotent migrations driven by `schema_meta.version`
(`_SCHEMA_VERSION = 3`).

Tables:

- `jobs(id PK, status, strategy_name, config_json, worker_id, study_name,
  submitted_at, timeout_seconds, attempts)`
- `results(...)` — the columns are DERIVED from the `BacktestResult`
  dataclass (`core.job.result_field_specs`): every scalar field becomes a
  queryable column of the same name (sharpe_ratio, num_trades, pnl, sqn,
  error, duration_seconds, funding_pnl, positions_path, fills_path,
  positions_count, fills_count, in_sample_sharpe_ratio,
  in_sample_num_trades, in_sample_pnl, in_sample_sqn,
  in_sample_funding_pnl, in_sample_duration_seconds,
  out_of_sample_sharpe_ratio, out_of_sample_num_trades,
  out_of_sample_pnl, out_of_sample_sqn, out_of_sample_funding_pnl,
  out_of_sample_duration_seconds); every dict/list field persists as
  `<name>_json` TEXT (stats_json, positions_json, fills_json). Adding a
  metric field extends DDL, migration, insert and row decode automatically
  — no hand-written mirrors.
- `schema_meta(key PK, value)`

v1→v2 added jobs.timeout_seconds/attempts; v2→v3 derives results columns
from the dataclass and backfills any missing ones generically (guarded by
PRAGMA table_info, rerun-safe). Legacy v2 columns with no field
(`equity_curve_json`, `tearsheet_path`) survive on old databases but are
never written or read. v3→v4 replaced the `splits` dict with 12
first-class `in_sample_*` / `out_of_sample_*` scalar columns; legacy
`splits_json` survives on old databases but is never written.

Key methods: `save_job` upsert; `update_job_dispatch` (status+worker+
attempts); `complete_job` writes result + terminal status in ONE
transaction; `list_results(study_name=)` single JOIN query (no N+1).

Artifact spill (`core.runner._spill_artifacts`): when positions or fills
exceed `INLINE_ROW_BUDGET` (200 rows), write
`reports/artifacts/{job_id}/positions.parquet | fills.parquet` and carry
paths + counts in the result (inline lists stay empty); parquet failure
falls back to inline embedding. Anything crossing ZMQ/DB passes
`_jsonable_records` first (Timestamps → ISO strings, numpy scalars
unboxed) because json.dumps chokes on engine report types otherwise.

## 9. Optimizer

Entry: `optimize/study.py::run_optuna_study`. Objective modes:

- `sharpe` (default): 3-objective maximize (Sharpe 252d, num_trades, pnl);
  non-dominated filter in `_pareto_front`; report
  `generate_pareto_report`.
- `sqn`: single-objective maximize System Quality Number
  (`stats.system_quality_number` = sqrt(N) * mean/std(ddof=1) over closed
  positions' `realized_return`; None under 2 trades); report
  `generate_sqn_report`.

Executors (ask/tell loop over `study.ask()/study.tell()`):

- `LocalExecutor`: inline sequential runs; a crashed trial raises
  `optuna.TrialPruned` so the study never records (0,0,0).
- `SchedulerExecutor`: submits each trial as a scheduler job
  (`study_name` tags jobs), caps inflight at workers_total, polls
  `get_result` every 1s; FAILED/lost jobs → `TrialState.FAIL`.

Routing: `--local` forces local; otherwise ping `tcp://127.0.0.1:{port}`
— reachable → scheduler, else local fallback. Storage:
`sqlite:///{db_path}` (shared with ResultStore), TPESampler, study name
`opt_{strategy}_{objective}_{YYYYmmdd_HHMMSS}`, load_if_exists. Report
generation is skipped when no trial reached COMPLETE.

Param spec grammar (`optimize/param_parser.parse_param_spec`):
`name=int(min,max)` | `name=float(min,max)` | `name=cat(v1,v2,...)`;
`cat` auto-types true/false/int/float values. Default spaces exist for
overnight_drift / orb / glucksmann when no `--param` is given.
Overrides flow through `RunConfig.with_overrides(params)` →
`strategy_params`, which land on the strategy config via runner kwargs.

## 10. Data layer

Downloader (`python -m sbt.data`, `data.py`): ccxt with rate limiting;
unified `_paginate` generator — retries on NetworkError with exponential
backoff, halves page size (floor 50) when the exchange rejects the limit,
stops on empty page. Incremental resume by default: existing output file
is extended from its max timestamp (`--no-resume` refetches from --start,
`--page-limit` overrides rows/call; defaults 1000 ohlcv / 500 funding).
Output dedupes on timestamp and sorts before writing feather.
Naming goes through `core/feather.feather_path()`:
`data/{exchange}_{symbol_no_slash}_{tag}_YYYYMMDD[_YYYYMMDD].feather`
(*tag* = interval or `funding`; the end suffix comes from --end or
today). After writing, `actual_range_name()` renames conventional files
so their encoded range equals the actual min/max timestamps — a
conventional name always states what's inside (explicit `--output`
names not matching the convention are left alone).

Funding files are matched separately by the runner passing tag
`"funding"` to `find_feather`; funding data is metadata only (see §5).

## 11. Reporting

`sbt/report.py::print_report(engine, venue, title=..., pair=...,
exchange=..., interval=..., open_browser=True)` renders stats + positions
into `reports/*.html` and opens it unless suppressed
(`open_report: false` in TOML or CLI `--no-open`). The TradingView chart
block maps exchange → TV prefix (BINANCE:, BYBIT:, ...) and interval → TV
resolution; unknown exchanges omit the chart. With a train/val split,
`__main__.py` generates one tearsheet per window using
`runner.window_engines`. Optimizer reports:
`optimize/report.py` (Pareto 3D / SQN), also honoring open_report.

## 12. Invariants & gotchas (read before editing)

1. **BarDataWrangler is broken** on nautilus 1.230.0 ("buffer source array
   is read-only" for every input shape). Use `core.runner.load_bars`.
2. **msgspec kw_only trap**: every strategy config must declare
   `kw_only=True, frozen=True` on its subclass of the config tier — see §5.
   Missing it breaks struct construction at import/run time.
3. **tz-aware Timestamp rejection**: `pd.Timestamp(x, tz="UTC")` raises if
   x is already tz-aware. L2 loaders need plain date strings; normalize
   with `str(to_utc_ts(x))` (`core/feather.py`).
4. **Slippage has exactly one mechanism**: fee-bps added to taker fee
   (§4.2). Never reintroduce FillModel — it double-counts.
5. **Funding never flows through engine PnL**; it is a side-channel
   surfaced as `funding_pnl` (positive = paid).
6. **Worker code isolation**: `.worktrees/worker-N` run committed code.
   Re-seed after edits or commit before expecting workers to see changes.
7. **Client framing**: use DEALER `[b"", payload]` (`SbtClient`); REQ
   lockstep does not survive the ROUTER reply framing.
8. **Kill-before-requeue ordering** in ACK timeout handling prevents
   double-execution; preserve it when touching `_check_acks`.
9. **Split boundary bar belongs to OOS**; IS ends one bar interval early
   (`TrainValSplit.expand`). Warm-up bars load before window starts but
   cannot trade (`active_from` gate).
10. **Plugin params stay flat** on strategy configs; plugins declare
    `required_config_fields` and validation raises — do not add nested
    param objects or silent defaults.
11. **Results schema is fields-derived**: `BacktestResult` is the single
    source of truth (`core.job.result_field_specs`); never hand-write a
    column list, codec, or migration for it. Legacy v2 columns
    (`equity_curve_json`) may linger in old DBs — don't read them.
12. **JSON boundary discipline**: engine outputs must pass
    `_jsonable_records` before ZMQ/DB/JSON serialization.
13. **Deferred imports exist to break cycles** (§2 import-graph rule).
    `utils.interval_delta` inside `TrainValSplit.expand`,
    plugin registry lookup inside `PluginHost.from_config`.
14. Environment: Python >= 3.14, managed by uv; no tests/lint/CI exist —
    verify changes by running strategies end-to-end.
15. Gitignored: `data/`, `reports/` (incl. artifacts), `.worktrees/`,
    `sbt.db`.
