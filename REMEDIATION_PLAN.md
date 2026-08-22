# SBT Remediation Plan

Working checklist for the architectural remediation. Update statuses as phases
complete. Decisions locked in with the user (2026-08-21):

- **Sizing**: unify all strategies to compounding live account equity
  (historical `sbt.db` rows become non-comparable; accepted).
- **Slippage**: fee-bps only (`slippage_ticks * tick_size / ref_price` bps into
  taker fee); remove `FillModel` double-count.
- **Base class**: introduce shared `SBTStrategy` base (equity sizing, window
  gating, funding tracker) and rewrite all 5 strategies onto it.
- **`.kilo` worktree**: remove it.

---

## Phase 1 — Correctness

### 1.1 Shared strategy base class  `[x]`
New `sbt/strategies/base.py`:
- `SBTStrategy(Strategy)`:
  - `equity() -> float` — live account balance via `self.cache.account_for_venue(...)`
  - `sized_quantity(notional, price) -> Quantity | None`
  - Window gating: base `on_bar()` drops bars before an `active_from` timestamp;
    subclasses implement `on_trading_bar(bar)`
  - `FundingTracker` instance as `self.funding` (replaces `_trade_funding_costs`)
- Rewrite `overnight_drift`, `orb`, `key_breakout`, `glucksmann`,
  `bitcoin_intraday_momentum` to subclass it. Each keeps its own risk-formula
  shape but swaps `float(self.config.capital)` -> `self.equity()` (compounding).
- `bitcoin_intraday_momentum` gains a `risk_percent` field for uniform semantics.
- Delete duplicated `_calc_qty` / `_open_trade` / `_close_trade` copies.
- Runner reads typed `strategy.funding` instead of `hasattr(strategy, "_trade_funding_costs")`.

### 1.2 Slippage: single mechanism  `[x]`
`sbt/core/runner.py`: delete both `FillModel(prob_slippage=1.0)` branches
(L2 + bar paths). Keep bps addition to taker fee. Update AGENTS.md note.

### 1.3 Train/val split boundaries + warm-up  `[x]`
- In-sample window ends at `split_ts - one_bar_interval` (no boundary dupe).
- New `RunConfig.warmup_bars: int | None`; windows load data from
  `window_start - warmup_bars * interval` while trading gates at true start
  (`active_from` from 1.1).
- Remove silent `"1-MINUTE"` L2 fallback (runner.py ~380): fail loudly.

### 1.4 Strict plugin-parameter validation  `[x]`
`plugins/base.py` + `vol_scaling.py`: plugins declare
`required_config_fields: ClassVar[...]`; `PluginHost.from_config()` validates
against `dataclasses.fields(config)` and raises listing missing names.

### 1.5 Funding promoted, chart fixed  `[x]`
- `funding_pnl` added to per-window `splits`, compare dashboard, CLI output.
- `report.py`: exchange->TV prefix map (BINANCE:/BYBIT:; unknown -> omit chart),
  interval->TV resolution map; `print_report()` takes exchange/interval args.

**Verify** `[x] 2026-08-21`: all 5 strategies ran end-to-end on hyperliquid BTC 1h
(compounding visible in account report); `--train-val-split 0.7 --warmup-bars 50`
produced both windows with funding_pnl; taker fee shows exactly one slippage
increment (`slippage` column 0.0 — FillModel gone). Unit checks: config
round-trip, plugin validation raise/pass, FundingTracker conventions,
active_from parsing.

---

## Phase 2 — Scheduler durability, optimizer routing, payload slimming

### 2.1 Durable scheduler  `[x]`
(`scheduler.py`, `worker.py`, `job.py`)
- Startup reconciliation: DB scan pending/running; stale running -> pending;
  enqueue ordered by submitted_at.
- ACK protocol: worker ACKs job receipt; no ACK in 30 s -> kill + requeue
  (kill happens BEFORE requeue so a late-starting worker can't double-run).
- Reaper: heartbeat PINGs to busy workers; 3 missed pongs -> kill+respawn
  worker, requeue its job; `BacktestJob.timeout_seconds` (default 3600).
- Child-process liveness reap/respawn of exited Popen handles.
- `BacktestJob.attempts` persisted; MAX_ATTEMPTS (env SBT_MAX_ATTEMPTS, 2)
  then FAILED. Knobs: SBT_ACK_TIMEOUT / SBT_HEARTBEAT_INTERVAL /
  SBT_MAX_MISSED_PONGS. Late/duplicate RESULTs dropped via owner check.

### 2.2 Optimizer through scheduler  `[x]`
(`optimize/study.py`)
- ask/tell loop: `SchedulerExecutor` (submit per trial, inflight <= worker
  pool, poll get_result) and `LocalExecutor`. Auto-route: ping scheduler on
  --port, else local; `--local` forces inline. Failed runs ->
  TrialState.FAIL (local exceptions raise optuna.TrialPruned) — no more
  (0,0,0) rows; report generation skips when no trial completed.
- FIXED latent bug: client<->scheduler REQ framing never worked (REQ's empty
  delimiter frame broke json parsing server-side); SbtClient now uses DEALER
  with explicit framing.

### 2.3 Result slimming + DB hardening  `[x]`
- Positions/fills > 200 rows (`INLINE_ROW_BUDGET`) ->
  `reports/artifacts/{job_id}/*.parquet`; result carries paths + counts;
  small runs stay inline; parquet failure falls back to inline.
- Engine report records sanitized to JSON-safe types (Timestamp/np scalars)
  at the runner — this crash was latent until results traveled over ZMQ.
- SQLite WAL + busy_timeout=5000; `schema_meta` version table; v2 migration
  adds jobs.timeout_seconds/attempts + results artifact columns (idempotent).

**Verify** `[x] 2026-08-22`: kill -9 scheduler mid-batch -> restart recovered
2 unfinished jobs, all completed, interrupted job attempts=2; kill -9 worker
mid-job -> proc-liveness requeue + respawn, done exactly once (attempts=2);
optimize auto-routed through scheduler (4 SQN trials across 2 workers);
spill/migration/reconcile unit-checked.

---

## Phase 3 — Honest abstractions

### 3.2 Real worktree isolation  `[x]`  (do FIRST — tiny)
Spawn workers with `cwd=worktree_path` (+ PYTHONPATH) so `-m sbt.server.worker`
imports the worktree's code. mkdir fallback warns loudly about degraded isolation.
**Verify** `[x]`: edited log string in worktree copy appeared in worker logs;
`sbt.__file__` resolved inside worktree; fallback dir reports isolated=False.

### 3.1 Honest RunnerPlugin  `[x]`
Fix `RunnerPlugin.combine(job_id, results, windows)` signature mismatch; add
`expand(cfg) -> dict[name, Window]`; register TrainValSplit in a runner-plugin
registry; runner dispatch driven by flat `train_val_split: float` field.
**Verify** `[x] 2026-08-22`: expand unit checks (boundary bar math, warmup
grid quantization, L2 missing-end raise, registry lookup); end-to-end split
run identical to pre-refactor (IS 0->18:30, OOS boundary-bar start, summary
table, splits payload). Runner now only loads data + executes windows;
window math lives in the plugin (`Window` NamedTuple carries label/bounds/
pre-sliced df).

---

## Phase 4 — Efficiency & hygiene

| Item | Change | Status |
|---|---|---|
| Bar ingestion | BarDataWrangler unusable (nautilus 1.230.0 raises "buffer source array is read-only" for every input shape) -> array-based loader: int64-nanos via `as_unit("ns")` + numpy columns, no per-row pandas boxing. 1.17x at 9k bars (Bar ctor dominates). | `[x]` |
| N+1 queries | `ResultStore.list_results(study_name=None)` single query replaces per-job get_result loop in scheduler; `complete_job()` persists result + terminal status in one transaction. | `[x]` |
| Config duplication | to_dict/from_dict derived from `dataclasses.fields` + type hints (`_coerce` handles Decimal/int/float/bool/Optional); from_toml delegates to the codec (TOML keys map 1:1 to fields); overrides stay `dataclasses.replace`. Added `open_report` field. | `[x]` |
| Downloader | Unified `_paginate` generator (retries, no-progress guard, adaptive halving when exchange rejects limit); incremental resume from feather max ts by default (`--no-resume` forces refetch); `--page-limit` override. | `[x]` |
| find_feather | Exchange-prefixed matches preferred; bare files used only when unique; multiple prefixed matches ranked by range coverage > overlap > newest; chosen file printed. | `[x]` |
| Hygiene | orb.py.bak deleted; .kilo worktree removed + `.kilo/` gitignored; stats.RunConfig -> RunConfigStatistic; equity_curve dropped from result model/ZMQ/DB writes (legacy column untouched); tearsheet browser open gated by `open_report` / CLI `--no-open`. | `[x]` |

**Verify** `[x] 2026-08-22`: server round-trip (submit -> done, real metrics,
single-query listing); optimizer through scheduler 3/3 trials COMPLETE with
clean exit (report browser-open now honors open_report / client --no-open);
local single + split runs exit 0; find_feather/config/resume unit checks pass.
Incident resolved: repo `data/hyperliquid_BTCUSDC:USDC_1h_*.feather` had been
overwritten by the durability harness's synthetic 600-bar file — re-downloaded
(4600 rows, symbol history starts 2025-12-13; end-to-end run on restored data
verified). Note: `.worktrees/worker-*` held stale code and crashed workers
until re-seeded — seeding requirement now documented in AGENTS.md.

---

## Test suite  `[ ]`
pytest as dev dep.
- Unit: SQN, vol-scaling RV/monthly, param parser round-trips, find_feather,
  config serialize round-trip, DB round-trip+migration, split boundaries,
  plugin-validation failures.
- Integration: synthetic ~500-bar feather -> end-to-end run of 2 strategies,
  deterministic trade counts; slippage fee assertion.
- Server: ephemeral ports tmpdir scheduler+worker; batch submit; kill-worker requeue.

AGENTS.md updated per phase.

**Execution order**: 1 -> 3.2 -> 2 -> 3.1 -> 4. Tests woven in after Phase 1.
