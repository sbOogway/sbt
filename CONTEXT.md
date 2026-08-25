# SBT Context

Glossary for the domain language used across strategies, runner, and
plugins. Anchors are `file.py` + symbol names. Mechanics live in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); decisions in
[docs/adr/](docs/adr/).

## Glossary

- **Runner-injected fields** — the kwargs the runner supplies on every
  strategy-config construction (`instrument_id`, `capital`, `leverage`,
  `backtest_start_date`, `active_from`; plus `bar_type` for bar mode).
  They live once on `SBTStrategyConfig` (`plugins/base.py`); concrete
  strategies declare only signal parameters.
- **Bar-driven strategy** — consumes the hourly/daily bar stream;
  subclasses `SBTStrategy` and its config subclasses
  **SBTBarStrategyConfig** (the tier carrying `bar_type`). Lives in
  `sbt/strategies/ohlc/`.
- **L2 strategy** — order-book/event driven, no bar stream; plain
  nautilus `Strategy`, config subclasses `SBTStrategyConfig` directly
  (never receives `bar_type`). Lives in `sbt/strategies/l2/`.
- **Trading window / active_from** — bars before `active_from` warm up
  indicators and plugins but cannot produce orders; enforced by
  `SBTStrategy` gating.
- **Strategy plugin** — opt-in behaviour attached to a strategy via the
  flat `plugins` tuple; may contribute a sizing multiplier
  (`SizingPlugin.size_multiplier()`).
- **Runner plugin** — expands one job into named execution **Windows**
  and merges their results (`expand`/`combine`/`summarize`).
- **Window** — one execution slice `(label, start, end, df)`; `df` may
  include warm-up rows ahead of the trading start.
- **Feather file convention** — the naming contract for data files:
  `{exchange}_{symbol}_{tag}_{YYYYMMDD}[_{YYYYMMDD}].feather`, where
  *tag* is the bar interval (OHLCV) or `funding` (funding rates).
  Owned by `core/feather.py` (naming via `feather_path`, parsing via
  `parse_range`, discovery/ranking via `find_feather`); a conventional
  filename always states its contents' range — the downloader renames
  on resume (`actual_range_name`) to keep name == content.
- **BacktestResult** — structured output of a single backtest run;
  carries optimisation objectives (`sharpe_ratio`, `num_trades`, `pnl`,
  `sqn`), full engine stats, positions/fills, and funding PnL. Built
  by `_collect_result()` in `core/runner.py` after `engine.run()`.
  _Avoid_: run output, backtest output.
- **ResultStore** — thin SQLite wrapper (`core/db.py`) that persists
  `BacktestJob` and `BacktestResult` rows. Shared by the scheduler
  daemon (server path) and the runner (local/CLI path). The same
  database file doubles as Optuna storage.
- **Runner-level persistence** — the pattern where `BacktestRunner`
  owns DB writes via `ResultStore` at the end of `run()`, as opposed
  to server/scheduler-level persistence. Enabled by passing `db_path`
  to the runner constructor; omitted in the server path where the
  scheduler handles it.
