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
