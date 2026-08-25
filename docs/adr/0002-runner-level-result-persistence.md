# ADR-0002: Runner-level result persistence

BacktestRunner persists BacktestResult to the results table via
ResultStore at the end of `run()`, enabled by an explicit `db_path`
constructor parameter. This ensures every backtest execution —
CLI, local optimizer, and tests — writes results to the database
without relying on the server scheduler.

`db_path` lives on the runner constructor, not RunConfig, because it
is an infrastructure concern (where to store output) rather than a
backtest execution parameter (what to run). RunConfig serializes to
JSON in the jobs table; embedding db_path there would store the
"write to DB" instruction inside the DB itself.

In the server path, the worker omits `db_path` (defaults to None),
so the runner does not persist. The scheduler's existing
`complete_job()` handles persistence, avoiding cross-worktree
file conflicts.
