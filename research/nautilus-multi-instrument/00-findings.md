# Wayfinder — Multi-instrument portfolio engine (map #24)

## Ticket resolution: Nautilus multi-instrument shared-account mechanics (#25)

Research resolved on the installed nautilus-trader **1.230.0** at
`.venv/lib64/python3.14/site-packages/nautilus_trader` (primary source),
confirmed at runtime by a probe (`/tmp/opencode/probe_multi.py`: one
BacktestEngine, venue `HL` NETTING/MARGIN/USDC, 3 CryptoPerpetual
instruments, 4 hourly external bars each at identical ts_init, one
strategy subscribed to all 3 bar types).

### 1. One engine takes N instruments — YES
`BacktestEngine.add_instrument` (backtest/engine.pyx:733-782) appends;
requires venue registered first and no CASH CurrencyPair restriction.
`SimulatedExchange` keeps instruments in an OrderedDict by id and creates
ONE matching engine per instrument. No count limit.

### 2. N interleaved bar streams — YES
`add_data` (backtest/engine.pyx:783+) appends and re-sorts stably by
`ts_init` (:903) into a single named stream; `run()` rejects unsorted.
Merge heap key `(ts_init, data_priority, cursor)` (:2585-2590). Equal-ts
items keep insertion order = **add_data call order + list order** (the
tie-break). One `BacktestMarketDataClient` per venue dispatches all.

### 3. Shared account — YES (one account per venue)
`BacktestExecClient` creates the single account
`AccountId(f"{exchange.id}-001")` (backtest/execution_client.pyx:81);
`Cache.account_for_venue` returns it (cache/cache.pyx:4108-4142).
NETTING positions are per-instrument
(position_id `{instrument}-{strategy_id}`, engine.pyx:5158); margins
tracked per instrument inside one MarginAccount, aggregated per currency.
**`Account.balance_total` (accounting/accounts/base.pyx:226) is the
wallet aggregate across all legs** — the correct whole-portfolio capital
base for `SBTStrategy.equity()`. Caveat: it excludes unrealized PnL (sits
per-position in the portfolio), so use for the capital base, not per-leg
headers inside the loop.

### 4. One strategy, N bar types — YES, with a GOTCHA
`Strategy.subscribe_bars` one call per BarType (common/actor.pyx:1907-
1952), publish per-topic with per-bar-type sequencing guard
(data/engine.pyx:2790-2800,2818), `handle_bar` dispatches to `on_bar`
per `bar.bar_type.id_spec_key()` (actor.pyx:4673-4703,4692).
**GOTCHA: `Bar` has NO `.instrument_id` attribute in 1.230.0** — read it
as `bar.bar_type.instrument_id` (model/data.pyx:1307, Bar.bar_type :1630).

### 5. Aggregation for reports/stats — YES
`PortfolioAnalyzer` keys realized/unrealized PnL by instrument+account but
aggregates over all (analysis/analyzer.py:195,338, get_performance_stats
:562,:609,:703). `trader.generate_positions_report/fills_report` dump all
cache positions/orders (trading/trader.py:876-889/865-874); rows carry
`instrument_id`. Analyzer stats reflect the whole book.

### 6. Equity/sizing — mapping confirmed
Existing `SBTStrategy.equity()` (sbt/strategies/base.py:137-143) maps
directly: `cache.account_for_venue(venue).balance_total()`.

### 7. Memory leak — UNVERIFIED
The "second BacktestEngine in-process SIGABRTs" claim (map #20 Notes) is
external; not confirmed from source here. Nothing pathological observed on
the small run. NOT a blocker for one-engine-many-instruments.

### Extra gotchas
- Each `add_data` call must be one homogeneous batch (one instrument's
  bars) — docstring warns of same-type assumption.
- Fill/event ordering at equal timestamps is insertion-order-sensitive; a
  COIN0 bar's fill was visible to a later-inserted COIN1 bar at the same
  timestamp.
- Bar `volume` precision must match the instrument's `size_precision` or
  the matching engine raises (engine.pyx:4775, "invalid
  bar.volume.precision").

### Recommended refactor shape for sbt
One engine/venue/account → `add_instrument(pp)` per instrument → one
`add_data(bars_i)` per instrument (insertion order = tie-break) → one
strategy instance calling `subscribe_bars` per instrument, routing each
`on_bar` via `bar.bar_type.instrument_id`, sizing off
`account_for_venue(...).balance_total()`. Fully supported by the installed
engine, zero multi-engine round-trips.

## What's downstream
This research feeds the runner-seam ticket and the per-instrument
SBTStrategy ticket (both blocked by this). Map: #24. Related tickets:
#26 (runner seam, prototype), #27 (per-instrument strategy model,
grilling), #28 (result semantics, grilling).
