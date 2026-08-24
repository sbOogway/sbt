# ADR-0001: Strategy configs carry the runner-injected fields

Date: 2026-08-24
Status: Accepted

## Context

The runner builds every strategy config by injecting a fixed set of
kwargs — `instrument_id`, `capital`, `leverage`,
`backtest_start_date`, `active_from`, plus `bar_type` in bar mode.
Before this decision, each of the (then 14) strategy configs redeclared
those fields itself, so:

- Adding one runner-injected field meant editing the runner **and every
  config class**.
- The L2 execution branch survived on a runtime
  `__annotations__`/`hasattr` sniff whose only purpose was guessing
  whether the target config declared `bar_type`.
- Field defaults drifted between tiers (ohlc configs required
  `capital`; L2 configs defaulted it) despite nobody reading most of
  them at runtime.
- Unknown `strategy_params` keys from optimizer specs or server
  overrides landed silently as unused dict entries.

## Decision

1. `SBTStrategyConfig` (`sbt/plugins/base.py`) owns all runner-injected
   fields. Defaults: `capital=Decimal("1000")`, `leverage=1.0`,
   `backtest_start_date="2020-01-01"`, `active_from=None`;
   `instrument_id` stays required.
2. `SBTBarStrategyConfig(SBTStrategyConfig)` adds `bar_type` as a
   required field. Bar-driven strategies subclass this tier; L2
   strategies subclass `SBTStrategyConfig` directly and never receive
   `bar_type`. The mode↔config relationship is expressed in types, not
   sniffed at runtime.
3. All construction goes through `core.runner._build_strategy_config`,
   which raises (→ FAILED result) listing unknown keys plus the valid
   field names, mirroring `PluginHost.required_config_fields` strictness.
4. The msgspec per-subclass rule stands: every config subclass declares
   `(…, kw_only=True, frozen=True)` because msgspec does not inherit
   them.

## Consequences

- A new runner-injected field is a one-edit change to
  `SBTStrategyConfig`.
- The shape-sniffing hack in `_run_window`'s L2 branch is deleted; the
  bar/L2 branches are symmetric around one construction helper.
- Direct construction in tests needs only signal params (defaults cover
  the rest); `tests/test_strategy_configs.py` pins the contract for
  every registry entry so drift fails in CI, not at backtest time.
- Persisted jobs store `RunConfig` JSON only — strategy configs never
  serialize — so no migration was needed.
