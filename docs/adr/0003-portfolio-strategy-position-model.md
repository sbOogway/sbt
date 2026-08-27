# ADR-0003: Portfolio strategies live in a separate subclass, not a reworked base

Date: 2026-08-27
Status: Accepted

## Context

The engine must support multi-instrument portfolio strategies (map #24) that
co-manage one position per instrument ("leg") on a single shared margin
account. Today `SBTStrategy` tracks exactly one position via scalar
`position_side` / `_open_qty` and a single `FundingTracker`, and roughly
seventy bar strategies read `self.position_side` directly as an attribute in
their conditions. Backward compatibility for those single-instrument
strategies is binding.

## Decision

`SBTStrategy` stays a single-position base, unchanged. A new
`SBTPortfolioStrategy` subclasses it and owns per-leg state in
`_legs: dict[InstrumentId, LegState]` (side + quantity + `FundingTracker` +
latest price). It overrides `open_position`/`exit_market`/`submit_market` with
an `instrument_id` keyword defaulting to the injected primary `instrument_id`
(so the single-arg form still works), exposes `leg_*` accessors and
`position_map` (deliberately not overloading the single-position
`position_side`/`in_position`), routes bars via a per-leg `on_instrument_bar`
hook, routes funding per-leg by `funding_rate.instrument_id`, and exposes a
synthetic aggregate `funding` tracker whose `total_paid` sums all legs so the
runner's funding collection path is unchanged.

## Considered Options

- **Rework the base to a shared per-instrument primitive.** Rejected: would
  churn the `position_side` attribute contract that ~70 strategies compile
  against, at high risk for no benefit to single-instrument strategies.
- **Overload `position_side`/`in_position` on the portfolio subclass.**
  Rejected: ambiguous when a long+short basket holds several legs at once.

## Consequences

- Single-instrument strategies compile and run unchanged (they never touch the
  portfolio subclass).
- Portfolio strategies are self-contained in one subclass; per-leg sizing uses
  whole-portfolio `equity()` divided by the selected-basket size via a
  `leg_quantity(price, n_legs)` helper.
- `_collect_result` reads `strategy.funding.total_paid` unchanged for both
  modes (portfolio `funding` is the synthetic sum).
