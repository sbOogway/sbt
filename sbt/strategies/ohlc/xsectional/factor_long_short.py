"""Cross-sectional factor long-short following Liu, Tsyvinski & Wu (2019).

"Common Risk Factors in Cryptocurrency" (NBER w25882). Each week the basket
is sorted cross-sectionally into quintiles on a factor value and a zero-cost
long-short portfolio (top quintile long, bottom quintile short) is held for
the following week, then re-sorted — the paper's Q5-Q1 winner/loser spread.

This mirrors the rank-sort machinery of ``XSectionalMomentum`` (the canonical
portfolio template) but rebalances on a weekly boundary (not monthly) and
sorts on a configurable factor. ``factor="momentum"`` ranks by the trailing
``lookback_weeks`` return; ``factor="volume"`` ranks by trailing dollar
volume. Size/market-cap is ***not*** available from OHLCV-only feathers, so
size is left unset rather than approximated from price.
"""

import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ....plugins import SBTPortfolioStrategyConfig
from ...base import SBTPortfolioStrategy


class FactorLongShortConfig(SBTPortfolioStrategyConfig, kw_only=True, frozen=True):
    factor: str = "momentum"  # "momentum" | "volume"
    lookback_weeks: int = 3   # paper's best momentum sort (3-week ~4.1%/wk)
    top_fraction: float = 0.2  # quintile tails (Q5-Q1 => 20%): long/short each


class FactorLongShort(SBTPortfolioStrategy):
    _FACTORS = ("momentum", "volume")

    def __init__(self, config: FactorLongShortConfig) -> None:
        super().__init__(config)
        if config.factor not in self._FACTORS:
            raise ValueError(
                f"factor must be one of {self._FACTORS}, got {config.factor!r}"
            )
        self._volume_seen: bool = False
        self._last_week: str | None = None

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        if (
            bar.volume is not None
            and float(bar.volume.as_double()) > 0
        ):
            self._volume_seen = True

        # Rebalance once per ISO week, keyed on the primary leg so a
        # basket-wide decision runs exactly once per week boundary.
        if instrument_id != self._primary_iid:
            return
        dt = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        week = dt.strftime("%G-%V")  # ISO year-week
        if week == self._last_week or not self.trading_active:
            return
        self._last_week = week
        self._rebalance(dt)

    def _formation_bounds(self, ts: pd.Timestamp) -> tuple[int, int]:
        # The week just ended, and `lookback_weeks` further back; both as
        # nanosecond bucket edges so per-leg closes are sampled at/before them.
        cur = ts.to_period("W").to_timestamp().tz_localize("UTC")
        end_ns = int((cur - pd.Timedelta(days=1)).normalize().value)
        start_ns = int(
            (
                cur
                - pd.DateOffset(weeks=self.config.lookback_weeks)
                - pd.Timedelta(days=1)
            )
            .normalize()
            .value
        )
        return end_ns, start_ns

    def _factor_values(self, ts: pd.Timestamp) -> dict[InstrumentId, float]:
        if self.config.factor == "volume" and not self._volume_seen:
            raise ValueError(
                "factor='volume' requires bars with non-zero volume data; "
                "none seen so far. Check the data feed."
            )
        end_ns, start_ns = self._formation_bounds(ts)
        if self.config.factor == "volume":
            return self.history.dollar_volumes(start_ns, end_ns)
        return self.history.formation_returns(end_ns, start_ns)

    def _rebalance(self, ts: pd.Timestamp) -> None:
        values = self._factor_values(ts)
        if not values:
            return
        ordered = sorted(values, key=lambda iid: values[iid])
        n = len(ordered)
        n_sel = max(1, round(n * self.config.top_fraction))
        losers: set[InstrumentId] = set(ordered[:n_sel])
        winners: set[InstrumentId] = set(ordered[max(0, n - n_sel) :])

        targets: dict[InstrumentId, OrderSide | None] = {
            iid: (OrderSide.BUY if iid in winners
                  else OrderSide.SELL if iid in losers
                  else None)
            for iid in self._legs
        }
        self.apply_targets(targets)
        self.prune_history_weeks(self.config.lookback_weeks + 1)
