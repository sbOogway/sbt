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
        # Per-leg (ts_ns, close, volume). Volume is required for the
        # "volume" factor and harmless for "momentum".
        self._series: dict[InstrumentId, list[tuple[int, float, float]]] = {
            iid: [] for iid in self._legs
        }
        self._last_week: str | None = None

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        volume = (
            float(bar.volume.as_double()) if hasattr(bar, "volume") else 0.0
        )
        if volume > 0:
            self._volume_seen = True
        self._series[instrument_id].append(
            (
                bar.ts_event,
                float(bar.close.as_double()),
                volume,
            )
        )

        # Rebalance once per ISO week, keyed on the primary leg so a
        # basket-wide decision runs exactly once per week boundary.
        if instrument_id != self._primary_iid:
            return
        dt = self._ts(bar)
        week = dt.strftime("%G-%V")  # ISO year-week
        if week == self._last_week or not self.trading_active:
            return
        self._last_week = week
        self._rebalance(dt)

    @staticmethod
    def _ts(bar: Bar) -> pd.Timestamp:
        return pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")

    def _formation_bounds(self, ts: pd.Timestamp) -> tuple[int, int]:
        # The week just ended, and `lookback_weeks` further back; both as
        # nanosecond bucket edges so per-leg closes are sampled at/before them.
        cur = ts.to_period("W").to_timestamp().tz_localize("UTC")
        end = int((cur - pd.Timedelta(days=1)).normalize().value)
        start = int(
            (
                cur
                - pd.DateOffset(weeks=self.config.lookback_weeks)
                - pd.Timedelta(days=1)
            )
            .normalize()
            .value
        )
        return end, start

    @staticmethod
    def _close_at_or_before(pairs: list[tuple[int, float, float]], ns: int) -> float | None:
        """Close of the last bar at or before *ns* (pairs sorted ascending)."""
        close: float | None = None
        for t, c, _v in pairs:
            if t > ns:
                break
            close = c
        return close

    def _factor_values(self, ts: pd.Timestamp) -> dict[InstrumentId, float]:
        if self.config.factor == "volume" and not self._volume_seen:
            raise ValueError(
                "factor='volume' requires bars with non-zero volume data; "
                "none seen so far. Check the data feed."
            )
        end_ns, start_ns = self._formation_bounds(ts)
        factor = self.config.factor
        values: dict[InstrumentId, float] = {}
        for iid, pairs in self._series.items():
            c_end = self._close_at_or_before(pairs, end_ns)
            c_start = self._close_at_or_before(pairs, start_ns)
            if c_end is None or c_start is None or c_start <= 0:
                continue
            if factor == "volume":
                # Dollar volume over the window (sum vol*close, per bar).
                values[iid] = sum(
                    v * c
                    for t, c, v in pairs
                    if start_ns < t <= end_ns
                )
            else:  # momentum
                values[iid] = c_end / c_start - 1.0
        return values

    def _rebalance(self, ts: pd.Timestamp) -> None:
        values = self._factor_values(ts)
        if not values:
            return
        ordered = sorted(values, key=lambda iid: values[iid])
        n = len(ordered)
        n_sel = max(1, round(n * self.config.top_fraction))
        losers: set[InstrumentId] = set(ordered[:n_sel])
        winners: set[InstrumentId] = set(ordered[max(0, n - n_sel) :])

        for iid in list(self._legs):
            leg = self._leg(iid)
            target: OrderSide | None = (
                OrderSide.BUY if iid in winners else OrderSide.SELL if iid in losers else None
            )
            if leg.side is None:
                if target is not None and leg.price:
                    self.open_position(target, leg.price, iid)
            elif target is None:
                self.exit_market(iid)
            elif leg.side != target:
                self.exit_market(iid)
                if leg.price:
                    self.open_position(target, leg.price, iid)
