"""Cross-sectional momentum following Jegadeesh & Titman (1993).

A true long-short portfolio strategy running on the multi-instrument engine:
one shared margin account, one leg per basket symbol. At each month start it
ranks the basket by trailing ``formation_months`` price return, then opens
equal-weight long legs on the winners and short legs on the losers (market
neutral by construction), flattening any leg whose rank no longer qualifies.
"""

import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ....plugins import SBTPortfolioStrategyConfig
from ...base import SBTPortfolioStrategy


class XSectionalMomentumConfig(SBTPortfolioStrategyConfig, kw_only=True, frozen=True):
    formation_months: int = 6
    top_fraction: float = 1.0 / 3.0


class XSectionalMomentum(SBTPortfolioStrategy):
    def __init__(self, config: XSectionalMomentumConfig) -> None:
        super().__init__(config)
        # Per-leg (ts_ns, close) pairs, one append per forwarded bar.
        self._series: dict[InstrumentId, list[tuple[int, float]]] = {
            iid: [] for iid in self._legs
        }
        self._last_month: str | None = None

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        self._series[instrument_id].append(
            (bar.ts_event, float(bar.close.as_double()))
        )

        # Rebalance once per calendar month, keyed on the primary leg so a
        # basket-wide decision runs exactly once per month boundary.
        if instrument_id != self._primary_iid:
            return
        dt = self._ts(bar)
        month = dt.strftime("%Y-%m")
        if month == self._last_month or not self.trading_active:
            return
        self._last_month = month
        self._rebalance(dt)

    @staticmethod
    def _ts(bar: Bar) -> pd.Timestamp:
        return pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")

    def _formation_bounds(self, ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        cur_month_start = ts.to_period("M").to_timestamp().tz_localize("UTC")
        end_bucket = (cur_month_start - pd.Timedelta(days=1)).normalize()
        start_bucket = (
            cur_month_start
            - pd.DateOffset(months=self.config.formation_months)
            - pd.Timedelta(days=1)
        ).normalize()
        return end_bucket, start_bucket

    @staticmethod
    def _close_at_or_before(pairs: list[tuple[int, float]], ns: int) -> float | None:
        """Close of the last bar at or before *ns* (pairs sorted ascending)."""
        val: float | None = None
        for t, c in pairs:
            if t > ns:
                break
            val = c
        return val

    def _momentum_returns(self, ts: pd.Timestamp) -> dict[InstrumentId, float]:
        end_bucket, start_bucket = self._formation_bounds(ts)
        end_ns = int(end_bucket.value)
        start_ns = int(start_bucket.value)
        rets: dict[InstrumentId, float] = {}
        for iid, pairs in self._series.items():
            c_end = self._close_at_or_before(pairs, end_ns)
            c_start = self._close_at_or_before(pairs, start_ns)
            if c_end is None or c_start is None or c_start == 0:
                continue
            rets[iid] = c_end / c_start - 1.0
        return rets

    def _rebalance(self, ts: pd.Timestamp) -> None:
        rets = self._momentum_returns(ts)
        if not rets:
            return
        ordered = sorted(rets, key=lambda iid: rets[iid])
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
