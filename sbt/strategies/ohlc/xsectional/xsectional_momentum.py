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
        self._last_month: str | None = None

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        # Rebalance once per calendar month, keyed on the primary leg so a
        # basket-wide decision runs exactly once per month boundary.
        if instrument_id != self._primary_iid:
            return
        dt = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        month = dt.strftime("%Y-%m")
        if month == self._last_month or not self.trading_active:
            return
        self._last_month = month
        self._rebalance(dt)

    def _formation_bounds(self, ts: pd.Timestamp) -> tuple[int, int]:
        cur_month_start = ts.to_period("M").to_timestamp().tz_localize("UTC")
        end_ns = int((cur_month_start - pd.Timedelta(days=1)).normalize().value)
        start_ns = int(
            (
                cur_month_start
                - pd.DateOffset(months=self.config.formation_months)
                - pd.Timedelta(days=1)
            )
            .normalize()
            .value
        )
        return end_ns, start_ns

    def _rebalance(self, ts: pd.Timestamp) -> None:
        end_ns, start_ns = self._formation_bounds(ts)
        rets = self.history.formation_returns(end_ns, start_ns)
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
        self.prune_history_weeks(4 * (self.config.formation_months + 1))
