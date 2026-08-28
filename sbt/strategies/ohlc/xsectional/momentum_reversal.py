"""Crypto momentum vs reversal, Dobrynskaya (2023), J/K winner-minus-loser.

Each week the basket is sorted by its trailing cumulative return over **J**
weeks; the top ``top_bottom`` fraction are the *winners* (long) and the bottom
fraction the *losers* (short) — a zero-cost winner-minus-loser (WML) portfolio
held for **K** weeks. Short-horizon J,K ~2-4 weeks show positive momentum; the
signal flips to **reversal** at longer horizons (past losers drive it).

The paper's overlapping 1/K tranche schedule (weekly rebalancing with holding
> 1) is ***not*** expressed here: the engine keeps a single equal-weight
position per leg, so each leg would need K partial tranches. Instead this
implements the paper's non-overlapping variant — re-form the whole WML book
once every ``holding_weeks``. Equal-weight per-leg sizing via ``leg_quantity``;
the paper's capitalization-weighting is a data-rich refinement the curated
basket doesn't support.
"""

import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ....plugins import SBTPortfolioStrategyConfig
from ...base import SBTPortfolioStrategy

_WEEK_NS = 7 * 86_400_000_000_000


class MomentumReversalConfig(SBTPortfolioStrategyConfig, kw_only=True, frozen=True):
    formation_weeks: int = 2   # J: trailing-return sort window (2/2 sweet spot)
    holding_weeks: int = 2     # K: holding period between WML re-formations
    top_bottom: float = 0.30   # top/bottom fraction treated as winners/losers


class MomentumReversal(SBTPortfolioStrategy):
    def __init__(self, config: MomentumReversalConfig) -> None:
        super().__init__(config)
        if not 0 < config.top_bottom < 0.5:
            raise ValueError(
                f"top_bottom must be in (0, 0.5), got {config.top_bottom}"
            )
        if config.formation_weeks < 1 or config.holding_weeks < 1:
            raise ValueError("formation_weeks and holding_weeks must be >= 1")
        # Per-leg (ts_ns, close).
        self._series: dict[InstrumentId, list[tuple[int, float]]] = {
            iid: [] for iid in self._legs
        }
        self._formed_ns: int | None = None

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        self._series[instrument_id].append(
            (bar.ts_event, float(bar.close.as_double()))
        )

        # Re-form the WML book once per holding period, keyed on the primary
        # leg so a basket-wide decision runs exactly once per boundary.
        if instrument_id != self._primary_iid:
            return
        dt = self._ts(bar)
        if not self.trading_active:
            return
        if self._formed_ns is not None:
            if dt.value - self._formed_ns < self.config.holding_weeks * _WEEK_NS:
                return
        self._formed_ns = dt.value
        self._rebalance(dt)

    @staticmethod
    def _ts(bar: Bar) -> pd.Timestamp:
        return pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")

    def _formation_bounds(self, ts: pd.Timestamp) -> tuple[int, int]:
        cur = ts.to_period("W").to_timestamp().tz_localize("UTC")
        end = int((cur - pd.Timedelta(days=1)).normalize().value)
        start = int(
            (
                cur
                - pd.DateOffset(weeks=self.config.formation_weeks)
                - pd.Timedelta(days=1)
            )
            .normalize()
            .value
        )
        return end, start

    @staticmethod
    def _close_at_or_before(pairs: list[tuple[int, float]], ns: int) -> float | None:
        close: float | None = None
        for t, c in pairs:
            if t > ns:
                break
            close = c
        return close

    def _formation_returns(self, ts: pd.Timestamp) -> dict[InstrumentId, float]:
        end_ns, start_ns = self._formation_bounds(ts)
        rets: dict[InstrumentId, float] = {}
        for iid, pairs in self._series.items():
            c_end = self._close_at_or_before(pairs, end_ns)
            c_start = self._close_at_or_before(pairs, start_ns)
            if c_end is None or c_start is None or c_start <= 0:
                continue
            rets[iid] = c_end / c_start - 1.0
        return rets

    def _rebalance(self, ts: pd.Timestamp) -> None:
        rets = self._formation_returns(ts)
        if not rets:
            return
        ordered = sorted(rets, key=lambda iid: rets[iid])
        n = len(ordered)
        n_sel = max(1, round(n * self.config.top_bottom))
        losers: set[InstrumentId] = set(ordered[:n_sel])
        winners: set[InstrumentId] = set(ordered[max(0, n - n_sel):])

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
