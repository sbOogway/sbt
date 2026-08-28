"""Short-term daily reversal vs momentum, Zaremba et al. (2021).

"Up or down? Short-term reversal, momentum, and liquidity effects in
cryptocurrencies" (IRFA). Each UTC day the basket is sorted cross-sectionally
on the **lagged 1-day log-return** (LRET) and a quintile long-short is built:
reversal longs the worst performers and shorts the best (last day's losers
rebound), held one day and re-sorted daily.

The paper's core finding is the liquidity dependence: the daily reversal is
driven by the small, illiquid majority, while the largest / most liquid coins
show daily **momentum** instead. ``liquidity_top_quantile`` restricts the
universe to the top fraction of symbols by trailing dollar volume (the liquid
subset), which the paper associates with momentum — combine with
``reverse=False`` to trade that side of the book. Dollar volume is used as
the liquidity proxy because OHLCV feathers carry volume but no market cap.
"""

from math import log
import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ....plugins import SBTPortfolioStrategyConfig
from ...base import SBTPortfolioStrategy


class ZarembaReversalConfig(SBTPortfolioStrategyConfig, kw_only=True, frozen=True):
    reverse: bool = True  # True=reversal (long losers/short winners of LRET)
    liquidity_top_quantile: float | None = None  # subset to liquid coins by $vol
    liquidity_window_days: int = 5  # trailing days for the dollar-volume proxy
    top_fraction: float = 0.2  # quintile tails (Q1/Q5) taken on each side


class ZarembaReversal(SBTPortfolioStrategy):
    def __init__(self, config: ZarembaReversalConfig) -> None:
        super().__init__(config)
        if config.top_fraction <= 0 or config.top_fraction > 0.5:
            raise ValueError(f"top_fraction must be in (0, 0.5], got {config.top_fraction}")
        if config.liquidity_top_quantile is not None and not 0 < config.liquidity_top_quantile <= 1:
            raise ValueError(
                "liquidity_top_quantile must be in (0, 1] or None, "
                f"got {config.liquidity_top_quantile}"
            )
        # Per-leg (ts_ns, close, volume).
        self._series: dict[InstrumentId, list[tuple[int, float, float]]] = {
            iid: [] for iid in self._legs
        }
        self._last_day: str | None = None

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        self._series[instrument_id].append(
            (
                bar.ts_event,
                float(bar.close.as_double()),
                float(bar.volume.as_double() if hasattr(bar, "volume") else 0.0),
            )
        )

        # Rebalance once per UTC day, keyed on the primary leg.
        if instrument_id != self._primary_iid:
            return
        dt = self._ts(bar)
        day = dt.strftime("%Y-%m-%d")
        if day == self._last_day or not self.trading_active:
            return
        self._last_day = day
        self._rebalance(dt)

    @staticmethod
    def _ts(bar: Bar) -> pd.Timestamp:
        return pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")

    def _buckets(self, ts: pd.Timestamp) -> tuple[int, int]:
        # Bucket edges as integer ns: yesterday's close and the day before it.
        today = ts.normalize().value
        end_ns = today - 1  # last bar before today's midnight
        start_ns = end_ns - 86_400_000_000_000  # one UTC day earlier
        return end_ns, start_ns

    @staticmethod
    def _close_at_or_before(pairs: list[tuple[int, float, float]], ns: int) -> float | None:
        close: float | None = None
        for t, c, _v in pairs:
            if t > ns:
                break
            close = c
        return close

    def _dollar_volume(self, pairs: list[tuple[int, float, float]], end_ns: int, days: int) -> float:
        start = end_ns - days * 86_400_000_000_000
        return sum(
            v * c for t, c, v in pairs if start < t <= end_ns and v is not None
        )

    def _rebalance(self, ts: pd.Timestamp) -> None:
        end_ns, start_ns = self._buckets(ts)

        # Optional liquidity filter: keep only the top fraction by trailing
        # dollar volume (the liquid subset the paper associates with momentum).
        universe: set[InstrumentId] = set(self._legs)
        top_q = self.config.liquidity_top_quantile
        if top_q is not None:
            dv = {
                iid: self._dollar_volume(
                    pairs, end_ns, self.config.liquidity_window_days
                )
                for iid, pairs in self._series.items()
            }
            dv = {iid: d for iid, d in dv.items() if d > 0}
            if dv:
                ranked = sorted(dv, key=lambda iid: dv[iid], reverse=True)
                n_liquid = max(1, round(len(ranked) * top_q))
                universe = set(ranked[:n_liquid])

        # LRET = lagged 1-day log-return over the flip side of the day.
        lret: dict[InstrumentId, float] = {}
        for iid in universe:
            pairs = self._series[iid]
            c_end = self._close_at_or_before(pairs, end_ns)
            c_start = self._close_at_or_before(pairs, start_ns)
            if c_end is None or c_start is None or c_start <= 0 or c_end <= 0:
                continue
            lret[iid] = log(c_end / c_start)

        if not lret:
            return

        ordered = sorted(lret, key=lambda iid: lret[iid])
        n = len(ordered)
        n_sel = max(1, round(n * self.config.top_fraction))
        if self.config.reverse:
            # Reversal: long the worst performers, short the best.
            long_side: set[InstrumentId] = set(ordered[:n_sel])
            short_side: set[InstrumentId] = set(ordered[max(0, n - n_sel):])
        else:
            # Momentum: long the best performers, short the worst.
            long_side = set(ordered[max(0, n - n_sel):])
            short_side = set(ordered[:n_sel])

        for iid in list(self._legs):
            leg = self._leg(iid)
            target: OrderSide | None = (
                OrderSide.BUY if iid in long_side
                else OrderSide.SELL if iid in short_side
                else None
            )
            if leg.side is None:
                if target is not None and leg.price:
                    self.open_position(target, leg.price, iid)
            elif target is None:
                self.exit_market(iid)
            elif leg.side != target and leg.price:
                self.exit_market(iid)
                self.open_position(target, leg.price, iid)
