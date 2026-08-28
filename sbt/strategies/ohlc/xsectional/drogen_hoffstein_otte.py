"""Cross-sectional momentum in crypto, Drogen, Hoffstein & Otte (2023, SSRN
4322637).

Rank the basket by trailing ``formation_days`` (30) price momentum and hold the
**top quintile** winners on a ``continuation_days`` (7) weekly rebalance
schedule — a 7-day continuation after the 30-day filter. The headline book is a
**long-only** top-quintile portfolio benchmarked vs Bitcoin buy-and-hold; a
long-short WML variant is available via ``long_only=False`` for comparison with
the other rank-sort tickets on the map.

The universe is filtered each rebalance: an asset must have an average dollar
volume ≥ ``liquidity_min_dollar`` over the prior ``liquidity_days`` (paper: $5M
over 30 days) to be tradable.
"""

import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ....plugins import SBTPortfolioStrategyConfig
from ...base import SBTPortfolioStrategy

_DAY_NS = 86_400_000_000_000


class MomentumWinnersConfig(SBTPortfolioStrategyConfig, kw_only=True, frozen=True):
    formation_days: int = 30      # trailing 30-day momentum ranking window
    continuation_days: int = 7    # weekly 7-day rebalance / holding cadence
    top_fraction: float = 0.2     # top quintile
    long_only: bool = True        # False => WML (short the bottom quintile too)
    liquidity_min_dollar: float = 5_000_000.0   # paper: $5M avg dollar volume threshold
    liquidity_days: int = 30      # trailing window for the liquidity filter


class MomentumWinners(SBTPortfolioStrategy):
    def __init__(self, config: MomentumWinnersConfig) -> None:
        super().__init__(config)
        if config.formation_days <= 0 or config.continuation_days <= 0:
            raise ValueError("formation_days and continuation_days must be positive")
        if not 0 < config.top_fraction < 0.5:
            raise ValueError(f"top_fraction must be in (0, 0.5), got {config.top_fraction}")
        # Per-leg (ts_ns, close, volume).
        self._series: dict[InstrumentId, list[tuple[int, float, float]]] = {
            iid: [] for iid in self._legs
        }
        self._next_rebal_ns: int | None = None

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        self._series[instrument_id].append(
            (
                bar.ts_event,
                float(bar.close.as_double()),
                float(bar.volume.as_double() if hasattr(bar, "volume") else 0.0),
            )
        )
        if instrument_id != self._primary_iid:
            return
        day_ns = int(self._ts(bar).normalize().value)
        if self._next_rebal_ns is None:
            self._next_rebal_ns = day_ns
        if not self.trading_active or day_ns < self._next_rebal_ns:
            return
        self._next_rebal_ns = day_ns + self.config.continuation_days * _DAY_NS
        self._rebalance(day_ns)

    @staticmethod
    def _ts(bar: Bar) -> pd.Timestamp:
        return pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")

    @staticmethod
    def _window(
        pairs: list[tuple[int, float, float]], lo: int, hi: int
    ) -> list[tuple[int, float, float]]:
        return [(t, c, v) for t, c, v in pairs if lo < t <= hi]

    def _rebalance(self, day_ns: int) -> None:
        form_start = day_ns - self.config.formation_days * _DAY_NS
        liq_start = day_ns - self.config.liquidity_days * _DAY_NS

        mom: dict[InstrumentId, float] = {}
        liquid: set[InstrumentId] = set()

        for iid, pairs in self._series.items():
            form = self._window(pairs, form_start, day_ns)
            if not form:
                continue
            first, last = form[0][1], form[-1][1]
            if first <= 0:
                continue
            mom[iid] = last / first - 1.0

            liq = self._window(pairs, liq_start, day_ns)
            dollar = sum(v * c for _t, c, v in liq)
            avg = dollar / len(liq) if liq else 0.0
            if avg >= self.config.liquidity_min_dollar:
                liquid.add(iid)

        if not mom:
            return

        eligible = [iid for iid in mom if iid in liquid]
        if len(eligible) < 2:
            return
        ordered = sorted(eligible, key=lambda iid: mom[iid])
        n = len(ordered)
        n_sel = max(1, round(n * self.config.top_fraction))
        longset: set[InstrumentId] = set(ordered[max(0, n - n_sel):])
        shortset: set[InstrumentId] = set(ordered[:n_sel]) if not self.config.long_only else set()

        targets: dict[InstrumentId, OrderSide | None] = {
            iid: (OrderSide.BUY if iid in longset
                  else OrderSide.SELL if iid in shortset
                  else None)
            for iid in self._legs
        }
        self.apply_targets(targets)
