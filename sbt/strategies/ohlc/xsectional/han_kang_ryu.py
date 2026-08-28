"""Time-series and cross-sectional momentum with realistic costs, Han, Kang &
Ryu (2024, AUT WP).

**Time-series momentum (primary, strong)**: buy the market basket when its
equal-weighted return over ``lookback_days`` is positive, else flat; re-form
every ``holding_days``. Optimal pair in the paper: look-back 28d / holding 5d.
Marked to market daily by the engine (honest cost accounting — interim
drawdowns are never hidden).

**Cross-sectional momentum (secondary, weak)**: rank legs by the trailing
``lookback_days`` return each ``holding_days`` and go long the top ``top_fraction``
winners / short the bottom ``top_fraction`` losers. The paper finds CS momentum is
fragile — profit concentrated in the long leg, losers rebound, and most of the
edge vanishes net of costs — so net-of-fee CS underperformance is the honest
finding, not a bug. Enable ``subscribe_funding`` and set a realistic
``taker_fee``/``slippage_ticks`` at the run level to surface the real cost burden.
"""

import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ....plugins import SBTPortfolioStrategyConfig
from ...base import SBTPortfolioStrategy

_DAY_NS = 86_400_000_000_000


class TSXSMomentumConfig(SBTPortfolioStrategyConfig, kw_only=True, frozen=True):
    mode: str = "ts"            # "ts" (strong) | "cs" (weak)
    lookback_days: int = 28     # ts: 28, cs: 14
    holding_days: int = 5       # ts: 5, cs: 7
    top_fraction: float = 0.3   # cs long/short tails
    subscribe_funding: bool = True


class TSXSMomentum(SBTPortfolioStrategy):
    _MODES = ("ts", "cs")

    def __init__(self, config: TSXSMomentumConfig) -> None:
        super().__init__(config)
        if config.mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {config.mode!r}")
        if config.lookback_days <= 0 or config.holding_days <= 0:
            raise ValueError("lookback_days and holding_days must be positive")
        if not 0 < config.top_fraction < 0.5:
            raise ValueError(f"top_fraction must be in (0, 0.5), got {config.top_fraction}")
        # Per-leg daily (ts_ns, close).
        self._series: dict[InstrumentId, list[tuple[int, float]]] = {
            iid: [] for iid in self._legs
        }
        self._next_rebal_ns: int | None = None

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        self._series[instrument_id].append(
            (bar.ts_event, float(bar.close.as_double()))
        )
        if instrument_id != self._primary_iid:
            return
        day_ns = int(self._ts(bar).normalize().value)
        if self._next_rebal_ns is None:
            self._next_rebal_ns = day_ns
        if not self.trading_active or day_ns < self._next_rebal_ns:
            return
        self._next_rebal_ns = day_ns + self.config.holding_days * _DAY_NS
        self._rebalance(day_ns)

    @staticmethod
    def _ts(bar: Bar) -> pd.Timestamp:
        return pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")

    @staticmethod
    def _close_at_or_before(pairs: list[tuple[int, float]], ns: int) -> float | None:
        close: float | None = None
        for t, c in pairs:
            if t > ns:
                break
            close = c
        return close

    @classmethod
    def _return(
        cls, pairs: list[tuple[int, float]], start_ns: int, end_ns: int
    ) -> float | None:
        c_end = cls._close_at_or_before(pairs, end_ns)
        c_start = cls._close_at_or_before(pairs, start_ns)
        if c_end is None or c_start is None or c_start <= 0:
            return None
        return c_end / c_start - 1.0

    def _rebalance(self, day_ns: int) -> None:
        start_ns = day_ns - self.config.lookback_days * _DAY_NS
        if self.config.mode == "ts":
            self._rebalance_ts(day_ns, start_ns)
        else:
            self._rebalance_cs(day_ns, start_ns)

    def _rebalance_ts(self, day_ns: int, start_ns: int) -> None:
        rets = [
            self._return(self._series[iid], start_ns, day_ns)
            for iid in self._legs
        ]
        rets = [r for r in rets if r is not None]
        if not rets:
            return
        long_market = sum(rets) / len(rets) > 0
        for iid in list(self._legs):
            leg = self._leg(iid)
            if long_market:
                if leg.side is None and leg.price:
                    self.open_position(OrderSide.BUY, leg.price, iid)
            elif leg.side is not None:
                self.exit_market(iid)

    def _rebalance_cs(self, day_ns: int, start_ns: int) -> None:
        signals: dict[InstrumentId, float] = {}
        for iid in self._legs:
            r = self._return(self._series[iid], start_ns, day_ns)
            if r is not None:
                signals[iid] = r
        if len(signals) < 2:
            return
        ordered = sorted(signals, key=lambda iid: signals[iid])
        n = len(ordered)
        n_sel = max(1, round(n * self.config.top_fraction))
        longset: set[InstrumentId] = set(ordered[max(0, n - n_sel):])
        shortset: set[InstrumentId] = set(ordered[:n_sel])

        for iid in list(self._legs):
            leg = self._leg(iid)
            if iid in longset:
                target = OrderSide.BUY
            elif iid in shortset:
                target = OrderSide.SELL
            else:
                target = None
            if target is None:
                if leg.side is not None:
                    self.exit_market(iid)
            elif leg.side is None:
                if leg.price:
                    self.open_position(target, leg.price, iid)
            elif leg.side != target:
                self.exit_market(iid)
                if leg.price:
                    self.open_position(target, leg.price, iid)
