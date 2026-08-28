"""Size/volume-conditioned momentum-reversal + high-momentum, Fičura (2023).

FFA Working Paper. Weekly frequency. The 1-week return **reversal** occurs
only for small/illiquid coins, while large/liquid coins show weekly
**momentum**; the strategy splits the basket by liquidity and applies the
matching direction within each group. A separate **high-momentum** signal
(George & Hwang 2004) ranks coins by the distance of the week-close from the
``hk_weeks``-week intraday high: ``hmom = ln(C_t) - ln(H_{t,h})`` where
``H_{t,h}`` is the highest intraday price over the past ``hk_weeks`` — a
superior, momentum-compatible predictor that amplifies the large/small gap.

Size/market-cap is ***not*** in OHLCV data, so liquidity is proxied by
trailing **dollar volume** (the paper itself finds the reversal is driven by
low trading volume). ``liquid_fraction`` takes the top fraction of legs by
dollar volume as the large/liquid group; the rest are small/illiquid.
"""

from math import log
import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ....plugins import SBTPortfolioStrategyConfig
from ...base import SBTPortfolioStrategy

_WEEK_NS = 7 * 86_400_000_000_000


class SizeVolumeMomentumConfig(SBTPortfolioStrategyConfig, kw_only=True, frozen=True):
    signal: str = "momentum"  # "momentum" | "high_momentum"
    group: str = "large"      # "large" | "small" | "both"
    hk_weeks: int = 1         # h in hmom: weeks for the intraday-high window
    liquid_fraction: float = 0.5  # top fraction by $vol treated as large/liquid
    liquid_window_weeks: int = 1  # trailing weeks for the dollar-volume proxy
    top_fraction: float = 0.2  # quintile tails (Q1/Q5) within a group


class SizeVolumeMomentum(SBTPortfolioStrategy):
    _SIGNALS = ("momentum", "high_momentum")
    _GROUPS = ("large", "small", "both")

    def __init__(self, config: SizeVolumeMomentumConfig) -> None:
        super().__init__(config)
        if config.signal not in self._SIGNALS:
            raise ValueError(f"signal must be one of {self._SIGNALS}, got {config.signal!r}")
        if config.group not in self._GROUPS:
            raise ValueError(f"group must be one of {self._GROUPS}, got {config.group!r}")
        if not 0 < config.top_fraction < 0.5:
            raise ValueError(f"top_fraction must be in (0, 0.5), got {config.top_fraction}")
        if not 0 < config.liquid_fraction <= 1:
            raise ValueError(
                f"liquid_fraction must be in (0, 1], got {config.liquid_fraction}"
            )
        # Per-leg (ts_ns, close, high, volume).
        self._series: dict[InstrumentId, list[tuple[int, float, float, float]]] = {
            iid: [] for iid in self._legs
        }
        self._last_week: str | None = None

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        self._series[instrument_id].append(
            (
                bar.ts_event,
                float(bar.close.as_double()),
                float(bar.high.as_double()),
                float(bar.volume.as_double() if hasattr(bar, "volume") else 0.0),
            )
        )

        # Rebalance once per ISO week, keyed on the primary leg.
        if instrument_id != self._primary_iid:
            return
        dt = self._ts(bar)
        week = dt.strftime("%G-%V")
        if week == self._last_week or not self.trading_active:
            return
        self._last_week = week
        self._rebalance(dt)

    @staticmethod
    def _ts(bar: Bar) -> pd.Timestamp:
        return pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")

    def _week_end_ns(self, ts: pd.Timestamp) -> int:
        cur = ts.to_period("W").to_timestamp().tz_localize("UTC")
        return int((cur - pd.Timedelta(days=1)).normalize().value)

    @staticmethod
    def _close_at_or_before(pairs: list[tuple[int, float, float, float]], ns: int) -> float | None:
        close: float | None = None
        for t, c, _h, _v in pairs:
            if t > ns:
                break
            close = c
        return close

    @staticmethod
    def _max_high_in(pairs: list[tuple[int, float, float, float]], lo: int, hi: int) -> float | None:
        best: float | None = None
        for t, _c, h, _v in pairs:
            if t <= lo:
                continue
            if t > hi:
                break
            best = h if best is None else max(best, h)
        return best

    @staticmethod
    def _dollar_volume(pairs: list[tuple[int, float, float, float]], lo: int, hi: int) -> float:
        return sum(
            v * c for t, c, _h, v in pairs if lo < t <= hi
        )

    def _rebalance(self, ts: pd.Timestamp) -> None:
        end_ns = self._week_end_ns(ts)

        # Per-leg signal and liquidity.
        signal: dict[InstrumentId, float] = {}
        liquidity: dict[InstrumentId, float] = {}
        for iid, pairs in self._series.items():
            c_end = self._close_at_or_before(pairs, end_ns)
            if c_end is None or c_end <= 0:
                continue
            liq = self._dollar_volume(pairs, end_ns - self.config.liquid_window_weeks * _WEEK_NS, end_ns)
            liquidity[iid] = liq
            if self.config.signal == "high_momentum":
                H = self._max_high_in(
                    pairs, end_ns - self.config.hk_weeks * _WEEK_NS, end_ns
                )
                if H is None or H <= 0:
                    continue
                signal[iid] = log(c_end) - log(H)
            else:  # momentum: trailing 1-week log-return
                c_start = self._close_at_or_before(pairs, end_ns - _WEEK_NS)
                if c_start is None or c_start <= 0:
                    continue
                signal[iid] = log(c_end / c_start)

        if not signal or not liquidity:
            return

        # Liquidity split: large/liquid = top `liquid_fraction` by dollar volume.
        liq_rank = sorted(liquidity, key=lambda iid: liquidity[iid], reverse=True)
        n_liquid = max(1, round(len(liq_rank) * self.config.liquid_fraction))
        liquid: set[InstrumentId] = set(liq_rank[:n_liquid])
        illiquid: set[InstrumentId] = set(liq_rank[n_liquid:])

        # Direction: within a group, momentum (or high-momentum) longs the best
        # signal; illiquid coins reverse => long the worst. Select by `group`.
        traded: dict[InstrumentId, bool] = {}  # iid -> reverse?
        if self.config.group == "large":
            traded = {iid: False for iid in liquid}
        elif self.config.group == "small":
            traded = {iid: True for iid in illiquid}
        else:  # both
            traded = {**{iid: False for iid in liquid}, **{iid: True for iid in illiquid}}

        active = {iid: v for iid, v in traded.items() if iid in signal}
        if not active:
            return

        ordered = sorted(active, key=lambda iid: signal[iid])
        n = len(ordered)
        n_sel = max(1, round(n * self.config.top_fraction))
        best: set[InstrumentId] = set(ordered[max(0, n - n_sel):])
        worst: set[InstrumentId] = set(ordered[:n_sel])

        for iid in list(self._legs):
            if iid not in active:
                target: OrderSide | None = None
            else:
                reverse = active[iid]
                if reverse:
                    # Reversal: long the worst-signal tail, short the best.
                    target = (
                        OrderSide.BUY if iid in worst else OrderSide.SELL if iid in best else None
                    )
                else:
                    # Momentum: long the best-signal tail, short the worst.
                    target = (
                        OrderSide.BUY if iid in best else OrderSide.SELL if iid in worst else None
                    )
            leg = self._leg(iid)
            if leg.side is None:
                if target is not None and leg.price:
                    self.open_position(target, leg.price, iid)
            elif target is None:
                self.exit_market(iid)
            elif leg.side != target:
                self.exit_market(iid)
                if leg.price:
                    self.open_position(target, leg.price, iid)
