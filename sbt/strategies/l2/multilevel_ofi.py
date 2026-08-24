"""Multi-level order flow imbalance (MLOFI).

Xu, Gould & Howison (2019), "Multi-Level Order-Flow Imbalance in a
Limit Order Book", arXiv:1907.06230. The MLOFI vector tracks net
order flow at the first M price levels per side (their eqs. 9-12);
the paper fits a linear relation with mid-price changes via ridge
regression. Here the per-level flows are combined with exponential
level weights (a shrinkage stand-in for ridge), EWMA-smoothed across
sampling windows and normalized by weighted depth.
"""

from ...plugins import SBTStrategyConfig
from .base import L2EventStrategy, clamped_dt_s, ewma_alpha


class L2MultilevelOFIConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Configuration for the multi-level OFI strategy."""

    signal_interval_ms: int = 500
    entry_threshold: float = 0.3
    exit_threshold: float = 0.05
    max_hold_seconds: int = 120
    max_spread_bps: float = 0.0
    capital_fraction: float = 0.10
    ofi_levels: int = 5
    ofi_level_decay: float = 0.5
    ml_ofi_half_life_ms: int = 5000


class L2MultilevelOFI(L2EventStrategy):
    """EWMA of depth-weighted multi-level flow, normalized by depth."""

    def __init__(self, config: L2MultilevelOFIConfig) -> None:
        super().__init__(config)
        self._prev_bid_levels: list[tuple[float, float]] = []
        self._prev_ask_levels: list[tuple[float, float]] = []
        self._ml_ewma: float = 0.0
        self._ml_last_ts: int | None = None
        self._ml_window: float | None = None

    def _level_snapshot(
        self, book: dict[float, float], m: int, is_bid: bool
    ) -> list[tuple[float, float]]:
        keys = sorted(book, reverse=is_bid)[:m]
        return [(p, book[p]) for p in keys]

    @staticmethod
    def _bid_flow(cur, old) -> float:
        if cur is None:
            return -old[1] if old is not None else 0.0
        if old is None:
            return cur[1]
        if cur[0] > old[0]:
            return cur[1]
        if cur[0] < old[0]:
            return -old[1]
        return cur[1] - old[1]

    @staticmethod
    def _ask_flow(cur, old) -> float:
        if cur is None:
            return -old[1] if old is not None else 0.0
        if old is None:
            return cur[1]
        if cur[0] < old[0]:
            return cur[1]
        if cur[0] > old[0]:
            return -old[1]
        return cur[1] - old[1]

    def _compute_signal(self, ts_event: int) -> float | None:
        m = max(1, min(self.config.ofi_levels, 25))
        cur_bids = self._level_snapshot(self._bids, m, True)
        cur_asks = self._level_snapshot(self._asks, m, False)
        decay = self.config.ofi_level_decay
        raw = 0.0
        denom = 0.0
        for j in range(m):
            w = decay**j
            cb = cur_bids[j] if j < len(cur_bids) else None
            ca = cur_asks[j] if j < len(cur_asks) else None
            ob = self._prev_bid_levels[j] if j < len(self._prev_bid_levels) else None
            oa = self._prev_ask_levels[j] if j < len(self._prev_ask_levels) else None
            e = self._bid_flow(cb, ob) - self._ask_flow(ca, oa)
            raw += w * e
            if cb is not None:
                denom += w * cb[1]
            if ca is not None:
                denom += w * ca[1]
        if self._ml_window is not None and self._ml_last_ts is not None:
            dt_s = clamped_dt_s(ts_event, self._ml_last_ts)
            hl = self.config.ml_ofi_half_life_ms / 1000.0
            alpha = ewma_alpha(dt_s, hl)
            self._ml_ewma += alpha * (self._ml_window - self._ml_ewma)
        self._ml_window = raw
        self._ml_last_ts = ts_event
        self._prev_bid_levels = cur_bids
        self._prev_ask_levels = cur_asks
        if denom <= 0:
            return None
        return self._ml_ewma / denom
