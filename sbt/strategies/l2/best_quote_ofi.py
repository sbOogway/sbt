"""Best-quote order flow imbalance (OFI).

Cont, Kukanov & Stoikov (2014), "The price impact of order book
events", Journal of Financial Econometrics 12(1). Per-event flow
e_n (their eq. 1) at the best bid/ask is accumulated over sampling
windows, EWMA-smoothed and normalized by average best-quote depth:
the paper documents a linear relation dP ~ OFI / D. Traded by
thresholding the normalized flow.
"""

import math

from nautilus_trader.model.data import OrderBookDelta
from nautilus_trader.model.enums import BookAction, OrderSide

from ...plugins import SBTStrategyConfig
from .base import L2EventStrategy


class L2BestQuoteOFIConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Configuration for the best-quote OFI strategy."""

    signal_interval_ms: int = 500
    entry_threshold: float = 0.3
    exit_threshold: float = 0.05
    max_hold_seconds: int = 120
    max_spread_bps: float = 0.0
    capital_fraction: float = 0.10
    ofi_half_life_ms: int = 2000


class L2BestQuoteOFI(L2EventStrategy):
    """EWMA of best-quote OFI divided by EWMA of best-quote depth."""

    def __init__(self, config: L2BestQuoteOFIConfig) -> None:
        super().__init__(config)
        self._best_bid: tuple[float, float] | None = None
        self._best_ask: tuple[float, float] | None = None
        self._ofi_window: float = 0.0
        self._depth_sum: float = 0.0
        self._depth_n: int = 0
        self._ofi_ewma: float = 0.0
        self._depth_ewma: float = 0.0

    def _on_order_event(
        self, delta: OrderBookDelta, price: float, size: float
    ) -> None:
        is_bid = delta.order.side == OrderSide.BUY
        book = self._bids if is_bid else self._asks
        action = delta.action

        def rescan() -> tuple[float, float] | None:
            if not book:
                return None
            px = max(book) if is_bid else min(book)
            return (px, book[px])

        prev = self._best_bid if is_bid else self._best_ask
        new_best: tuple[float, float] | None
        if action == BookAction.CLEAR:
            new_best = rescan()
        elif action == BookAction.DELETE:
            new_best = rescan() if prev is None or price == prev[0] else prev
        elif (
            prev is None
            or price == prev[0]
            or (is_bid and price > prev[0])
            or (not is_bid and price < prev[0])
        ):
            new_best = (price, book[price])
        else:
            new_best = prev

        if new_best is not None:
            if is_bid:
                self._best_bid = new_best
            else:
                self._best_ask = new_best

        bb, ba = self._best_bid, self._best_ask
        if bb is not None and ba is not None:
            self._depth_sum += bb[1] + ba[1]
            self._depth_n += 1

        if prev is None or new_best is None or new_best == prev:
            return

        p_prev, q_prev = prev
        p_new, q_new = new_best
        if is_bid:
            if p_new > p_prev:
                e = q_new
            elif p_new < p_prev:
                e = -q_prev
            else:
                e = q_new - q_prev
        else:
            if p_new < p_prev:
                e = -q_new
            elif p_new > p_prev:
                e = q_prev
            else:
                e = q_prev - q_new
        self._ofi_window += e

    @staticmethod
    def _alpha(dt_s: float, half_life_s: float) -> float:
        if half_life_s <= 0.0 or dt_s <= 0.0:
            return 1.0
        return 1.0 - math.exp(-math.log(2.0) * dt_s / half_life_s)

    def _compute_signal(self, ts_event: int) -> float | None:
        if self._best_bid is None or self._best_ask is None:
            return None
        dt_s = (
            (ts_event - self._last_sample_ts) / 1e9
            if self._last_sample_ts is not None
            else 0.0
        )
        dt_s = min(max(dt_s, 0.0), 60.0)
        alpha = self._alpha(dt_s, self.config.ofi_half_life_ms / 1000.0)
        self._ofi_ewma += alpha * (self._ofi_window - self._ofi_ewma)
        if self._depth_n > 0:
            window_depth = self._depth_sum / self._depth_n
            self._depth_ewma += alpha * (window_depth - self._depth_ewma)
        self._ofi_window = 0.0
        self._depth_sum = 0.0
        self._depth_n = 0
        if self._depth_ewma <= 0:
            return None
        return self._ofi_ewma / self._depth_ewma
