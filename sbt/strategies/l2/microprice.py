"""Microprice deviation strategy.

Avellaneda & Stoikov (2008), "High-frequency trading in a limit
order book", Quantitative Finance 8(3). The fair value of the
instrument is better estimated by the queue-weighted microprice than
by the mid; Stoikov's micro-price formalization weights the best ask
and bid by the opposing queue sizes. The strategy buys when the
microprice sits meaningfully above the mid (bid-heavy queue) and
sells when it sits below, exiting when the deviation collapses.
"""

from ...plugins import SBTStrategyConfig
from .base import L2EventStrategy


class L2MicropriceConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Configuration for the microprice strategy."""

    signal_interval_ms: int = 250
    entry_threshold: float = 0.25
    exit_threshold: float = 0.05
    max_hold_seconds: int = 60
    max_spread_bps: float = 0.0
    capital_fraction: float = 0.10


class L2Microprice(L2EventStrategy):
    """Trades the microprice deviation from mid, in ticks.

    Signal = (microprice - mid) / tick_size, so thresholds are
    expressed in tick units. Microprice = (q_bid*p_ask + q_ask*p_bid)
    / (q_bid + q_ask).
    """

    def _compute_signal(self, ts_event: int) -> float | None:
        if not self._bids or not self._asks:
            return None
        bid_px = max(self._bids)
        ask_px = min(self._asks)
        q_bid = self._bids[bid_px]
        q_ask = self._asks[ask_px]
        total = q_bid + q_ask
        if total <= 0:
            return None
        micro = (q_bid * ask_px + q_ask * bid_px) / total
        mid = (bid_px + ask_px) / 2.0
        instrument = self.cache.instrument(self.instrument_id)
        tick = float(instrument.price_increment)
        if tick <= 0:
            return None
        return (micro - mid) / tick
