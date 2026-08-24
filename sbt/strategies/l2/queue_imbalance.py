"""Queue imbalance as a one-tick-ahead price predictor.

Gould & Bonart (2016), "Queue Imbalance as a One-Tick-Ahead Price
Predictor in a Limit Order Book", Quantitative Finance 16(2).
Signal is the imbalance between best-bid and best-ask queue sizes;
a logit fit on Nasdaq stocks shows it predicts the direction of the
next mid-price move. Traded here with a ratio threshold, hysteresis
exit and a short time stop.
"""

from ...plugins import SBTStrategyConfig
from .base import L2EventStrategy


class L2QueueImbalanceConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Configuration for the queue imbalance strategy."""

    signal_interval_ms: int = 250
    entry_threshold: float = 0.25
    exit_threshold: float = 0.05
    max_hold_seconds: int = 30
    max_spread_bps: float = 0.0
    capital_fraction: float = 0.10


class L2QueueImbalance(L2EventStrategy):
    """Long when the bid queue dominates the best level, short when the
    ask queue dominates."""

    def _compute_signal(self, ts_event: int) -> float | None:
        if not self._bids or not self._asks:
            return None
        q_bid = self._bids[max(self._bids)]
        q_ask = self._asks[min(self._asks)]
        total = q_bid + q_ask
        if total <= 0:
            return None
        return (q_bid - q_ask) / total
