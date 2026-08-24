"""Order book volume pressure across depth levels.

Bouchaud, Mezard & Potters (2002), "Statistical properties of stock
order books: empirical results and models", Quantitative Finance 2(4).
The LOB has a well-defined average shape; volume fluctuations around
that shape carry predictive information. The strategy measures the
imbalance of depth-weighted resting volume across the first N levels
on each side ("level pressure") and trades its sign on a slow grid.
"""

from ...plugins import SBTStrategyConfig
from .base import L2EventStrategy


class L2BookPressureConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Configuration for the book pressure strategy."""

    signal_interval_ms: int = 1000
    entry_threshold: float = 0.2
    exit_threshold: float = 0.05
    max_hold_seconds: int = 60
    max_spread_bps: float = 0.0
    capital_fraction: float = 0.10
    pressure_levels: int = 10
    level_decay: float = 1.0


class L2BookPressure(L2EventStrategy):
    """Weighted volume imbalance over the first N price levels per side."""

    def _volume_pressure(self, is_bid: bool) -> float:
        book = self._bids if is_bid else self._asks
        n = max(1, min(self.config.pressure_levels, 50))
        keys = sorted(book, reverse=is_bid)[:n]
        total = 0.0
        for j, p in enumerate(keys):
            w = 1.0 / ((j + 1) ** self.config.level_decay)
            total += w * book[p]
        return total

    def _compute_signal(self, ts_event: int) -> float | None:
        v_bid = self._volume_pressure(True)
        v_ask = self._volume_pressure(False)
        total = v_bid + v_ask
        if total <= 0:
            return None
        return (v_bid - v_ask) / total
