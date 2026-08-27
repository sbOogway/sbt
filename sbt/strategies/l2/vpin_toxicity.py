"""VPIN-gated order-flow reversal.

Easley, Lopez de Prado & O'Hara (2012), "Flow Toxicity and Liquidity in a
High-Frequency World", Review of Financial Studies 25(5). Flow is "toxic" when
it adversely selects liquidity providers; they estimate this with the
Volume-Synchronized Probability of Informed Trading (VPIN), the average
absolute signed order imbalance over equal-volume buckets, updated in volume
time. High VPIN marks a toxicity regime that precedes sharp short-horizon
volatility, and the paper argues traders should avoid (or trade cautiously in)
extreme-toxicity states.

Here VPIN is computed directly from aggressor-side-tagged trade ticks
(simplification of the paper's bulk classification, which we need not use
because the tick stream carries an aggressor flag). Trades are packed into
fixed-volume buckets (volume time); each completed bucket yields a signed
imbalance S = V_buy - V_sell. VPIN = mean(|S|) over the trailing n buckets.
Toxicity is judged relative to recent history -- a rolling z-score of VPIN
against the per-bucket imbalance distribution, mirroring the paper's "relative
VPIN via CDF" guidance (footnote 19).

Trading rule (our adaptation -- the paper's metric is non-directional):
when toxicity z is in its upper tail, fade the dominant flow (contrarian
reversal). The paper stresses high VPIN precedes large moves but not their sign
(footnote 21); the contrarian direction leans on the flash-crash recovery and
the LOB-reversal finding that imbalance signals are most often faded
profitably. In quiet regimes (low z) the strategy stands aside.
"""

import math
from collections import deque

from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide

from ...plugins import SBTStrategyConfig
from .base import L2EventStrategy


class L2VpinToxicityConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Configuration for the VPIN toxicity reversal strategy."""

    # Event-time sampling grid (ms) on which the VPIN state is polled.
    signal_interval_ms: int = 250
    # Volume-clock: base-quantity (BTC) that fills one volume bucket.
    # The paper anchors at ADV/50 but stresses VPIN is robust to V and n;
    # a finer clock is used here to yield a meaningful signal count on a
    # ~3-week sample.
    bucket_btc: float = 200.0
    # Number of trailing buckets averaged into the VPIN toxicity estimate.
    vpin_buckets: int = 50
    # Number of buckets used for the rolling mean/std of per-bucket imbalance
    # against which VPIN's relative (z) toxicity is judged.
    tox_avg_buckets: int = 100
    # Trade only when VPIN z-score exceeds this many std above its recent mean.
    toxicity_z_entry: float = 1.0
    # Whether to fade the dominant toxic flow (contrarian); False follows it.
    fade: bool = True
    # Base entry/exit thresholds applied to the normalized signed signal.
    entry_threshold: float = 0.35
    exit_threshold: float = 0.05
    max_hold_seconds: int = 900
    # Skip new entries while the top-of-book spread exceeds this (bps).
    max_spread_bps: float = 5.0
    # Fraction of equity (times leverage) traded per entry.
    capital_fraction: float = 0.10


class L2VpinToxicity(L2EventStrategy):
    """Fades dominant order flow when volume-time flow toxicity is elevated."""

    needs_trade_ticks: bool = True

    def __init__(self, config: L2VpinToxicityConfig) -> None:
        super().__init__(config)
        # Current (partial) volume bucket accumulation by aggressor side.
        self._cur_vol: float = 0.0
        self._cur_buy: float = 0.0
        self._cur_sell: float = 0.0
        # Completed buckets' signed imbalances V_buy - V_sell.
        self._buckets: deque[float] = deque()
        self._signed_final: float = 0.0

    def on_trade_tick(self, tick: TradeTick) -> None:
        size = float(tick.size)
        if size <= 0:
            return
        aggressor = tick.aggressor_side
        if aggressor == AggressorSide.BUYER:
            v_buy, v_sell = size, 0.0
        elif aggressor == AggressorSide.SELLER:
            v_buy, v_sell = 0.0, size
        else:  # NO_AGGRESSOR / unknown: split the volume 50/50
            v_buy = v_sell = size / 2.0
        self._cur_vol += size
        self._cur_buy += v_buy
        self._cur_sell += v_sell
        if self._cur_vol >= self.config.bucket_btc:
            self._signed_final = self._cur_buy - self._cur_sell
            self._buckets.append(self._signed_final)
            cap = max(self.config.vpin_buckets, self.config.tox_avg_buckets)
            while len(self._buckets) > cap:
                self._buckets.popleft()
            # Carry the excess volume (and any imbalance) into the next bucket.
            excess = self._cur_vol - self.config.bucket_btc
            self._cur_vol = excess
            self._cur_buy = 0.0
            self._cur_sell = 0.0

    def _toxicity(self) -> float | None:
        """Return (vpin, z) where vpin in [0, 0.5], z = (vpin-mu)/sigma."""
        window = self._buckets
        if len(window) < self.config.tox_avg_buckets:
            return None
        ois = [abs(s) for s in window]
        n = self.config.vpin_buckets
        recent = ois[-n:]
        vpin = sum(recent) / len(recent) if recent else 0.0
        mu = sum(ois) / len(ois)
        var = sum((o - mu) ** 2 for o in ois) / len(ois)
        sigma = math.sqrt(var)
        if sigma <= 0.0:
            return vpin, 0.0
        return vpin, (vpin - mu) / sigma

    def _compute_signal(self, ts_event: int) -> float | None:
        tox = self._toxicity()
        if tox is None:
            return None
        _, z = tox
        if z <= self.config.toxicity_z_entry:
            return 0.0
        # Gate open: magnitude ramps smoothly with toxicity elevation and
        # saturates, so a meaningfully-open gate reliably crosses the entry
        # threshold without arbitrary hard caps.
        mag = 1.0 - math.exp(-(z - self.config.toxicity_z_entry))
        sign_s = -1.0 if self._signed_final < 0 else (1.0 if self._signed_final > 0 else 0.0)
        if not self.config.fade:
            sign_s = -sign_s
        return mag * sign_s
