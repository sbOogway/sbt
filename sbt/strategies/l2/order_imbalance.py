"""Layer 2 Order Book Imbalance & Microstructure Strategy.

Signals follow the order-flow imbalance literature:
- Cont, Kukanov & Stoikov (2014): best-quote order flow imbalance (OFI),
  per-event contributions accumulated with exponential decay and normalized
  by average best-quote depth (linear impact: dP ~ OFI / (2D)).
- Xu et al. (2019): multi-level OFI -- net flow at deeper levels adds
  predictive power; implemented as exponentially weighted rank-matched
  level-volume changes between sampling snapshots (simplification of their
  ridge-regression combination).
- Gould & Bonart (2016): static queue imbalance predicts the direction of
  the next mid-price move.
- Bieganowski & Slepaczuk (2026): signed trade flow carries crypto perpetual
  microstructure information; spread acts as a liquidity gate.
"""

from nautilus_trader.model.data import OrderBookDelta, TradeTick
from nautilus_trader.model.enums import AggressorSide, BookAction, OrderSide

from ...plugins import SBTStrategyConfig
from .base import L2EventStrategy, clamped_dt_s, ewma_alpha


class L2OrderImbalanceConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Configuration for L2 Order Book Imbalance Strategy."""

    # Composite entry threshold applied to the blended signal z.
    entry_threshold: float = 0.6
    # Legacy event-count sampling, used only when signal_interval_ms <= 0.
    cooldown_events: int = 50
    # Fraction of current account equity (times leverage) traded as notional
    # per entry, e.g. 0.10 -> $100 notional on a $1000 account at 1x.
    # Must exceed one size lot (0.001 BTC) times price to be tradable.
    capital_fraction: float = 0.10
    # Hysteresis exit: flatten when |z| decays below this (0 disables;
    # legacy behavior flips only on a full opposite entry signal).
    exit_threshold: float = 0.0
    # Force-flat time stop in seconds (0 disables).
    max_hold_seconds: int = 0
    # Depth (price levels per side) used for the static imbalance measurement.
    top_levels: int = 5

    # Minimum event-time span between signal evaluations (ms);
    # <= 0 falls back to legacy cooldown_events counting.
    signal_interval_ms: int = 500
    # Best-quote OFI: EWMA half-life (ms) shared by flow and depth normalizer.
    ofi_half_life_ms: int = 2000
    # Multi-level OFI: levels tracked and exponential decay weight per deeper
    # level (w_j = ofi_level_decay ** (j - 1)).
    ofi_levels: int = 5
    ofi_level_decay: float = 0.5
    ml_ofi_half_life_ms: int = 5000
    # Signed trade-flow EWMA half-life (ms).
    trade_half_life_ms: int = 10000
    # Component weights of the composite signal (normalized internally).
    depth_weight: float = 0.3
    ofi_weight: float = 0.5
    ml_ofi_weight: float = 0.2
    trade_flow_weight: float = 0.2
    # Skip NEW entries while top-of-book spread exceeds this (bps; 0 disables).
    max_spread_bps: float = 5.0


class L2OrderImbalance(L2EventStrategy):
    """Exploits short-term microstructural order book depth and flow imbalance.

    Blends four components into one composite signal: static top-of-book
    depth imbalance, CKS best-quote OFI, multi-level OFI, and signed trade
    flow — each EWMA-smoothed between sampling snapshots.
    """

    needs_trade_ticks = True

    def __init__(self, config: L2OrderImbalanceConfig) -> None:
        super().__init__(config)
        self._events_since_trade: int = 0

        self._best_bid: tuple[float, float] | None = None
        self._best_ask: tuple[float, float] | None = None

        # Window accumulators (raw sums since the previous signal evaluation)
        # are smoothed into these EWMAs on the sampling grid.
        self._ofi_ewma: float = 0.0
        self._ofi_depth_ewma: float = 0.0
        self._ofi_window: float = 0.0
        self._ofi_depth_sum: float = 0.0
        self._ofi_depth_n: int = 0
        self._ml_ewma: float = 0.0
        self._ml_last_ts: int | None = None
        self._ml_window: float | None = None
        self._prev_top_bids: list[float] = []
        self._prev_top_asks: list[float] = []
        self._tf_signed_ewma: float = 0.0
        self._tf_abs_ewma: float = 0.0
        self._tf_signed_window: float = 0.0
        self._tf_abs_window: float = 0.0

        weights = (
            config.depth_weight,
            config.ofi_weight,
            config.ml_ofi_weight,
            config.trade_flow_weight,
        )
        self._weight_sum = sum(w for w in weights if w and w > 0)
        self._use_flow = self._weight_sum > 0

    def _sample_due(self, ts_event: int) -> bool:
        if self._interval_ns > 0:
            return super()._sample_due(ts_event)
        return self._events_since_trade >= self.config.cooldown_events

    def _on_order_event(
        self, delta: OrderBookDelta, price: float, size: float
    ) -> None:
        """Track the best quote and accumulate its event flow (CKS)."""
        self._events_since_trade += 1
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

        bb = self._best_bid
        ba = self._best_ask
        if bb is not None and ba is not None:
            self._ofi_depth_sum += bb[1] + ba[1]
            self._ofi_depth_n += 1

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

    def on_trade_tick(self, tick: TradeTick) -> None:
        """Accumulate matched trade prints as signed flow."""
        if not self._use_flow:
            return
        aggressor = tick.aggressor_side
        if aggressor == AggressorSide.NO_AGGRESSOR:
            return
        signed = float(tick.size) * (1.0 if aggressor == AggressorSide.BUYER else -1.0)
        self._tf_signed_window += signed
        self._tf_abs_window += abs(signed)

    def _imbalance(self) -> float | None:
        bid_vol = sum(self._top_sizes(self._bids, self.config.top_levels, True))
        ask_vol = sum(self._top_sizes(self._asks, self.config.top_levels, False))
        total_vol = bid_vol + ask_vol
        if total_vol <= 0:
            return None
        return (bid_vol - ask_vol) / total_vol

    def _multi_level_flow(
        self, ts_event: int, top_bids: list[float], top_asks: list[float]
    ) -> float:
        """Rank-matched level-volume changes, EWMA-smoothed across samples."""
        decay = self.config.ofi_level_decay
        raw = 0.0
        denom = 0.0
        for j in range(max(len(top_bids), len(top_asks))):
            w = decay**j
            b = top_bids[j] if j < len(top_bids) else 0.0
            a = top_asks[j] if j < len(top_asks) else 0.0
            pb = self._prev_top_bids[j] if j < len(self._prev_top_bids) else b
            pa = self._prev_top_asks[j] if j < len(self._prev_top_asks) else a
            raw += w * ((b - pb) - (a - pa))
            denom += w * (b + a)
        if self._ml_window is not None:
            dt_s = clamped_dt_s(ts_event, self._ml_last_ts)
            alpha = ewma_alpha(dt_s, self.config.ml_ofi_half_life_ms / 1000.0)
            self._ml_ewma += alpha * (self._ml_window - self._ml_ewma)
        self._ml_window = raw
        self._ml_last_ts = ts_event
        self._prev_top_bids = top_bids
        self._prev_top_asks = top_asks
        return self._ml_ewma / denom if denom > 0 else 0.0

    def _composite_signal(self, ts_event: int) -> float | None:
        """Blend depth imbalance, best-quote OFI, multi-level OFI, trade flow."""
        cfg = self.config
        depth_imb = self._imbalance()
        if depth_imb is None:
            return None
        z = cfg.depth_weight * depth_imb

        if cfg.ofi_weight > 0:
            window_ofi = self._ofi_window
            window_depth = (
                self._ofi_depth_sum / self._ofi_depth_n
                if self._ofi_depth_n > 0
                else 0.0
            )
            dt_s = clamped_dt_s(ts_event, self._last_sample_ts)
            a_ofi = ewma_alpha(dt_s, cfg.ofi_half_life_ms / 1000.0)
            self._ofi_ewma += a_ofi * (window_ofi - self._ofi_ewma)
            if window_depth > 0:
                self._ofi_depth_ewma += a_ofi * (window_depth - self._ofi_depth_ewma)
            self._ofi_window = 0.0
            self._ofi_depth_sum = 0.0
            self._ofi_depth_n = 0
            if self._ofi_depth_ewma > 0:
                z += cfg.ofi_weight * (self._ofi_ewma / self._ofi_depth_ewma)

        m = max(1, min(cfg.ofi_levels, 25))
        top_bids = self._top_sizes(self._bids, m, True)
        top_asks = self._top_sizes(self._asks, m, False)
        if cfg.ml_ofi_weight > 0:
            z += cfg.ml_ofi_weight * self._multi_level_flow(ts_event, top_bids, top_asks)

        if cfg.trade_flow_weight > 0:
            dt_s = clamped_dt_s(ts_event, self._last_sample_ts)
            a_tf = ewma_alpha(dt_s, cfg.trade_half_life_ms / 1000.0)
            self._tf_signed_ewma += a_tf * (self._tf_signed_window - self._tf_signed_ewma)
            self._tf_abs_ewma += a_tf * (self._tf_abs_window - self._tf_abs_ewma)
            self._tf_signed_window = 0.0
            self._tf_abs_window = 0.0
            if self._tf_abs_ewma > 0:
                z += cfg.trade_flow_weight * (self._tf_signed_ewma / self._tf_abs_ewma)

        return z / self._weight_sum

    def _compute_signal(self, ts_event: int) -> float | None:
        if self._use_flow:
            return self._composite_signal(ts_event)
        return self._imbalance()

    def _flatten(self) -> None:
        super()._flatten()
        self._events_since_trade = 0
