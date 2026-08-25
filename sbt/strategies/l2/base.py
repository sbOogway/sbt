"""Shared plumbing for event-driven L2 order book strategies.

Maintains the full L2 book from OrderBookDelta events, evaluates a
strategy-defined signal on a fixed time grid, and executes market
entries/exits with equity-fraction sizing. Subclasses implement
``_compute_signal`` and declare their own thresholds on their Config.
"""

import math
from decimal import Decimal

from nautilus_trader.model.data import OrderBookDelta, OrderBookDeltas
from nautilus_trader.model.enums import BookAction, BookType, OrderSide
from nautilus_trader.trading.strategy import Strategy


def clamped_dt_s(ts_event: int, last_ts: int | None, cap_s: float = 60.0) -> float:
    """Seconds since ``last_ts``, clamped to [0, cap] (no last_ts -> 0)."""
    if last_ts is None:
        return 0.0
    return min(max((ts_event - last_ts) / 1e9, 0.0), cap_s)


def ewma_alpha(dt_s: float, half_life_s: float) -> float:
    """Per-step EWMA smoothing weight for exponential decay (half-life).

    1.0 snaps the average to the latest observation (first step or
    non-positive half-life); otherwise the classic
    ``1 - exp(-ln(2) * dt / half_life)``.
    """
    if half_life_s <= 0.0 or dt_s <= 0.0:
        return 1.0
    return 1.0 - math.exp(-math.log(2.0) * dt_s / half_life_s)


class L2EventStrategy(Strategy):
    """Base class for L2 order-book-driven strategies."""

    needs_trade_ticks: bool = False

    def __init__(self, config) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._last_sample_ts: int | None = None
        self._entry_ts: int | None = None
        self._last_side: OrderSide | None = None
        self._interval_ns = int(getattr(config, "signal_interval_ms", 500)) * 1_000_000

    def on_start(self) -> None:
        self.subscribe_order_book_deltas(
            self.instrument_id,
            book_type=BookType.L2_MBP,
        )
        if self.needs_trade_ticks:
            self.subscribe_trade_ticks(self.instrument_id)

    def on_order_book_deltas(self, deltas: OrderBookDeltas) -> None:
        for delta in deltas.deltas:
            self._process_delta(delta)

    def _process_delta(self, delta: OrderBookDelta) -> None:
        if delta.order is None:
            return
        price = float(delta.order.price)
        size = float(delta.order.size)
        book = self._bids if delta.order.side == OrderSide.BUY else self._asks
        if delta.action == BookAction.ADD or delta.action == BookAction.UPDATE:
            book[price] = size
        elif delta.action == BookAction.DELETE:
            book.pop(price, None)
        elif delta.action == BookAction.CLEAR:
            book.clear()
        self._on_order_event(delta, price, size)
        ts_event = int(delta.ts_event)
        if self._sample_due(ts_event):
            self._check_signal(ts_event)

    def _sample_due(self, ts_event: int) -> bool:
        """Sampling-grid hook: True when a signal evaluation is due.

        Default is the fixed time grid (``signal_interval_ms``);
        subclasses may override for e.g. event-count sampling.
        """
        return (
            self._last_sample_ts is None
            or ts_event - self._last_sample_ts >= self._interval_ns
        )

    def _on_order_event(self, delta: OrderBookDelta, price: float, size: float) -> None:
        """Hook for subclasses accumulating per-event flow."""

    def _compute_signal(self, ts_event: int) -> float | None:
        raise NotImplementedError

    def _check_signal(self, ts_event: int) -> None:
        if not self._bids or not self._asks:
            return
        signal = self._compute_signal(ts_event)
        if signal is None:
            return
        self._last_sample_ts = ts_event

        if self.config.max_hold_seconds > 0 and self._entry_ts is not None:
            if ts_event - self._entry_ts > self.config.max_hold_seconds * 1_000_000_000:
                self._flatten()
                return

        exit_th = float(getattr(self.config, "exit_threshold", 0.0))
        if exit_th > 0.0:
            if self._last_side == OrderSide.BUY and signal < exit_th:
                self._flatten()
                return
            if self._last_side == OrderSide.SELL and signal > -exit_th:
                self._flatten()
                return

        max_spread = float(getattr(self.config, "max_spread_bps", 0.0))
        if self._last_side is None and max_spread > 0.0:
            spread = self._spread_bps()
            if spread is not None and spread > max_spread:
                return

        th = float(self.config.entry_threshold)
        if signal > th:
            if self._last_side != OrderSide.BUY:
                self._submit_market(OrderSide.BUY)
                self._last_side = OrderSide.BUY
                self._entry_ts = ts_event
        elif signal < -th:
            if self._last_side != OrderSide.SELL:
                self._submit_market(OrderSide.SELL)
                self._last_side = OrderSide.SELL
                self._entry_ts = ts_event

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mid_price(self) -> float | None:
        if not self._bids or not self._asks:
            return None
        return (max(self._bids) + min(self._asks)) / 2.0

    def _spread_bps(self) -> float | None:
        mid = self._mid_price()
        if mid is None or mid <= 0:
            return None
        return (min(self._asks) - max(self._bids)) / mid * 10000.0

    def _top_sizes(self, book: dict[float, float], n: int, is_bid: bool) -> list[float]:
        if is_bid:
            return sorted(book.values(), reverse=True)[:n]
        return sorted(book.values())[:n]

    def _equity(self, instrument) -> float:
        account = self.portfolio.account(self.instrument_id.venue)
        if account is not None:
            money = account.balance_total(instrument.quote_currency)
            if money is not None:
                return float(money)
        return float(self.config.capital)

    def _flatten(self) -> None:
        open_positions = self.cache.positions(
            venue=self.instrument_id.venue,
            instrument_id=self.instrument_id,
        )
        position = next((p for p in open_positions if p.is_open), None)
        if position is None or position.quantity.as_double() <= 0:
            self._last_side = None
            self._entry_ts = None
            return
        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY
        self._submit_market(close_side, quantity=position.quantity)
        self._last_side = None
        self._entry_ts = None

    def _submit_market(self, order_side: OrderSide, quantity=None) -> None:
        instrument = self.cache.instrument(self.instrument_id)
        ref_price = self._mid_price()
        if ref_price is None:
            return
        if quantity is None:
            notional = (
                self._equity(instrument)
                * float(self.config.leverage)
                * float(self.config.capital_fraction)
            )
            quantity = instrument.make_qty(Decimal(str(notional / ref_price)))
            if quantity.as_double() <= 0:
                quantity = instrument.size_increment
        if quantity.as_double() <= 0:
            return
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=quantity,
        )
        self.submit_order(order)
