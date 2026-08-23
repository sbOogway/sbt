"""Layer 2 Order Book Imbalance & Microstructure Strategy."""

from decimal import Decimal
from nautilus_trader.model.data import OrderBookDelta, OrderBookDeltas, TradeTick
from nautilus_trader.model.enums import BookAction, BookType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from ..plugins import SBTStrategyConfig


class L2OrderImbalanceConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Configuration for L2 Order Book Imbalance Strategy."""

    instrument_id: InstrumentId
    capital: Decimal = Decimal("1000")
    leverage: float = 1.0
    backtest_start_date: str = "2020-01-01"

    imbalance_threshold: float = 0.6  # Imbalance threshold: (bid - ask) / (bid + ask)
    cooldown_events: int = 50
    # Fraction of current account equity (times leverage) traded as notional
    # per entry, e.g. 0.02 -> $20 notional on a $1000 account at 1x.
    capital_fraction: float = 0.02
    # Hysteresis exit: flatten when |imbalance| decays below this (0 disables;
    # legacy behavior flips only on a full opposite entry signal).
    exit_threshold: float = 0.0
    # Force-flat time stop in seconds (0 disables).
    max_hold_seconds: int = 0
    # Depth (price levels per side) used for the imbalance measurement.
    top_levels: int = 5


class L2OrderImbalance(Strategy):
    """Exploits short-term microstructural order book depth imbalance."""

    def __init__(self, config: L2OrderImbalanceConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self._bids: dict[float, float] = {}  # price -> size
        self._asks: dict[float, float] = {}  # price -> size
        self._events_since_trade: int = 0
        self._last_side: OrderSide | None = None
        self._entry_ts: int | None = None

    def on_start(self) -> None:
        self.subscribe_order_book_deltas(
            self.instrument_id,
            book_type=BookType.L2_MBP,
        )
        self.subscribe_trade_ticks(self.instrument_id)

    def on_order_book_deltas(self, deltas: OrderBookDeltas) -> None:
        """Maintain top of book depth from order book deltas."""
        for delta in deltas.deltas:
            self._process_delta(delta)

    def _process_delta(self, delta: OrderBookDelta) -> None:
        if delta.order is None:
            return

        price = float(delta.order.price)
        size = float(delta.order.size)
        side = delta.order.side
        action = delta.action

        book = self._bids if side == OrderSide.BUY else self._asks

        if action == BookAction.ADD or action == BookAction.UPDATE:
            book[price] = size
        elif action == BookAction.DELETE:
            book.pop(price, None)
        elif action == BookAction.CLEAR:
            book.clear()

        self._events_since_trade += 1
        if self._events_since_trade >= self.config.cooldown_events:
            self._check_signal(int(delta.ts_event))

    def on_trade_tick(self, tick: TradeTick) -> None:
        """Process real-time matched trade prints."""
        pass

    def _imbalance(self) -> float | None:
        n = self.config.top_levels
        top_bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)[:n]
        top_asks = sorted(self._asks.items(), key=lambda x: x[0])[:n]

        bid_vol = sum(v for _, v in top_bids)
        ask_vol = sum(v for _, v in top_asks)

        total_vol = bid_vol + ask_vol
        if total_vol <= 0:
            return None
        return (bid_vol - ask_vol) / total_vol

    def _check_signal(self, ts_event: int) -> None:
        if not self._bids or not self._asks:
            return

        imbalance = self._imbalance()
        if imbalance is None:
            return

        # Time stop
        if (
            self.config.max_hold_seconds > 0
            and self._entry_ts is not None
            and ts_event - self._entry_ts > self.config.max_hold_seconds * 1_000_000_000
        ):
            self._flatten()
            return

        # Hysteresis exit: decay against the position beyond exit_threshold.
        if self.config.exit_threshold > 0.0:
            if self._last_side == OrderSide.BUY and imbalance < self.config.exit_threshold:
                self._flatten()
                return
            if self._last_side == OrderSide.SELL and imbalance > -self.config.exit_threshold:
                self._flatten()
                return

        # Strong bid volume dominance -> Long
        if imbalance > self.config.imbalance_threshold:
            if self._last_side != OrderSide.BUY:
                self._submit_market(OrderSide.BUY)
                self._last_side = OrderSide.BUY
                self._entry_ts = ts_event
                self._events_since_trade = 0
        # Strong ask volume dominance -> Short
        elif imbalance < -self.config.imbalance_threshold:
            if self._last_side != OrderSide.SELL:
                self._submit_market(OrderSide.SELL)
                self._last_side = OrderSide.SELL
                self._entry_ts = ts_event
                self._events_since_trade = 0

    def _flatten(self) -> None:
        open_positions = self.cache.positions(
            venue=self.instrument_id.venue,
            instrument_id=self.instrument_id,
        )
        position = next((p for p in open_positions if p.is_open), None)
        if position is None or position.quantity.as_double() <= 0:
            self._last_side = None
            self._entry_ts = None
            self._events_since_trade = 0
            return

        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY
        self._submit_market(close_side, quantity=position.quantity)
        self._last_side = None
        self._entry_ts = None
        self._events_since_trade = 0

    def _submit_market(
        self, order_side: OrderSide, quantity=None
    ) -> None:
        instrument = self.cache.instrument(self.instrument_id)
        ref_price = self._mid_price()
        if ref_price is None:
            return

        if quantity is None:
            notional = (
                self._equity(instrument)
                * self.config.leverage
                * self.config.capital_fraction
            )
            quantity = instrument.make_qty(Decimal(str(notional / ref_price)))

        if quantity.as_double() <= 0:
            return

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=quantity,
        )
        self.submit_order(order)

    def _mid_price(self) -> float | None:
        if not self._bids or not self._asks:
            return None
        return (max(self._bids) + min(self._asks)) / 2

    def _equity(self, instrument) -> float:
        account = self.portfolio.account(self.instrument_id.venue)
        if account is not None:
            money = account.balance_total(instrument.quote_currency)
            if money is not None:
                return float(money)
        return float(self.config.capital)
