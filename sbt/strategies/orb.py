from datetime import date
from decimal import Decimal

import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from ..plugins import PluginHost, SBTStrategyConfig


class ORBConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    capital: Decimal
    leverage: float

    backtest_start_date: str = "2020-01-01"

    orb_period: int = 12
    atr_period: int = 14
    atr_stop_multiple: float = 0.5
    risk_per_trade: float = 0.01

    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 30
    vol_max_scale: float = 3.0


class ORBStrategy(Strategy):
    def __init__(self, config: ORBConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type

        # Daily OHLC for ATR
        self._daily_high: float | None = None
        self._daily_low: float | None = None
        self._daily_close: float | None = None
        self._prev_daily_close: float | None = None
        self._true_ranges: list[float] = []
        self._atr_value: float = 0.0

        # ORB state
        self._current_day: date | None = None
        self._bar_index: int = 0
        self._first_bar_open: float | None = None
        self._first_bar_close: float | None = None
        self._orb_high: float | None = None
        self._orb_low: float | None = None
        self._orb_direction: str | None = None
        self._orb_ready: bool = False
        self._entry_triggered: bool = False

        # Position state
        self._in_position: bool = False
        self._position_side: OrderSide | None = None
        self._entry_price: float | None = None
        self._stop_price: float | None = None
        self._position_qty: Quantity | None = None

        # Volatility scaling via plugin (auto daily close-to-close tracking)
        self.plugins = PluginHost.from_config(config)

    def on_start(self) -> None:
        self.plugins.on_start(self)
        self.subscribe_bars(self.bar_type)

    def _calc_qty(self, price: float, stop_distance: float) -> Quantity | None:
        if stop_distance <= 0:
            return None
        weight = self.plugins.size_multiplier()
        risk_amount = (
            float(self.config.capital)
            * self.config.leverage
            * self.config.risk_per_trade
            * weight
        )
        size = risk_amount / stop_distance
        return Quantity(round(size, 3), precision=3)

    def _close_daily_bar(self) -> float | None:
        closed_daily_close = self._daily_close
        if (
            self._daily_high is not None
            and self._daily_low is not None
            and self._prev_daily_close is not None
        ):
            tr = max(
                self._daily_high - self._daily_low,
                abs(self._daily_high - self._prev_daily_close),
                abs(self._daily_low - self._prev_daily_close),
            )
            self._true_ranges.append(tr)
            if len(self._true_ranges) >= self.config.atr_period:
                self._atr_value = (
                    sum(self._true_ranges[-self.config.atr_period :])
                    / self.config.atr_period
                )
        self._prev_daily_close = self._daily_close
        self._daily_high = None
        self._daily_low = None
        self._daily_close = None
        return closed_daily_close

    def _reset_day(self) -> None:
        self._bar_index = 0
        self._first_bar_open = None
        self._first_bar_close = None
        self._orb_high = None
        self._orb_low = None
        self._orb_direction = None
        self._orb_ready = False
        self._entry_triggered = False

    def _start_new_day(self) -> None:
        if self._in_position:
            self._close_position()
        self._close_daily_bar()
        self._reset_day()

    def _accumulate_opening_range(self, bar: Bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()

        if self._orb_high is None or high > self._orb_high:
            self._orb_high = high
        if self._orb_low is None or low < self._orb_low:
            self._orb_low = low

        if self._bar_index == 1:
            self._first_bar_open = bar.open.as_double()
            self._first_bar_close = bar.close.as_double()

        if self._bar_index >= self.config.orb_period:
            if self._first_bar_close is not None and self._first_bar_open is not None:
                if self._first_bar_close > self._first_bar_open:
                    self._orb_direction = "long"
                elif self._first_bar_close < self._first_bar_open:
                    self._orb_direction = "short"
            self._orb_ready = True

    def _check_entry(self, bar: Bar) -> None:
        if self._entry_triggered or not self._orb_ready or self._orb_direction is None:
            return

        high = bar.high.as_double()
        low = bar.low.as_double()
        stop_distance = self.config.atr_stop_multiple * self._atr_value

        if self._orb_direction == "long" and high >= self._orb_high:
            qty = self._calc_qty(self._orb_high, stop_distance)
            if qty is not None:
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                )
                self.submit_order(order)
                self._in_position = True
                self._position_side = OrderSide.BUY
                self._entry_price = self._orb_high
                self._position_qty = qty
                self._stop_price = self._orb_high - stop_distance
                self._entry_triggered = True

        if self._orb_direction == "short" and low <= self._orb_low:
            qty = self._calc_qty(self._orb_low, stop_distance)
            if qty is not None:
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=qty,
                )
                self.submit_order(order)
                self._in_position = True
                self._position_side = OrderSide.SELL
                self._entry_price = self._orb_low
                self._position_qty = qty
                self._stop_price = self._orb_low + stop_distance
                self._entry_triggered = True

    def _check_stop(self, bar: Bar) -> None:
        if not self._in_position or self._stop_price is None:
            return
        low = bar.low.as_double()
        high = bar.high.as_double()
        if self._position_side == OrderSide.BUY and low <= self._stop_price or self._position_side == OrderSide.SELL and high >= self._stop_price:
            self._close_position()

    def _close_position(self) -> None:
        if (
            not self._in_position
            or self._position_side is None
            or self._position_qty is None
        ):
            return
        close_side = (
            OrderSide.SELL if self._position_side == OrderSide.BUY else OrderSide.BUY
        )
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=close_side,
            quantity=self._position_qty,
        )
        self.submit_order(order)
        self._in_position = False
        self._position_side = None
        self._entry_price = None
        self._stop_price = None
        self._position_qty = None

    def on_bar(self, bar: Bar) -> None:
        self.plugins.on_bar(self, bar)

        bar_ts = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        bar_date = bar_ts.date()

        if self._current_day is not None and bar_date != self._current_day:
            self._start_new_day()

        if self._current_day is None:
            self._current_day = bar_date

        self._current_day = bar_date

        self._bar_index += 1
        self._update_daily_ohlc(bar)

        is_weekend = bar_date.weekday() >= 5

        if not self._orb_ready and not is_weekend:
            self._accumulate_opening_range(bar)

        if (
            self._orb_ready
            and self._bar_index > self.config.orb_period
            and not is_weekend
        ):
            self._check_entry(bar)
            self._check_stop(bar)

    def _update_daily_ohlc(self, bar: Bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        if self._daily_high is None or high > self._daily_high:
            self._daily_high = high
        if self._daily_low is None or low < self._daily_low:
            self._daily_low = low
        self._daily_close = close
