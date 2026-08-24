from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class KeyBreakoutConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    enable_inside_breakout: bool = True
    enable_outside_breakout: bool = True
    enable_swing_breakout: bool = True
    swing_lookback: int = 20

    atr_period: int = 4
    atr_stop_multiple: float = 2.0
    max_holding_bars: int | None = None

    risk_per_trade: float = 0.01

    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class KeyBreakout(SBTStrategy):
    def __init__(self, config: KeyBreakoutConfig) -> None:
        super().__init__(config)
        self._atr_value: float = 0.0
        self._true_ranges: list[float] = []
        self._prev_close: float | None = None

        self._prev_high: float | None = None
        self._prev_low: float | None = None

        self._setup_high: float | None = None
        self._setup_low: float | None = None

        self._swing_highs: list[float] = []
        self._swing_lows: list[float] = []

        self._entry_price: float | None = None
        self._stop_price: float | None = None
        self._best_price: float | None = None
        self._bars_held: int = 0

    def _update_atr(self, high: float, low: float, close: float) -> None:
        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
            self._true_ranges.append(tr)
            if len(self._true_ranges) >= self.config.atr_period:
                self._atr_value = (
                    sum(self._true_ranges[-self.config.atr_period :])
                    / self.config.atr_period
                )
        self._prev_close = close

    def _enter(self, side: OrderSide, price: float) -> None:
        stop_distance = self.config.atr_stop_multiple * self._atr_value
        qty = self.risk_quantity(stop_distance, self.config.risk_per_trade)
        if not self.enter_market(side, qty):
            return
        self._entry_price = price
        self._best_price = price
        self._bars_held = 0
        self._stop_price = (
            price - stop_distance if side == OrderSide.BUY else price + stop_distance
        )

    def _close_position(self) -> None:
        if self.exit_market():
            self._entry_price = None
            self._stop_price = None
            self._best_price = None
            self._bars_held = 0

    def _manage_position(self, high: float, low: float) -> None:
        self._bars_held += 1
        stop_distance = self.config.atr_stop_multiple * self._atr_value
        if self.position_side == OrderSide.BUY:
            self._best_price = max(self._best_price, high)
            self._stop_price = max(self._stop_price, self._best_price - stop_distance)
            if low <= self._stop_price:
                self._close_position()
                return
        else:
            self._best_price = min(self._best_price, low)
            self._stop_price = min(self._stop_price, self._best_price + stop_distance)
            if high >= self._stop_price:
                self._close_position()
                return
        if (
            self.config.max_holding_bars is not None
            and self._bars_held >= self.config.max_holding_bars
        ):
            self._close_position()

    def _check_setup_entry(
        self, high: float, low: float, open_: float, close: float
    ) -> None:
        setup_high, setup_low = self._setup_high, self._setup_low
        self._setup_high = None
        self._setup_low = None
        if setup_high is None or setup_low is None:
            return
        broke_long = high >= setup_high
        broke_short = low <= setup_low
        if broke_long and broke_short:
            broke_long = close >= open_
            broke_short = not broke_long
        if broke_long:
            self._enter(OrderSide.BUY, setup_high)
        elif broke_short:
            self._enter(OrderSide.SELL, setup_low)

    def _check_swing_entry(self, close: float) -> None:
        lookback = self.config.swing_lookback
        if len(self._swing_highs) < lookback or len(self._swing_lows) < lookback:
            return
        if close > max(self._swing_highs[-lookback:]):
            self._enter(OrderSide.BUY, close)
        elif close < min(self._swing_lows[-lookback:]):
            self._enter(OrderSide.SELL, close)

    def _detect_setup(self, high: float, low: float) -> None:
        if self._prev_high is None or self._prev_low is None:
            return
        if high < self._prev_high and low > self._prev_low:
            if self.config.enable_inside_breakout:
                self._setup_high = high
                self._setup_low = low
        elif high > self._prev_high and low < self._prev_low:
            if self.config.enable_outside_breakout:
                self._setup_high = high
                self._setup_low = low

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        open_ = bar.open.as_double()

        self._update_atr(high, low, close)

        had_position = self.in_position
        if self.in_position:
            self._manage_position(high, low)
        exited_this_bar = had_position and not self.in_position

        if not self.in_position and not exited_this_bar and self._atr_value > 0:
            self._check_setup_entry(high, low, open_, close)
            if not self.in_position and self.config.enable_swing_breakout:
                self._check_swing_entry(close)

        self._detect_setup(high, low)

        self._swing_highs.append(high)
        self._swing_lows.append(low)
        if len(self._swing_highs) > self.config.swing_lookback:
            self._swing_highs.pop(0)
            self._swing_lows.pop(0)

        self._prev_high = high
        self._prev_low = low
