from nautilus_trader.indicators import (
    BollingerBands,
    DonchianChannel,
    SimpleMovingAverage,
)
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Quantity

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class GlucksmannConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    bb_period: int = 20
    bb_std: float = 2.0

    vol_fast_period: int = 50
    vol_slow_period: int = 200

    sma_fast_period: int = 20
    sma_mid_period: int = 50
    sma_slow_period: int = 100
    sma_veryslow_period: int = 200

    vli_fast_period: int = 200
    vli_slow_period: int = 1000

    enable_short: bool = True

    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0
    # Preserve the original strategy behaviour: weight refreshes once a month.
    vol_rebalance_freq: str = "monthly"


class _RunningStats:
    def __init__(self, period: int) -> None:
        self.period = period
        self._values: list[float] = []

    def update(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self.period:
            self._values.pop(0)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean
        var = sum((x - m) ** 2 for x in self._values) / len(self._values)
        return var**0.5

    @property
    def initialized(self) -> bool:
        return len(self._values) == self.period


class GlucksmannStrategy(SBTStrategy):
    def __init__(self, config: GlucksmannConfig) -> None:
        super().__init__(config)

        self.bb = BollingerBands(config.bb_period, config.bb_std)
        self.sma_20 = SimpleMovingAverage(config.sma_fast_period)
        self.sma_50 = SimpleMovingAverage(config.sma_mid_period)
        self.sma_100 = SimpleMovingAverage(config.sma_slow_period)
        self.sma_200 = SimpleMovingAverage(config.sma_veryslow_period)

        self.sma_vol_fast = SimpleMovingAverage(config.vol_fast_period)
        self.sma_vol_slow = SimpleMovingAverage(config.vol_slow_period)

        self._bbw_sma_200 = SimpleMovingAverage(config.vli_fast_period)
        self._bbw_stats = _RunningStats(config.vli_slow_period)

        self.dc_20 = DonchianChannel(20)
        self.dc_10 = DonchianChannel(10)
        self.dc_5 = DonchianChannel(5)

        self._prev_close: float | None = None
        self._prev_bb_upper: float | None = None
        self._prev_bb_lower: float | None = None
        self._prev_sma_100: float | None = None
        self._prev_sma_50: float | None = None

        self._entry_price: float | None = None
        self._entry_candle_low: float | None = None
        self._entry_candle_high: float | None = None
        self._stop_price: float | None = None
        self._just_closed: bool = False

        self._extreme_entry: bool = False

        self._wait_for_short: bool = False
        self._need_long_entry: bool = False
        self._need_short_entry: bool = False

    def _calc_qty(self, price: float) -> Quantity | None:
        notional = (
            self.equity() * self.config.leverage * self.vol_multiplier()
        )
        return self.sized_quantity(notional / price)

    def _vol_ok(self) -> bool:
        return (
            self.sma_vol_fast.initialized
            and self.sma_vol_slow.initialized
            and self.sma_vol_fast.value > self.sma_vol_slow.value
        )

    def _bbw(self) -> float:
        mid = self.bb.middle
        return (self.bb.upper - self.bb.lower) / mid if mid != 0 else 0.0

    def _vli_fast(self) -> float:
        return self._bbw_sma_200.value if self._bbw_sma_200.initialized else 0.0

    def _vli_slow(self) -> float:
        return self._bbw_stats.mean

    def _vli_top(self) -> float:
        return self._bbw_stats.mean + 2.0 * self._bbw_stats.std

    def _low_vol(self) -> bool:
        return self._vli_fast() < self._vli_slow()

    def _bearish_aligned(self) -> bool:
        return self.sma_200.value > self.sma_100.value > self.sma_50.value

    def _smas_ready(self) -> bool:
        return all(
            (
                self.sma_20.initialized,
                self.sma_50.initialized,
                self.sma_100.initialized,
                self.sma_200.initialized,
            )
        )

    def _enter_long(self, bar, close: float) -> None:
        qty = self._calc_qty(close)
        if not self.enter_market(OrderSide.BUY, qty):
            return
        self._entry_price = close
        self._entry_candle_low = bar.low.as_double()
        self._entry_candle_high = bar.high.as_double()

        bbw = self._bbw()
        self._extreme_entry = bbw >= self._vli_top()
        if self._extreme_entry:
            # Thesis: "stoploss at: low of last candle"
            self._stop_price = bar.low.as_double()
        else:
            # Thesis: "stoploss: at 5% below low of entry candle"
            self._stop_price = bar.low.as_double() * 0.95

        self._wait_for_short = False

    def _enter_short(self, bar, close: float) -> None:
        qty = self._calc_qty(close)
        if not self.enter_market(OrderSide.SELL, qty):
            return
        self._entry_price = close
        self._entry_candle_high = bar.high.as_double()
        self._stop_price = self.dc_20.upper
        self._wait_for_short = False

    def _close_position(self) -> None:
        if self.exit_market():
            self._just_closed = True
            self._reset()

    def _reset(self) -> None:
        self._entry_price = None
        self._entry_candle_low = None
        self._entry_candle_high = None
        self._stop_price = None
        self._extreme_entry = False
        self._wait_for_short = False
        self._need_long_entry = False
        self._need_short_entry = False

    def _check_long_entry(self, close: float, crossdown_bb_top: bool) -> None:
        if not crossdown_bb_top:
            return
        if not self._vol_ok():
            return
        if close <= self.sma_20.value:
            return

        bbw = self._bbw()
        extreme = bbw >= self._vli_top()

        if not extreme:
            if self._low_vol():
                if self.sma_50.value <= self.sma_200.value:
                    return
            elif self._bearish_aligned():
                return
        elif self.sma_100.value <= self.sma_200.value:
            return

        self._need_long_entry = True

    def _check_short_entry(self, close: float, crossup_bb_bot: bool) -> None:
        if not self.config.enable_short:
            return
        if crossup_bb_bot and self._bbw() < self._vli_slow():
            self._wait_for_short = True
        if (
            self._wait_for_short
            and self._prev_sma_100 is not None
            and self._prev_sma_50 is not None
        ):
            sma100_crossdown_sma50 = (
                self._prev_sma_100 >= self._prev_sma_50
                and self.sma_100.value < self.sma_50.value
            )
            if sma100_crossdown_sma50 and self._bbw() < self._vli_top():
                self._need_short_entry = True

    def _update_long_stops(self, bar, close: float) -> None:
        low = bar.low.as_double()
        profit = (close - self._entry_price) / self._entry_price

        # Hard floor: 5% below entry candle low (applies to ALL entries)
        hard_floor = self._entry_candle_low * 0.95

        # Generic milestone stopwins (apply to ALL entries)
        stop = hard_floor
        for pct, trail in [
            (0.20, 0.15),
            (0.25, 0.20),
            (0.30, 0.25),
            (0.35, 0.30),
            (0.40, 0.35),
        ]:
            if profit >= pct:
                stop = max(stop, self._entry_price * (1 + trail))

        # Extreme entry: tight stop at low of last candle + 3%→1% stopwin
        if self._extreme_entry:
            tight_stop = self._entry_candle_low
            if profit >= 0.03:
                tight_stop = max(tight_stop, self._entry_price * 1.01)
            stop = max(stop, tight_stop)

        self._stop_price = stop

        if low <= self._stop_price:
            self._close_position()

    def _update_short_stops(self, bar, close: float) -> None:
        high = bar.high.as_double()

        profit = (self._entry_price - close) / self._entry_price

        stop = self.dc_20.upper
        if profit >= 0.10 and self.dc_10.initialized:
            stop = min(stop, self.dc_10.upper)
        if profit >= 0.15 and self.dc_5.initialized:
            stop = min(stop, self.dc_5.upper)

        for pct, trail in [(0.25, 0.20), (0.30, 0.25), (0.35, 0.30), (0.40, 0.35)]:
            if profit >= pct:
                stop = min(stop, self._entry_price * (1 - trail))

        self._stop_price = stop

        if high >= self._stop_price:
            self._close_position()

    def on_trading_bar(self, bar) -> None:
        self.bb.handle_bar(bar)
        self.sma_20.handle_bar(bar)
        self.sma_50.handle_bar(bar)
        self.sma_100.handle_bar(bar)
        self.sma_200.handle_bar(bar)

        self.dc_20.handle_bar(bar)
        self.dc_10.handle_bar(bar)
        self.dc_5.handle_bar(bar)

        vol = bar.volume.as_double()
        self.sma_vol_fast.update_raw(vol)
        self.sma_vol_slow.update_raw(vol)

        close = bar.close.as_double()

        if self.bb.initialized:
            bbw = self._bbw()
            self._bbw_sma_200.update_raw(bbw)
            self._bbw_stats.update(bbw)

        if self.in_position:
            if self.position_side == OrderSide.BUY:
                self._update_long_stops(bar, close)
            else:
                self._update_short_stops(bar, close)

        self._need_long_entry = False
        self._need_short_entry = False

        if (
            self._prev_close is not None
            and self.bb.initialized
            and self._prev_bb_upper is not None
            and self._prev_bb_lower is not None
        ):
            crossdown_bb_top = (
                self._prev_close >= self._prev_bb_upper and close < self.bb.upper
            )
            crossdown_bb_bot = (
                self._prev_close >= self._prev_bb_lower and close < self.bb.lower
            )
            crossup_bb_bot = (
                self._prev_close <= self._prev_bb_lower and close > self.bb.lower
            )

            if self.in_position and self.position_side == OrderSide.BUY:
                if crossdown_bb_bot and self._vol_ok():
                    self._close_position()

            if (
                not self.in_position
                and not self._just_closed
                and self._smas_ready()
                and self._bbw_sma_200.initialized
            ):
                self._check_long_entry(close, crossdown_bb_top)
                self._check_short_entry(close, crossup_bb_bot)

        self._prev_close = close
        if self.bb.initialized:
            self._prev_bb_upper = self.bb.upper
            self._prev_bb_lower = self.bb.lower
            self._prev_sma_100 = self.sma_100.value
            self._prev_sma_50 = self.sma_50.value

        self._just_closed = False

        if self._need_long_entry:
            self._enter_long(bar, close)
        elif self._need_short_entry:
            self._enter_short(bar, close)
