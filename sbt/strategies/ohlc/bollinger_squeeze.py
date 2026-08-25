from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class BollingerSqueezeConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_atr_mult: float = 1.5
    atr_period: int = 20

    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class BollingerSqueeze(SBTStrategy):
    def __init__(self, config: BollingerSqueezeConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._true_ranges: list[float] = []
        self._prev_close: float | None = None
        self._in_squeeze: bool = False
        self._squeeze_just_fired: bool = False

    def _sma(self, data: list[float], period: int) -> float | None:
        if len(data) < period:
            return None
        return sum(data[-period:]) / period

    def _stdev(self, data: list[float], period: int) -> float | None:
        if len(data) < period:
            return None
        mean = sum(data[-period:]) / period
        variance = sum((x - mean) ** 2 for x in data[-period:]) / period
        return variance**0.5

    def _atr(self) -> float | None:
        period = self.config.atr_period
        if len(self._true_ranges) < period:
            return None
        return sum(self._true_ranges[-period:]) / period

    def _update_tr(self, high: float, low: float) -> None:
        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
            self._true_ranges.append(tr)
        self._prev_close = self._close if hasattr(self, "_close") else None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
            self._true_ranges.append(tr)
        self._prev_close = close

        period = self.config.bb_period
        if len(self._closes) < period:
            return

        bb_mid = self._sma(self._closes, period)
        bb_stdev = self._stdev(self._closes, period)
        atr = self._atr()

        if bb_mid is None or bb_stdev is None or atr is None:
            return

        bb_upper = bb_mid + self.config.bb_std * bb_stdev
        bb_lower = bb_mid - self.config.bb_std * bb_stdev

        kc_mid = self._sma(self._closes, self.config.kc_period)
        if kc_mid is None:
            return
        kc_upper = kc_mid + self.config.kc_atr_mult * atr
        kc_lower = kc_mid - self.config.kc_atr_mult * atr

        is_squeeze = bb_upper < kc_upper and bb_lower > kc_lower

        if self._in_squeeze and not is_squeeze:
            self._squeeze_just_fired = True
        else:
            self._squeeze_just_fired = False

        self._in_squeeze = is_squeeze

        if self.in_position:
            if self.position_side == OrderSide.BUY and close < bb_mid:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and close > bb_mid:
                self.exit_market()
            return

        if self._squeeze_just_fired:
            if close > bb_upper:
                self.open_position(OrderSide.BUY, close)
            elif close < bb_lower:
                self.open_position(OrderSide.SELL, close)
        elif self._in_squeeze:
            if close > bb_upper:
                self.open_position(OrderSide.BUY, close)
            elif close < bb_lower:
                self.open_position(OrderSide.SELL, close)
