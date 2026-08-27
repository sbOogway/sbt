from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class TrendStrengthConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 20
    adx_period: int = 14
    adx_min: float = 25.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class TrendStrength(SBTStrategy):
    def __init__(self, config: TrendStrengthConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._prev_close: float | None = None
        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._plus_dm: float = 0.0
        self._minus_dm: float = 0.0
        self._atr: float = 0.0
        self._adx: float = 0.0

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        if self._prev_high is not None:
            up = high - self._prev_high
            down = self._prev_low - low
            plus_dm = up if up > down and up > 0 else 0
            minus_dm = down if down > up and down > 0 else 0
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))

            n = self.config.adx_period
            if len(self._closes) == n + 2:
                self._plus_dm = plus_dm
                self._minus_dm = minus_dm
                self._atr = tr
            else:
                self._plus_dm = (self._plus_dm * (n - 1) + plus_dm) / n
                self._minus_dm = (self._minus_dm * (n - 1) + minus_dm) / n
                self._atr = (self._atr * (n - 1) + tr) / n

            if self._atr > 0:
                plus_di = self._plus_dm / self._atr * 100
                minus_di = self._minus_dm / self._atr * 100
                di_sum = plus_di + minus_di
                if di_sum > 0:
                    dx = abs(plus_di - minus_di) / di_sum * 100
                    if len(self._closes) == n + 2:
                        self._adx = dx
                    else:
                        self._adx = (self._adx * (n - 1) + dx) / n

        self._prev_close = close
        self._prev_high = high
        self._prev_low = low

        if len(self._closes) < self.config.period:
            return

        ret = close / self._closes[-self.config.period] - 1.0

        if not self.in_position:
            if self._adx > self.config.adx_min:
                if ret > 0:
                    self.open_position(OrderSide.BUY, close)
                else:
                    self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and ret < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and ret > 0:
                self.exit_market()
