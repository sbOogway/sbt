from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class AdxTrendConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    adx_period: int = 14
    adx_threshold: float = 25.0
    ema_fast: int = 12
    ema_slow: int = 26

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class AdxTrend(SBTStrategy):
    def __init__(self, config: AdxTrendConfig) -> None:
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

    def _ema_val(self, period: int) -> float | None:
        if len(self._closes) < period:
            return None
        k = 2.0 / (period + 1)
        ema = sum(self._closes[:period]) / period
        for p in self._closes[period:]:
            ema = p * k + ema * (1 - k)
        return ema

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

        fast = self._ema_val(self.config.ema_fast)
        slow = self._ema_val(self.config.ema_slow)
        if fast is None or slow is None:
            return

        if not self.in_position:
            if self._adx > self.config.adx_threshold:
                if fast > slow:
                    self.open_position(OrderSide.BUY, close)
                else:
                    self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and fast < slow:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and fast > slow:
                self.exit_market()
