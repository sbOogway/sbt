from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class T3Config(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 5
    v_factor: float = 0.7

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class T3(SBTStrategy):
    def __init__(self, config: T3Config) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._e1: float | None = None
        self._e2: float | None = None
        self._e3: float | None = None
        self._e4: float | None = None
        self._e5: float | None = None
        self._e6: float | None = None

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        if len(self._closes) < self.config.period:
            return

        k = 2.0 / (self.config.period + 1)
        v = self.config.v_factor

        def ema_val(prev: float | None, source: float) -> float:
            if prev is None:
                return source
            return source * k + prev * (1 - k)

        self._e1 = ema_val(self._e1, close)
        self._e2 = ema_val(self._e2, self._e1)
        self._e3 = ema_val(self._e3, self._e2)
        self._e4 = ema_val(self._e4, self._e3)
        self._e5 = ema_val(self._e5, self._e4)
        self._e6 = ema_val(self._e6, self._e5)

        if None in (self._e1, self._e6):
            return

        c1 = -(v**3)
        c2 = 3 * v**2 + 3 * v**3
        c3 = -6 * v**2 - 3 * v - 3 * v**3
        c4 = 1 + 3 * v + v**3 + 3 * v**2

        t3 = c1 * self._e6 + c2 * self._e5 + c3 * self._e4 + c4 * self._e3

        if len(self._closes) < 2:
            return
        prev_close = self._closes[-2]

        if not self.in_position:
            if prev_close <= t3 and close > t3:
                self.open_position(OrderSide.BUY, close)
            elif prev_close >= t3 and close < t3:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and close < t3:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and close > t3:
                self.exit_market()
