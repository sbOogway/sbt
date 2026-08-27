from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class VortexIndicatorConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 14

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class VortexIndicator(SBTStrategy):
    def __init__(self, config: VortexIndicatorConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []
        self._prev_close: float | None = None
        self._prev_high: float | None = None
        self._prev_low: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        if self._prev_close is None or len(self._closes) < self.config.period + 1:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return

        tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        vm_plus = abs(high - self._prev_low)
        vm_minus = abs(low - self._prev_high)

        atr_sum = sum(
            max(
                self._highs[-i] - self._lows[-i],
                abs(self._highs[-i] - self._closes[-i - 1]),
                abs(self._lows[-i] - self._closes[-i - 1]),
            )
            for i in range(1, self.config.period + 1)
        ) / self.config.period

        if atr_sum == 0:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return

        vi_plus = vm_plus / atr_sum
        vi_minus = vm_minus / atr_sum

        if not self.in_position:
            if vi_plus > 1.0 and vi_minus < 1.0:
                self.open_position(OrderSide.BUY, close)
            elif vi_minus > 1.0 and vi_plus < 1.0:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and vi_minus > vi_plus:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and vi_plus > vi_minus:
                self.exit_market()

        self._prev_close = close
        self._prev_high = high
        self._prev_low = low
