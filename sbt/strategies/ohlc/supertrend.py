from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class SupertrendConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    atr_period: int = 10
    multiplier: float = 3.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class Supertrend(SBTStrategy):
    def __init__(self, config: SupertrendConfig) -> None:
        super().__init__(config)
        self._true_ranges: list[float] = []
        self._prev_close: float | None = None
        self._prev_upper: float | None = None
        self._prev_lower: float | None = None
        self._prev_direction: int = 1

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
            self._true_ranges.append(tr)
        self._prev_close = close

        if len(self._true_ranges) < self.config.atr_period:
            return

        atr = sum(self._true_ranges[-self.config.atr_period :]) / self.config.atr_period
        hl2 = (high + low) / 2.0

        basic_upper = hl2 + self.config.multiplier * atr
        basic_lower = hl2 - self.config.multiplier * atr

        upper = basic_upper
        lower = basic_lower

        if self._prev_upper is not None and close <= self._prev_upper:
            upper = min(basic_upper, self._prev_upper)
        if self._prev_lower is not None and close >= self._prev_lower:
            lower = max(basic_lower, self._prev_lower)

        direction = self._prev_direction
        if self._prev_direction == 1:
            if close < lower:
                direction = -1
        else:
            if close > upper:
                direction = 1

        self._prev_upper = upper
        self._prev_lower = lower

        prev_dir = self._prev_direction
        self._prev_direction = direction

        if not self.in_position:
            if direction == 1 and prev_dir == -1:
                self.open_position(OrderSide.BUY, close)
            elif direction == -1 and prev_dir == 1:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and direction == -1:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and direction == 1:
                self.exit_market()
