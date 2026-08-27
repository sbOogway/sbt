from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class TrueRangeBreakoutConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    atr_period: int = 14
    breakout_mult: float = 1.5
    holding_bars: int = 10

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class TrueRangeBreakout(SBTStrategy):
    def __init__(self, config: TrueRangeBreakoutConfig) -> None:
        super().__init__(config)
        self._true_ranges: list[float] = []
        self._prev_close: float | None = None
        self._prev2_close: float | None = None
        self._bars_held: int = 0

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
        self._prev2_close = self._prev_close
        self._prev_close = close

        if len(self._true_ranges) < self.config.atr_period:
            return

        atr = sum(self._true_ranges[-self.config.atr_period :]) / self.config.atr_period
        threshold = self.config.breakout_mult * atr

        if self.in_position:
            self._bars_held += 1
            if self._bars_held >= self.config.holding_bars:
                self.exit_market()
                self._bars_held = 0
            return

        if self._prev2_close is not None:
            if close - self._prev2_close > threshold:
                self.open_position(OrderSide.BUY, close)
                self._bars_held = 0
            elif self._prev2_close - close > threshold:
                self.open_position(OrderSide.SELL, close)
                self._bars_held = 0
