from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class VolatilityBreakoutConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    atr_period: int = 14
    multiplier: float = 0.5

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class VolatilityBreakout(SBTStrategy):
    def __init__(self, config: VolatilityBreakoutConfig) -> None:
        super().__init__(config)
        self._true_ranges: list[float] = []
        self._prev_close: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        open_ = bar.open.as_double()

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
        threshold = self.config.multiplier * atr

        if not self.in_position:
            if close - open_ > threshold:
                self.open_position(OrderSide.BUY, close)
            elif open_ - close > threshold:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and close - open_ < -threshold / 2:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and open_ - close < -threshold / 2:
                self.exit_market()
