from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class DualSmaCrossoverConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    short_period: int = 10
    long_period: int = 40

    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class DualSmaCrossover(SBTStrategy):
    def __init__(self, config: DualSmaCrossoverConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._prev_short_ma: float | None = None
        self._prev_long_ma: float | None = None

    def _sma(self, period: int) -> float | None:
        if len(self._closes) < period:
            return None
        return sum(self._closes[-period:]) / period

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        short_ma = self._sma(self.config.short_period)
        long_ma = self._sma(self.config.long_period)

        if short_ma is None or long_ma is None:
            self._prev_short_ma = short_ma
            self._prev_long_ma = long_ma
            return

        prev_short = self._prev_short_ma
        prev_long = self._prev_long_ma
        self._prev_short_ma = short_ma
        self._prev_long_ma = long_ma

        if prev_short is None or prev_long is None:
            return

        if not self.in_position:
            if prev_short <= prev_long and short_ma > long_ma:
                self.open_position(OrderSide.BUY, close)
            elif prev_short >= prev_long and short_ma < long_ma:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and short_ma < long_ma:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and short_ma > long_ma:
                self.exit_market()
