from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class RangeExpansionConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 12
    threshold: float = 1.5
    holding_bars: int = 6

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class RangeExpansion(SBTStrategy):
    def __init__(self, config: RangeExpansionConfig) -> None:
        super().__init__(config)
        self._ranges: list[float] = []
        self._bars_held: int = 0

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        bar_range = high - low
        self._ranges.append(bar_range)

        if self.in_position:
            self._bars_held += 1
            if self._bars_held >= self.config.holding_bars:
                self.exit_market()
                self._bars_held = 0
            return

        if len(self._ranges) < self.config.period + 1:
            return

        avg_range = sum(self._ranges[-self.config.period - 1 : -1]) / self.config.period
        if avg_range == 0:
            return

        expansion = bar_range / avg_range

        if expansion > self.config.threshold:
            if close > (high + low) / 2:
                self.open_position(OrderSide.BUY, close)
            else:
                self.open_position(OrderSide.SELL, close)
            self._bars_held = 0
