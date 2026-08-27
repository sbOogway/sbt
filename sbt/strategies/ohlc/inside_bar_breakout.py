from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class InsideBarBreakoutConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    holding_bars: int = 5

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class InsideBarBreakout(SBTStrategy):
    def __init__(self, config: InsideBarBreakoutConfig) -> None:
        super().__init__(config)
        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._mother_high: float | None = None
        self._mother_low: float | None = None
        self._bars_held: int = 0

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        if self._prev_high is not None:
            is_inside = high <= self._prev_high and low >= self._prev_low
            if is_inside:
                self._mother_high = self._prev_high
                self._mother_low = self._prev_low
            elif self._mother_high is not None:
                if high > self._mother_high and not self.in_position:
                    self.open_position(OrderSide.BUY, close)
                    self._bars_held = 0
                elif low < self._mother_low and not self.in_position:
                    self.open_position(OrderSide.SELL, close)
                    self._bars_held = 0
                if self.in_position:
                    self._bars_held += 1
                    if self._bars_held >= self.config.holding_bars:
                        self.exit_market()
                        self._bars_held = 0

        self._prev_high = high
        self._prev_low = low
