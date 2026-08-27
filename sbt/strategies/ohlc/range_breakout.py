from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class RangeBreakoutConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    range_bars: int = 6
    holding_bars: int = 12

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class RangeBreakout(SBTStrategy):
    def __init__(self, config: RangeBreakoutConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._bars_held: int = 0

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)

        if self.in_position:
            self._bars_held += 1
            if self._bars_held >= self.config.holding_bars:
                self.exit_market()
                self._bars_held = 0
            return

        n = self.config.range_bars
        if len(self._highs) < n + 1:
            return

        range_high = max(self._highs[-n - 1 : -1])
        range_low = min(self._lows[-n - 1 : -1])

        if close > range_high:
            self.open_position(OrderSide.BUY, close)
            self._bars_held = 0
        elif close < range_low:
            self.open_position(OrderSide.SELL, close)
            self._bars_held = 0
