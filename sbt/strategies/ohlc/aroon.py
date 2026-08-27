from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class AroonConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 25
    threshold: float = 80.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class Aroon(SBTStrategy):
    def __init__(self, config: AroonConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)

        if len(self._highs) < self.config.period + 1:
            return

        window_highs = self._highs[-self.config.period - 1 :]
        window_lows = self._lows[-self.config.period - 1 :]

        periods_since_high = self.config.period
        for i, h in enumerate(window_highs[:-1]):
            if h == max(window_highs):
                periods_since_high = self.config.period - i - 1
                break

        periods_since_low = self.config.period
        for i, l in enumerate(window_lows[:-1]):
            if l == min(window_lows):
                periods_since_low = self.config.period - i - 1
                break

        aroon_up = (1 - periods_since_high / self.config.period) * 100
        aroon_down = (1 - periods_since_low / self.config.period) * 100

        if not self.in_position:
            if aroon_up > self.config.threshold and aroon_down < 100 - self.config.threshold:
                self.open_position(OrderSide.BUY, close)
            elif aroon_down > self.config.threshold and aroon_up < 100 - self.config.threshold:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and aroon_down > aroon_up:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and aroon_up > aroon_down:
                self.exit_market()
