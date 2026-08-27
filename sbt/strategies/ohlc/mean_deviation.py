from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class MeanDeviationConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 20
    dev_mult: float = 1.5

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class MeanDeviation(SBTStrategy):
    def __init__(self, config: MeanDeviationConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        if len(self._closes) < self.config.period:
            return

        window = self._closes[-self.config.period :]
        mean = sum(window) / len(window)
        mad = sum(abs(x - mean) for x in window) / len(window)

        upper = mean + self.config.dev_mult * mad
        lower = mean - self.config.dev_mult * mad

        if not self.in_position:
            if close < lower:
                self.open_position(OrderSide.BUY, close)
            elif close > upper:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and close > mean:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and close < mean:
                self.exit_market()
