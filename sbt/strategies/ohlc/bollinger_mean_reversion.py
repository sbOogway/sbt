from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class BollingerMeanReversionConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 20
    std_mult: float = 2.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class BollingerMeanReversion(SBTStrategy):
    def __init__(self, config: BollingerMeanReversionConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        if len(self._closes) < self.config.period:
            return

        window = self._closes[-self.config.period :]
        mean = sum(window) / len(window)
        std = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5

        upper = mean + self.config.std_mult * std
        lower = mean - self.config.std_mult * std

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
