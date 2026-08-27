from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class ElderRayConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 13

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class ElderRay(SBTStrategy):
    def __init__(self, config: ElderRayConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._prev_ema: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._closes.append(close)

        if len(self._closes) < self.config.period:
            return

        k = 2.0 / (self.config.period + 1)
        if self._prev_ema is None:
            ema = sum(self._closes[: self.config.period]) / self.config.period
            for p in self._closes[self.config.period :]:
                ema = p * k + ema * (1 - k)
        else:
            ema = close * k + self._prev_ema * (1 - k)
        self._prev_ema = ema

        bull_power = high - ema
        bear_power = low - ema

        if not self.in_position:
            if bull_power > 0 and bear_power > 0 and bear_power > -bear_power:
                self.open_position(OrderSide.BUY, close)
            elif bear_power < 0 and bull_power < 0 and bull_power > -bull_power:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and bear_power > 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and bull_power < 0:
                self.exit_market()
