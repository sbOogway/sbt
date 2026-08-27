from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class EnvelopeConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 20
    pct: float = 2.5

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class Envelope(SBTStrategy):
    def __init__(self, config: EnvelopeConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        if len(self._closes) < self.config.period:
            return

        sma = sum(self._closes[-self.config.period :]) / self.config.period
        upper = sma * (1 + self.config.pct / 100)
        lower = sma * (1 - self.config.pct / 100)

        if not self.in_position:
            if close > upper:
                self.open_position(OrderSide.BUY, close)
            elif close < lower:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and close < sma:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and close > sma:
                self.exit_market()
