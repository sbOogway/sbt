from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class WilliamsRConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 14
    oversold: float = -80.0
    overbought: float = -20.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class WilliamsR(SBTStrategy):
    def __init__(self, config: WilliamsRConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)

        if len(self._highs) < self.config.period:
            return

        hh = max(self._highs[-self.config.period :])
        ll = min(self._lows[-self.config.period :])
        if hh == ll:
            wr = -50.0
        else:
            wr = (hh - close) / (hh - ll) * -100

        if not self.in_position:
            if wr < self.config.oversold:
                self.open_position(OrderSide.BUY, close)
            elif wr > self.config.overbought:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and wr > -50:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and wr < -50:
                self.exit_market()
