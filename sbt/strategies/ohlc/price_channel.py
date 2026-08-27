from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class PriceChannelConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 20

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class PriceChannel(SBTStrategy):
    def __init__(self, config: PriceChannelConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._prev_mid: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)

        if len(self._highs) < self.config.period:
            return

        upper = max(self._highs[-self.config.period :])
        lower = min(self._lows[-self.config.period :])
        mid = (upper + lower) / 2

        prev_mid = self._prev_mid
        self._prev_mid = mid

        if prev_mid is None:
            return

        if not self.in_position:
            if close > upper:
                self.open_position(OrderSide.BUY, close)
            elif close < lower:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and close < mid:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and close > mid:
                self.exit_market()
