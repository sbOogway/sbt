from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class DonchianBreakoutConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    channel_period: int = 20
    exit_period: int = 10

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class DonchianBreakout(SBTStrategy):
    def __init__(self, config: DonchianBreakoutConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)

        if len(self._highs) < self.config.channel_period + 1:
            return

        upper = max(self._highs[-self.config.channel_period - 1 : -1])
        lower = min(self._lows[-self.config.channel_period - 1 : -1])

        if not self.in_position:
            if close > upper:
                self.open_position(OrderSide.BUY, close)
            elif close < lower:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY:
                exit_low = min(self._lows[-self.config.exit_period :])
                if close < exit_low:
                    self.exit_market()
            else:
                exit_high = max(self._highs[-self.config.exit_period :])
                if close > exit_high:
                    self.exit_market()
