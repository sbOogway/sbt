from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class ZscoreMeanReversionConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class ZscoreMeanReversion(SBTStrategy):
    def __init__(self, config: ZscoreMeanReversionConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        if len(self._closes) < self.config.lookback + 1:
            return

        window = self._closes[-self.config.lookback :]
        mean = sum(window) / len(window)
        std = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5
        if std <= 0:
            return

        z = (close - mean) / std

        if not self.in_position:
            if z < -self.config.entry_z:
                self.open_position(OrderSide.BUY, close)
            elif z > self.config.entry_z:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and z > -self.config.exit_z:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and z < self.config.exit_z:
                self.exit_market()
