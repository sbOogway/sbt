from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class ChaikinMoneyFlowConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 20
    threshold: float = 0.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class ChaikinMoneyFlow(SBTStrategy):
    def __init__(self, config: ChaikinMoneyFlowConfig) -> None:
        super().__init__(config)
        self._mfv: list[float] = []
        self._volumes: list[float] = []

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        volume = bar.volume.as_double() if hasattr(bar, "volume") else 1.0

        hl_range = high - low
        if hl_range == 0:
            mfm = 0.0
        else:
            mfm = ((close - low) - (high - close)) / hl_range

        self._mfv.append(mfm * volume)
        self._volumes.append(volume)

        if len(self._mfv) < self.config.period:
            return

        cmf = sum(self._mfv[-self.config.period :]) / max(
            sum(self._volumes[-self.config.period :]), 1e-10
        )

        if not self.in_position:
            if cmf > self.config.threshold:
                self.open_position(OrderSide.BUY, close)
            elif cmf < -self.config.threshold:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and cmf < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and cmf > 0:
                self.exit_market()
