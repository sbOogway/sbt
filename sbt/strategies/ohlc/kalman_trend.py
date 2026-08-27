from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class KalmanTrendConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    process_noise: float = 0.01
    measurement_noise: float = 1.0
    threshold: float = 0.001

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class KalmanTrend(SBTStrategy):
    def __init__(self, config: KalmanTrendConfig) -> None:
        super().__init__(config)
        self._q = config.process_noise
        self._r = config.measurement_noise
        self._x: float | None = None
        self._p: float = 1.0

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()

        if self._x is None:
            self._x = close
            return

        pred_x = self._x
        pred_p = self._p + self._q

        k = pred_p / (pred_p + self._r)
        self._x = pred_x + k * (close - pred_x)
        self._p = (1 - k) * pred_p

        slope = (self._x - pred_x) / close if close > 0 else 0.0

        if not self.in_position:
            if slope > self.config.threshold:
                self.open_position(OrderSide.BUY, close)
            elif slope < -self.config.threshold:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and slope < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and slope > 0:
                self.exit_market()
