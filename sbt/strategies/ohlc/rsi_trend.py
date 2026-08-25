from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class RsiTrendConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    rsi_period: int = 14
    trend_level: float = 50.0

    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class RsiTrend(SBTStrategy):
    def __init__(self, config: RsiTrendConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._prev_rsi: float | None = None
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._initialized: bool = False

    def _compute_rsi(self) -> float | None:
        period = self.config.rsi_period
        if len(self._closes) < period + 1:
            return None

        if not self._initialized:
            gains = []
            losses = []
            for i in range(1, period + 1):
                delta = self._closes[i] - self._closes[i - 1]
                gains.append(max(delta, 0.0))
                losses.append(max(-delta, 0.0))
            self._avg_gain = sum(gains) / period
            self._avg_loss = sum(losses) / period
            self._initialized = True
        else:
            delta = self._closes[-1] - self._closes[-2]
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            self._avg_gain = (self._avg_gain * (period - 1) + gain) / period
            self._avg_loss = (self._avg_loss * (period - 1) + loss) / period

        if self._avg_loss == 0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        rsi = self._compute_rsi()
        if rsi is None:
            return

        prev_rsi = self._prev_rsi
        self._prev_rsi = rsi

        if prev_rsi is None:
            return

        if not self.in_position:
            if prev_rsi <= self.config.trend_level and rsi > self.config.trend_level:
                self.open_position(OrderSide.BUY, close)
            elif prev_rsi >= self.config.trend_level and rsi < self.config.trend_level:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and rsi < self.config.trend_level:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and rsi > self.config.trend_level:
                self.exit_market()
