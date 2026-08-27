from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class MassIndexConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 9
    ema_period: int = 25
    sum_threshold: float = 27.0
    unroll_threshold: float = 26.5

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class MassIndex(SBTStrategy):
    def __init__(self, config: MassIndexConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._ratios: list[float] = []
        self._ema: float | None = None
        self._prev_ratio_sum: float = 0.0

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)

        hl = high - low
        k = 2.0 / (self.config.ema_period + 1)

        if self._ema is None:
            if len(self._highs) >= self.config.ema_period:
                self._ema = sum(
                    self._highs[i] - self._lows[i]
                    for i in range(len(self._highs) - self.config.ema_period, len(self._highs))
                ) / self.config.ema_period
            else:
                return
        else:
            self._ema = hl * k + self._ema * (1 - k)

        ratio = hl / self._ema if self._ema > 0 else 1.0
        self._ratios.append(ratio)

        if len(self._ratios) < self.config.period + 1:
            return

        ratio_sum = sum(self._ratios[-self.config.period :])
        prev_sum = sum(self._ratios[-self.config.period - 1 : -1])

        if not self.in_position:
            if prev_sum >= self.config.sum_threshold and ratio_sum < self.config.unroll_threshold:
                self.open_position(OrderSide.BUY, close)
        else:
            if ratio_sum > self.config.sum_threshold:
                self.exit_market()
