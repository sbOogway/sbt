from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class CoppockCurveConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    woa_period: int = 14
    momentum_period: int = 11
    ma_period: int = 10

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class CoppockCurve(SBTStrategy):
    def __init__(self, config: CoppockCurveConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._prev_wma: float | None = None

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        needed = self.config.woa_period + self.config.momentum_period + self.config.ma_period
        if len(self._closes) < needed:
            return

        mom1 = self._closes[-self.config.momentum_period] / self._closes[
            -self.config.momentum_period - self.config.woa_period
        ] - 1.0
        mom2 = self._closes[-1] / self._closes[-self.config.momentum_period - 1] - 1.0

        weights = list(range(1, self.config.ma_period + 1))
        wsum = sum(weights)
        wma = (mom1 * weights[0] + mom2 * weights[1]) / sum(weights[:2])

        prev = self._prev_wma
        self._prev_wma = wma

        if prev is None:
            return

        if not self.in_position:
            if prev < 0 and wma > 0:
                self.open_position(OrderSide.BUY, close)
        else:
            if wma < 0:
                self.exit_market()
