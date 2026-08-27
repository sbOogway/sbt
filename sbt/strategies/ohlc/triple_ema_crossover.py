from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class TripleEmaCrossoverConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    fast_period: int = 8
    mid_period: int = 21
    slow_period: int = 55

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class TripleEmaCrossover(SBTStrategy):
    def __init__(self, config: TripleEmaCrossoverConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []

    def _ema(self, period: int) -> float | None:
        if len(self._closes) < period:
            return None
        k = 2.0 / (period + 1)
        ema = sum(self._closes[:period]) / period
        for price in self._closes[period:]:
            ema = price * k + ema * (1 - k)
        return ema

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        fast = self._ema(self.config.fast_period)
        mid = self._ema(self.config.mid_period)
        slow = self._ema(self.config.slow_period)

        if fast is None or mid is None or slow is None:
            return

        if not self.in_position:
            if fast > mid > slow:
                self.open_position(OrderSide.BUY, close)
            elif fast < mid < slow:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and fast < mid:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and fast > mid:
                self.exit_market()
