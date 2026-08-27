from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class TrendFilterConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    fast_ma: int = 10
    slow_ma: int = 200
    filter_period: int = 50

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class TrendFilter(SBTStrategy):
    def __init__(self, config: TrendFilterConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []

    def _sma(self, period: int) -> float | None:
        if len(self._closes) < period:
            return None
        return sum(self._closes[-period:]) / period

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        fast = self._sma(self.config.fast_ma)
        slow = self._sma(self.config.slow_ma)

        if fast is None or slow is None:
            return

        filter_ma = self._sma(self.config.filter_period)
        if filter_ma is None:
            return

        if not self.in_position:
            if close > filter_ma and fast > slow:
                self.open_position(OrderSide.BUY, close)
            elif close < filter_ma and fast < slow:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and (fast < slow or close < filter_ma):
                self.exit_market()
            elif self.position_side == OrderSide.SELL and (fast > slow or close > filter_ma):
                self.exit_market()
