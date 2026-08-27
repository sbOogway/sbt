from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class TripleMaConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    fast: int = 5
    medium: int = 15
    slow: int = 30

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class TripleMa(SBTStrategy):
    def __init__(self, config: TripleMaConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []

    def _sma(self, period: int) -> float | None:
        if len(self._closes) < period:
            return None
        return sum(self._closes[-period:]) / period

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        f = self._sma(self.config.fast)
        m = self._sma(self.config.medium)
        s = self._sma(self.config.slow)

        if f is None or m is None or s is None:
            return

        if not self.in_position:
            if f > m > s:
                self.open_position(OrderSide.BUY, close)
            elif f < m < s:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and f < m:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and f > m:
                self.exit_market()
