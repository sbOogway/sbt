from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class CciConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 20
    upper: float = 100.0
    lower: float = -100.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class Cci(SBTStrategy):
    def __init__(self, config: CciConfig) -> None:
        super().__init__(config)
        self._typicals: list[float] = []
        self._prev_cci: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        tp = (high + low + close) / 3.0
        self._typicals.append(tp)

        if len(self._typicals) < self.config.period:
            return

        window = self._typicals[-self.config.period :]
        sma = sum(window) / len(window)
        mad = sum(abs(x - sma) for x in window) / len(window)
        if mad == 0:
            return

        cci = (tp - sma) / (0.015 * mad)

        prev = self._prev_cci
        self._prev_cci = cci

        if prev is None:
            return

        if not self.in_position:
            if prev < self.config.lower and cci > self.config.lower:
                self.open_position(OrderSide.BUY, close)
            elif prev > self.config.upper and cci < self.config.upper:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and cci > 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and cci < 0:
                self.exit_market()
