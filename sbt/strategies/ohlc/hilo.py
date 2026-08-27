from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class HiloConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 13

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class Hilo(SBTStrategy):
    def __init__(self, config: HiloConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._hilo_line: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)

        if len(self._highs) < self.config.period:
            return

        hh = max(self._highs[-self.config.period :])
        ll = min(self._lows[-self.config.period :])

        if self._hilo_line is None:
            self._hilo_line = hh
        elif self._hilo_line == hh:
            if close > self._hilo_line:
                self._hilo_line = ll
        elif self._hilo_line == ll:
            if close < self._hilo_line:
                self._hilo_line = hh

        if self._hilo_line is None:
            return

        if not self.in_position:
            if close > self._hilo_line:
                self.open_position(OrderSide.BUY, close)
            elif close < self._hilo_line:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and close < self._hilo_line:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and close > self._hilo_line:
                self.exit_market()
