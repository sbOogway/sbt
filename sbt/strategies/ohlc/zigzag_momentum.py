from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class ZigzagMomentumConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    swing_pct: float = 3.0
    holding_bars: int = 10

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class ZigzagMomentum(SBTStrategy):
    def __init__(self, config: ZigzagMomentumConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._last_pivot: float | None = None
        self._bars_held: int = 0

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        if self.in_position:
            self._bars_held += 1
            if self._bars_held >= self.config.holding_bars:
                self.exit_market()
                self._bars_held = 0
            return

        if self._last_pivot is None:
            self._last_pivot = close
            return

        pct = (close - self._last_pivot) / self._last_pivot * 100

        if abs(pct) > self.config.swing_pct:
            if pct > 0:
                self.open_position(OrderSide.BUY, close)
            else:
                self.open_position(OrderSide.SELL, close)
            self._last_pivot = close
            self._bars_held = 0
