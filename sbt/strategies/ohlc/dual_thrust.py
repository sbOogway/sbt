from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class DualThrustConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    lookback: int = 4
    k1: float = 0.5
    k2: float = 0.5
    holding_bars: int = 8

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class DualThrust(SBTStrategy):
    def __init__(self, config: DualThrustConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []
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

        n = self.config.lookback
        if len(self._highs) < n + 1:
            return

        hh = max(self._highs[-n - 1 : -1])
        hc = max(self._closes[-n - 1 : -1])
        lc = min(self._closes[-n - 1 : -1])
        ll = min(self._lows[-n - 1 : -1])

        range_val = max(hh - lc, hc - ll)
        prev_close = self._closes[-2]

        upper = prev_close + self.config.k1 * range_val
        lower = prev_close - self.config.k2 * range_val

        if close > upper:
            self.open_position(OrderSide.BUY, close)
            self._bars_held = 0
        elif close < lower:
            self.open_position(OrderSide.SELL, close)
            self._bars_held = 0
