from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class UltimateOscillatorConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period1: int = 7
    period2: int = 14
    period3: int = 28
    overbought: float = 70.0
    oversold: float = 30.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class UltimateOscillator(SBTStrategy):
    def __init__(self, config: UltimateOscillatorConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        needed = self.config.period3 + 1
        if len(self._closes) < needed:
            return

        def avg_pressure(p: int) -> float:
            bp = sum(
                self._closes[-i] - min(self._lows[-i - 1], self._closes[-i - 1])
                for i in range(1, p + 1)
            )
            tr_sum = sum(
                max(
                    self._highs[-i] - self._lows[-i],
                    abs(self._highs[-i] - self._closes[-i - 1]),
                    abs(self._lows[-i] - self._closes[-i - 1]),
                )
                for i in range(1, p + 1)
            )
            return bp / tr_sum if tr_sum > 0 else 0

        bp1 = avg_pressure(self.config.period1)
        bp2 = avg_pressure(self.config.period2)
        bp3 = avg_pressure(self.config.period3)

        uo = (4 * bp1 + 2 * bp2 + bp3) / 7.0 * 100

        if not self.in_position:
            if uo < self.config.oversold:
                self.open_position(OrderSide.BUY, close)
            elif uo > self.config.overbought:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and uo > 50:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and uo < 50:
                self.exit_market()
