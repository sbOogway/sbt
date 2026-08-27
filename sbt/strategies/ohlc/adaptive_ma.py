from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class AdaptiveMaConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    fast_period: int = 2
    slow_period: int = 30
    er_period: int = 10
    sc_fast: float = 2.0
    sc_slow: float = 0.02

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class AdaptiveMa(SBTStrategy):
    def __init__(self, config: AdaptiveMaConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._kama: float | None = None
        self._prev_kama: float | None = None

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        needed = self.config.er_period + 1
        if len(self._closes) < needed:
            return

        direction = close - self._closes[-needed]
        volatility = sum(
            abs(self._closes[-i] - self._closes[-i - 1])
            for i in range(1, self.config.er_period + 1)
        )

        if volatility == 0:
            er = 0.0
        else:
            er = abs(direction) / volatility

        sc = (er * (self.config.sc_fast - self.config.sc_slow) + self.config.sc_slow) ** 2

        if self._kama is None:
            self._kama = close
        else:
            self._kama = self._kama + sc * (close - self._kama)

        prev = self._prev_kama
        self._prev_kama = self._kama

        if prev is None:
            return

        if not self.in_position:
            if close > self._kama and prev <= self._kama:
                self.open_position(OrderSide.BUY, close)
            elif close < self._kama and prev >= self._kama:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and close < self._kama:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and close > self._kama:
                self.exit_market()
