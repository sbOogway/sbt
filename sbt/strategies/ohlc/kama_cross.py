from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class KamaCrossConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    fast_period: int = 10
    slow_period: int = 30
    er_period: int = 10
    sc_fast: float = 2.0
    sc_slow: float = 0.02

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class KamaCross(SBTStrategy):
    def __init__(self, config: KamaCrossConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._fast_kama: float | None = None
        self._slow_kama: float | None = None

    def _update_kama(self, prev_kama: float | None, closes: list[float]) -> float:
        needed = self.config.er_period + 1
        if len(closes) < needed:
            return closes[-1]

        direction = closes[-1] - closes[-needed]
        volatility = sum(abs(closes[-i] - closes[-i - 1]) for i in range(1, needed))
        er = abs(direction) / volatility if volatility > 0 else 0
        sc = (er * (self.config.sc_fast - self.config.sc_slow) + self.config.sc_slow) ** 2

        if prev_kama is None:
            return closes[-1]
        return prev_kama + sc * (closes[-1] - prev_kama)

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        self._fast_kama = self._update_kama(self._fast_kama, self._closes)
        self._slow_kama = self._update_kama(self._slow_kama, self._closes)

        if len(self._closes) < max(self.config.slow_period, self.config.er_period) + 2:
            return

        prev_fast = self._fast_kama
        prev_slow = self._slow_kama

        if not self.in_position:
            if prev_fast is not None and prev_slow is not None:
                if prev_fast <= prev_slow and self._fast_kama > self._slow_kama:
                    self.open_position(OrderSide.BUY, close)
                elif prev_fast >= prev_slow and self._fast_kama < self._slow_kama:
                    self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and self._fast_kama < self._slow_kama:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and self._fast_kama > self._slow_kama:
                self.exit_market()
