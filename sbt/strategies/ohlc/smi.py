from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class SmiConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 14
    signal_period: int = 3
    threshold: float = 20.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class SMI(SBTStrategy):
    def __init__(self, config: SmiConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._diff_ema: float | None = None
        self._sum_ema: float | None = None
        self._signal_ema: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        needed = self.config.period + 1
        if len(self._closes) < needed:
            return

        hl2 = (high + low) / 2
        diff = close - hl2
        summ = high - low

        k = 2.0 / (self.config.period + 1)
        if self._diff_ema is None:
            self._diff_ema = diff
            self._sum_ema = summ
        else:
            self._diff_ema = diff * k + self._diff_ema * (1 - k)
            self._sum_ema = summ * k + self._sum_ema * (1 - k)

        if self._sum_ema == 0:
            return

        stoch_k = self._diff_ema / (self._sum_ema / 2) * 100

        ks = 2.0 / (self.config.signal_period + 1)
        if self._signal_ema is None:
            self._signal_ema = stoch_k
        else:
            self._signal_ema = stoch_k * ks + self._signal_ema * (1 - ks)

        if not self.in_position:
            if stoch_k > self.config.threshold and self._signal_ema < self.config.threshold:
                self.open_position(OrderSide.BUY, close)
            elif stoch_k < -self.config.threshold and self._signal_ema > -self.config.threshold:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and stoch_k < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and stoch_k > 0:
                self.exit_market()
