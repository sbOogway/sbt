from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class MomentumReversalConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    fast_period: int = 5
    slow_period: int = 20
    rsi_period: int = 14

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class MomentumReversal(SBTStrategy):
    def __init__(self, config: MomentumReversalConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._rsi_init: bool = False

    def _ema(self, period: int) -> float | None:
        if len(self._closes) < period:
            return None
        k = 2.0 / (period + 1)
        ema = sum(self._closes[:period]) / period
        for p in self._closes[period:]:
            ema = p * k + ema * (1 - k)
        return ema

    def _rsi(self) -> float | None:
        period = self.config.rsi_period
        if len(self._closes) < period + 1:
            return None
        if not self._rsi_init:
            gains = []
            losses = []
            for i in range(1, period + 1):
                d = self._closes[i] - self._closes[i - 1]
                gains.append(max(d, 0))
                losses.append(max(-d, 0))
            self._avg_gain = sum(gains) / period
            self._avg_loss = sum(losses) / period
            self._rsi_init = True
        else:
            d = self._closes[-1] - self._closes[-2]
            self._avg_gain = (self._avg_gain * (period - 1) + max(d, 0)) / period
            self._avg_loss = (self._avg_loss * (period - 1) + max(-d, 0)) / period
        if self._avg_loss == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + self._avg_gain / self._avg_loss)

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        fast = self._ema(self.config.fast_period)
        slow = self._ema(self.config.slow_period)
        rsi = self._rsi()

        if fast is None or slow is None or rsi is None:
            return

        if not self.in_position:
            if fast > slow and rsi < 30:
                self.open_position(OrderSide.BUY, close)
            elif fast < slow and rsi > 70:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and (fast < slow or rsi > 60):
                self.exit_market()
            elif self.position_side == OrderSide.SELL and (fast > slow or rsi < 40):
                self.exit_market()
