from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class StochasticRsiConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    rsi_period: int = 14
    stoch_period: int = 14
    k_smooth: int = 3
    oversold: float = 20.0
    overbought: float = 80.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class StochasticRsi(SBTStrategy):
    def __init__(self, config: StochasticRsiConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._rsi_values: list[float] = []
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._initialized: bool = False

    def _compute_rsi(self) -> float | None:
        period = self.config.rsi_period
        if len(self._closes) < period + 1:
            return None
        if not self._initialized:
            gains = []
            losses = []
            for i in range(1, period + 1):
                delta = self._closes[i] - self._closes[i - 1]
                gains.append(max(delta, 0.0))
                losses.append(max(-delta, 0.0))
            self._avg_gain = sum(gains) / period
            self._avg_loss = sum(losses) / period
            self._initialized = True
        else:
            delta = self._closes[-1] - self._closes[-2]
            self._avg_gain = (self._avg_gain * (period - 1) + max(delta, 0)) / period
            self._avg_loss = (self._avg_loss * (period - 1) + max(-delta, 0)) / period
        if self._avg_loss == 0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        rsi = self._compute_rsi()
        if rsi is None:
            return
        self._rsi_values.append(rsi)

        needed = self.config.stoch_period + self.config.k_smooth
        if len(self._rsi_values) < needed:
            return

        stoch_rsi_vals = self._rsi_values[-self.config.stoch_period :]
        rsi_min = min(stoch_rsi_vals)
        rsi_max = max(stoch_rsi_vals)
        if rsi_max == rsi_min:
            k = 50.0
        else:
            k = (self._rsi_values[-1] - rsi_min) / (rsi_max - rsi_min) * 100

        if not self.in_position:
            if k < self.config.oversold:
                self.open_position(OrderSide.BUY, close)
            elif k > self.config.overbought:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and k > 50:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and k < 50:
                self.exit_market()
