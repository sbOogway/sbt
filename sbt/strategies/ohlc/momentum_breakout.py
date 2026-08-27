from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class MomentumBreakoutConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    roc_period: int = 12
    atr_period: int = 14
    atr_mult: float = 1.5

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class MomentumBreakout(SBTStrategy):
    def __init__(self, config: MomentumBreakoutConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._true_ranges: list[float] = []
        self._prev_close: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
            self._true_ranges.append(tr)
        self._prev_close = close
        self._closes.append(close)

        needed = max(self.config.roc_period + 1, self.config.atr_period)
        if len(self._closes) < needed:
            return

        roc = close / self._closes[-(self.config.roc_period + 1)] - 1.0
        atr = sum(self._true_ranges[-self.config.atr_period :]) / self.config.atr_period

        if not self.in_position:
            threshold = self.config.atr_mult * atr / close
            if roc > threshold:
                self.open_position(OrderSide.BUY, close)
            elif roc < -threshold:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and roc < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and roc > 0:
                self.exit_market()
