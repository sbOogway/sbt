from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class KeltnerChannelConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    ema_period: int = 20
    atr_period: int = 10
    atr_mult: float = 2.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class KeltnerChannel(SBTStrategy):
    def __init__(self, config: KeltnerChannelConfig) -> None:
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

        needed = max(self.config.ema_period, self.config.atr_period)
        if len(self._closes) < needed:
            return

        ema = sum(self._closes[-self.config.ema_period :]) / self.config.ema_period
        atr = sum(self._true_ranges[-self.config.atr_period :]) / self.config.atr_period

        upper = ema + self.config.atr_mult * atr
        lower = ema - self.config.atr_mult * atr

        if not self.in_position:
            if close > upper:
                self.open_position(OrderSide.BUY, close)
            elif close < lower:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and close < ema:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and close > ema:
                self.exit_market()
