from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class ForceIndexConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 13

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class ForceIndex(SBTStrategy):
    def __init__(self, config: ForceIndexConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._prev_close: float | None = None
        self._ema: float | None = None

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        volume = bar.volume.as_double() if hasattr(bar, "volume") else 1.0

        self._closes.append(close)

        if self._prev_close is None:
            self._prev_close = close
            return

        force = (close - self._prev_close) * volume
        self._prev_close = close

        k = 2.0 / (self.config.period + 1)
        if self._ema is None:
            self._ema = force
        else:
            self._ema = force * k + self._ema * (1 - k)

        if len(self._closes) < self.config.period + 2:
            return

        if not self.in_position:
            if self._ema > 0 and force > 0:
                self.open_position(OrderSide.BUY, close)
            elif self._ema < 0 and force < 0:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and self._ema < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and self._ema > 0:
                self.exit_market()
