from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class NegativeVolumeIndexConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    ema_period: int = 255

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class NegativeVolumeIndex(SBTStrategy):
    def __init__(self, config: NegativeVolumeIndexConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._volumes: list[float] = []
        self._nvi: float = 1000.0
        self._prev_vol: float | None = None
        self._nvi_ema: float | None = None

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        volume = bar.volume.as_double() if hasattr(bar, "volume") else 1.0

        self._closes.append(close)
        self._volumes.append(volume)

        if self._prev_vol is not None and volume < self._prev_vol:
            if self._prev_vol > 0:
                self._nvi *= close / self._closes[-2] if self._closes[-2] > 0 else 1.0

        self._prev_vol = volume

        k = 2.0 / (self.config.ema_period + 1)
        if self._nvi_ema is None:
            self._nvi_ema = self._nvi
        else:
            self._nvi_ema = self._nvi * k + self._nvi_ema * (1 - k)

        if len(self._closes) < self.config.ema_period:
            return

        if not self.in_position:
            if self._nvi > self._nvi_ema:
                self.open_position(OrderSide.BUY, close)
            elif self._nvi < self._nvi_ema:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and self._nvi < self._nvi_ema:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and self._nvi > self._nvi_ema:
                self.exit_market()
