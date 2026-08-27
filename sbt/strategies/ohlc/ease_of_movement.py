from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class EaseOfMovementConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 14
    signal_period: int = 9

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class EaseOfMovement(SBTStrategy):
    def __init__(self, config: EaseOfMovementConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[float] = []
        self._prev_ema: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        volume = bar.volume.as_double() if hasattr(bar, "volume") else 1.0

        self._highs.append(high)
        self._lows.append(low)
        self._volumes.append(volume)

        needed = max(self.config.period, self.config.signal_period) + 1
        if len(self._highs) < needed:
            return

        dm = ((high + low) / 2) - ((self._highs[-2] + self._lows[-2]) / 2)
        br = volume / (high - low) if high != low else 0
        eom = dm / br if br != 0 else 0

        k = 2.0 / (self.config.period + 1)
        if self._prev_ema is None:
            self._prev_ema = eom
        else:
            self._prev_ema = eom * k + self._prev_ema * (1 - k)

        if not self.in_position:
            if eom > 0 and eom > self._prev_ema:
                self.open_position(OrderSide.BUY, close)
            elif eom < 0 and eom < self._prev_ema:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and eom < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and eom > 0:
                self.exit_market()
