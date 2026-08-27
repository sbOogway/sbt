from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class ChaikinOscillatorConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    fast_period: int = 3
    slow_period: int = 10

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class ChaikinOscillator(SBTStrategy):
    def __init__(self, config: ChaikinOscillatorConfig) -> None:
        super().__init__(config)
        self._adl: list[float] = []
        self._prev_adl_fast: float | None = None
        self._prev_adl_slow: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        volume = bar.volume.as_double() if hasattr(bar, "volume") else 1.0

        hl = high - low
        mfm = ((close - low) - (high - close)) / hl if hl > 0 else 0
        prev_adl = self._adl[-1] if self._adl else 0.0
        self._adl.append(prev_adl + mfm * volume)

        needed = max(self.config.slow_period, self.config.fast_period) + 1
        if len(self._adl) < needed:
            return

        fast_ema = sum(self._adl[-self.config.fast_period :]) / self.config.fast_period
        slow_ema = sum(self._adl[-self.config.slow_period :]) / self.config.slow_period
        cho = fast_ema - slow_ema

        prev_cho = None
        if self._prev_adl_fast is not None and self._prev_adl_slow is not None:
            prev_cho = self._prev_adl_fast - self._prev_adl_slow

        k_f = 2.0 / (self.config.fast_period + 1)
        k_s = 2.0 / (self.config.slow_period + 1)
        new_fast = self._adl[-1] * k_f + (self._prev_adl_fast or self._adl[-1]) * (1 - k_f)
        new_slow = self._adl[-1] * k_s + (self._prev_adl_slow or self._adl[-1]) * (1 - k_s)
        self._prev_adl_fast = new_fast
        self._prev_adl_slow = new_slow

        if not self.in_position:
            if prev_cho is not None and prev_cho < 0 and cho > 0:
                self.open_position(OrderSide.BUY, close)
            elif prev_cho is not None and prev_cho > 0 and cho < 0:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and cho < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and cho > 0:
                self.exit_market()
