from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class TRIXConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 15
    signal_period: int = 9

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class TRIX(SBTStrategy):
    def __init__(self, config: TRIXConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._trix_values: list[float] = []
        self._ema1: float | None = None
        self._ema2: float | None = None
        self._ema3: float | None = None

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        needed = self.config.period * 3 + 1
        if len(self._closes) < needed:
            return

        k = 2.0 / (self.config.period + 1)
        if self._ema1 is None:
            self._ema1 = sum(self._closes[: self.config.period]) / self.config.period
            self._ema2 = self._ema1
            self._ema3 = self._ema1
            for p in self._closes[self.config.period : needed - 1]:
                self._ema1 = p * k + self._ema1 * (1 - k)
                self._ema2 = self._ema1 * k + self._ema2 * (1 - k)
                self._ema3 = self._ema2 * k + self._ema3 * (1 - k)

        self._ema1 = close * k + self._ema1 * (1 - k)
        self._ema2 = self._ema1 * k + self._ema2 * (1 - k)
        new_ema3 = self._ema2 * k + self._ema3 * (1 - k)

        if self._ema3 > 0:
            trix = (new_ema3 - self._ema3) / self._ema3 * 10000
        else:
            trix = 0.0
        self._ema3 = new_ema3
        self._trix_values.append(trix)

        if len(self._trix_values) < self.config.signal_period + 1:
            return

        signal = sum(self._trix_values[-self.config.signal_period :]) / self.config.signal_period

        if not self.in_position:
            if trix > signal and self._trix_values[-2] <= sum(
                self._trix_values[-self.config.signal_period - 1 : -1]
            ) / self.config.signal_period:
                self.open_position(OrderSide.BUY, close)
            elif trix < signal and self._trix_values[-2] >= sum(
                self._trix_values[-self.config.signal_period - 1 : -1]
            ) / self.config.signal_period:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and trix < signal:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and trix > signal:
                self.exit_market()
