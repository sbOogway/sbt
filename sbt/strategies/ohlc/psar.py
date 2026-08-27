from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class PSARConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    af_start: float = 0.02
    af_step: float = 0.02
    af_max: float = 0.2

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class PSAR(SBTStrategy):
    def __init__(self, config: PSARConfig) -> None:
        super().__init__(config)
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._psar: float | None = None
        self._af: float = config.af_start
        self._ep: float = 0.0
        self._bull: bool = True
        self._prev_psar: float | None = None

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()

        self._highs.append(high)
        self._lows.append(low)

        if len(self._highs) < 2:
            return

        if self._psar is None:
            self._bull = high > low
            self._psar = low if self._bull else high
            self._ep = high if self._bull else low
            self._prev_psar = self._psar
            return

        prev_psar = self._psar
        if self._bull:
            new_psar = prev_psar + self._af * (self._ep - prev_psar)
            new_psar = min(new_psar, self._lows[-2], self._lows[-1])
            if low < new_psar:
                self._bull = False
                self._psar = self._ep
                self._ep = low
                self._af = self.config.af_start
            else:
                self._psar = new_psar
                if high > self._ep:
                    self._ep = high
                    self._af = min(self._af + self.config.af_step, self.config.af_max)
        else:
            new_psar = prev_psar + self._af * (self._ep - prev_psar)
            new_psar = max(new_psar, self._highs[-2], self._highs[-1])
            if high > new_psar:
                self._bull = True
                self._psar = self._ep
                self._ep = high
                self._af = self.config.af_start
            else:
                self._psar = new_psar
                if low < self._ep:
                    self._ep = low
                    self._af = min(self._af + self.config.af_step, self.config.af_max)

        if self._prev_psar is not None:
            if not self.in_position:
                if self._bull and prev_psar <= close and self._psar > close:
                    self.open_position(OrderSide.BUY, close)
                elif not self._bull and prev_psar >= close and self._psar < close:
                    self.open_position(OrderSide.SELL, close)
            else:
                if self.position_side == OrderSide.BUY and not self._bull:
                    self.exit_market()
                elif self.position_side == OrderSide.SELL and self._bull:
                    self.exit_market()

        self._prev_psar = self._psar
