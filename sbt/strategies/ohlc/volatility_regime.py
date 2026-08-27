from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class VolatilityRegimeConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    vol_lookback: int = 20
    vol_expansion_mult: float = 1.2
    ema_period: int = 10

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class VolatilityRegime(SBTStrategy):
    def __init__(self, config: VolatilityRegimeConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._returns: list[float] = []
        self._prev_close: float | None = None

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()

        if self._prev_close is not None:
            ret = close / self._prev_close - 1.0
            self._returns.append(ret)
        self._prev_close = close
        self._closes.append(close)

        lb = self.config.vol_lookback
        if len(self._returns) < lb + 1:
            return

        current_vol = (sum(r**2 for r in self._returns[-lb:]) / lb) ** 0.5
        hist_vol = (sum(r**2 for r in self._returns[-2 * lb : -lb]) / lb) ** 0.5

        if hist_vol <= 0:
            return

        vol_ratio = current_vol / hist_vol

        k = 2.0 / (self.config.ema_period + 1)
        if len(self._closes) >= self.config.ema_period:
            ema = sum(self._closes[-self.config.ema_period :]) / self.config.ema_period
            for p in self._closes[-self.config.ema_period + 1 :]:
                ema = p * k + ema * (1 - k)
        else:
            return

        if not self.in_position:
            if vol_ratio < 1.0 / self.config.vol_expansion_mult:
                if close > ema:
                    self.open_position(OrderSide.BUY, close)
                else:
                    self.open_position(OrderSide.SELL, close)
            elif vol_ratio > self.config.vol_expansion_mult:
                if close > ema:
                    self.open_position(OrderSide.BUY, close)
                else:
                    self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and close < ema:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and close > ema:
                self.exit_market()
