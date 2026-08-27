from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class MomentumCrashConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    mom_period: int = 12
    vol_lookback: int = 21
    vol_scale_threshold: float = 1.5

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class MomentumCrash(SBTStrategy):
    def __init__(self, config: MomentumCrashConfig) -> None:
        super().__init__(config)
        self._returns: list[float] = []
        self._prev_close: float | None = None

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()

        if self._prev_close is not None:
            ret = close / self._prev_close - 1.0
            self._returns.append(ret)
        self._prev_close = close

        needed = max(self.config.mom_period + 1, self.config.vol_lookback)
        if len(self._returns) < needed:
            return

        mom = sum(self._returns[-self.config.mom_period - 1 : -1])
        current_vol = (sum(r**2 for r in self._returns[-self.config.vol_lookback:]) / self.config.vol_lookback) ** 0.5

        if len(self._returns) < 2 * self.config.vol_lookback:
            hist_vol = current_vol
        else:
            hist_vol = (sum(r**2 for r in self._returns[-2 * self.config.vol_lookback : -self.config.vol_lookback]) / self.config.vol_lookback) ** 0.5

        vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1.0

        if not self.in_position:
            if vol_ratio < self.config.vol_scale_threshold:
                if mom > 0:
                    self.open_position(OrderSide.BUY, close)
                else:
                    self.open_position(OrderSide.SELL, close)
            elif vol_ratio > self.config.vol_scale_threshold:
                if mom > 0:
                    self.open_position(OrderSide.SELL, close)
                else:
                    self.open_position(OrderSide.BUY, close)
        else:
            if self.position_side == OrderSide.BUY and mom < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and mom > 0:
                self.exit_market()
