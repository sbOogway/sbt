from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class ChandeMomentumConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 14
    threshold: float = 50.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class ChandeMomentum(SBTStrategy):
    def __init__(self, config: ChandeMomentumConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        if len(self._closes) < self.config.period + 1:
            return

        deltas = [
            self._closes[-i] - self._closes[-i - 1]
            for i in range(1, self.config.period + 1)
        ]
        gains = sum(d for d in deltas if d > 0)
        losses = sum(-d for d in deltas if d < 0)
        total = gains + losses
        if total == 0:
            return

        cmo = (gains - losses) / total * 100

        if not self.in_position:
            if cmo < -self.config.threshold:
                self.open_position(OrderSide.BUY, close)
            elif cmo > self.config.threshold:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and cmo > 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and cmo < 0:
                self.exit_market()
