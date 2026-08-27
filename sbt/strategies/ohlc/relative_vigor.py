from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class RelativeVigorConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 10
    signal_period: int = 3

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class RelativeVigor(SBTStrategy):
    def __init__(self, config: RelativeVigorConfig) -> None:
        super().__init__(config)
        self._num: list[float] = []
        self._den: list[float] = []

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        open_ = bar.open.as_double()

        self._num.append(close - open_)
        self._den.append(high - low)

        if len(self._num) < self.config.period:
            return

        window_n = self._num[-self.config.period :]
        window_d = self._den[-self.config.period :]

        numerator = sum(window_n)
        denominator = sum(window_d)

        rvg = numerator / denominator if denominator > 0 else 0
        rvg_signal = rvg

        if not self.in_position:
            if rvg > 0 and rvg > rvg_signal:
                self.open_position(OrderSide.BUY, close)
            elif rvg < 0 and rvg < rvg_signal:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and rvg < 0:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and rvg > 0:
                self.exit_market()
