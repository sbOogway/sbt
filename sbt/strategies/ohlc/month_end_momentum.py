import pandas as pd

from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class MonthEndMomentumConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    lookback_months: int = 1
    holding_days: int = 5

    subscribe_funding: bool = True
    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 30
    vol_max_scale: float = 3.0


class MonthEndMomentum(SBTStrategy):
    def __init__(self, config: MonthEndMomentumConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []
        self._dates: list[pd.Timestamp] = []
        self._bars_held: int = 0
        self._entry_date: pd.Timestamp | None = None

    def on_trading_bar(self, bar) -> None:
        dt = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        close = bar.close.as_double()

        self._closes.append(close)
        self._dates.append(dt)

        if self.in_position:
            self._bars_held += 1
            if self._bars_held >= self.config.holding_days:
                self.exit_market()
                self._bars_held = 0
            return

        if len(self._closes) < 30:
            return

        if dt.day <= 3 and self._entry_date != dt.normalize():
            monthly_ret = close / self._closes[-30] - 1.0
            if monthly_ret > 0:
                self.open_position(OrderSide.BUY, close)
            elif monthly_ret < 0:
                self.open_position(OrderSide.SELL, close)
            self._entry_date = dt.normalize()
            self._bars_held = 0
