import pandas as pd

from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class OvernightDriftConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    entry_time: str = "20:00"
    exit_time: str = "14:00"

    plugins: tuple[str, ...] = ("vol_scaling",)
    # Returns are fed manually at entry_time so sampling follows the
    # (optimizable) entry hour rather than fixed UTC midnight boundaries.
    vol_track_daily: bool = False
    rv_lookback: int = 5
    vol_max_scale: float = 2.0
    weekdays_only: bool = True


class OvernightDrift(SBTStrategy):
    def __init__(self, config: OvernightDriftConfig) -> None:
        super().__init__(config)
        self.prev_close: float | None = None

    def on_trading_bar(self, bar) -> None:
        dt_utc = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        time_str = dt_utc.strftime("%H:%M")

        close_price = bar.close.as_double()

        if time_str == self.config.entry_time:
            is_friday = dt_utc.weekday() == 4
            should_trade = not (self.config.weekdays_only and is_friday)
            if (
                self.prev_close is not None
                and close_price < self.prev_close
                and should_trade
            ):
                self.open_position(OrderSide.BUY, close_price)

            scaler = self.plugins.get("vol_scaling")
            if scaler is not None and self.prev_close is not None:
                daily_ret = close_price / self.prev_close - 1.0
                scaler.add_return(daily_ret)

            self.prev_close = close_price

        if time_str == self.config.exit_time:
            self.exit_market()
