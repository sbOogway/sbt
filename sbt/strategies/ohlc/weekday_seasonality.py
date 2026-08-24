import pandas as pd

from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class WeekdaySeasonalityConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    # Day-of-week effect: long over this UTC weekday (Monday premium:
    # Aharon & Qadan 2019; Caporale & Plastun 2019b; Long et al. 2020).
    # Daily bars are open-stamped, so the entry fills at the PRIOR day's
    # close and the exit at the target day's close — capturing exactly the
    # target weekday's daily return.
    entry_weekday: int = 0
    # Month-of-year avoidance (July/August weakness: Plastun et al. 2019);
    # empty = trade all months.
    skip_months: tuple[int, ...] = ()

    subscribe_funding: bool = True
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 30
    vol_max_scale: float = 3.0


class WeekdaySeasonality(SBTStrategy):
    def on_trading_bar(self, bar) -> None:
        dt_utc = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        close_price = bar.close.as_double()

        if self.position_side is not None:
            self.exit_market()
            return

        target_date = dt_utc.normalize() + pd.Timedelta(days=1)
        if (
            target_date.weekday() == self.config.entry_weekday
            and target_date.month not in self.config.skip_months
        ):
            self.open_position(OrderSide.BUY, close_price)
