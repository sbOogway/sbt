import pandas as pd
from decimal import Decimal

from nautilus_trader.model.data import BarType, FundingRateUpdate
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ...plugins import SBTStrategyConfig
from ..base import SBTStrategy


class WeekdaySeasonalityConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    # Retained for run provenance; sizing compounds off live account equity.
    capital: Decimal
    leverage: float
    risk_percent: float = 1.0

    backtest_start_date: str = "2020-01-01"

    # Day-of-week effect: long over this UTC weekday (Monday premium:
    # Aharon & Qadan 2019; Caporale & Plastun 2019b; Long et al. 2020).
    # Daily bars are open-stamped, so the entry fills at the PRIOR day's
    # close and the exit at the target day's close — capturing exactly the
    # target weekday's daily return.
    entry_weekday: int = 0
    # Month-of-year avoidance (July/August weakness: Plastun et al. 2019);
    # empty = trade all months.
    skip_months: tuple[int, ...] = ()

    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 30
    vol_max_scale: float = 3.0


class WeekdaySeasonality(SBTStrategy):
    def __init__(self, config: WeekdaySeasonalityConfig) -> None:
        super().__init__(config)
        self._latest_price: float | None = None

    def on_start(self) -> None:
        self.subscribe_funding_rates(self.instrument_id)
        super().on_start()

    def on_funding_rate(self, funding_rate: FundingRateUpdate) -> None:
        self.funding.accrue(
            self.position_side,
            self._open_qty,
            self._latest_price,
            float(funding_rate.rate),
        )

    def on_trading_bar(self, bar) -> None:
        dt_utc = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        close_price = bar.close.as_double()
        self._latest_price = close_price

        if self.position_side is not None:
            self.exit_market()
            return

        target_date = dt_utc.normalize() + pd.Timedelta(days=1)
        if (
            target_date.weekday() == self.config.entry_weekday
            and target_date.month not in self.config.skip_months
        ):
            notional = (
                self.equity()
                * self.config.risk_percent
                * self.config.leverage
                * self.plugins.size_multiplier()
            )
            self.enter_market(
                OrderSide.BUY,
                self.sized_quantity(notional / close_price),
            )
