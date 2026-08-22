from decimal import Decimal

import pandas as pd
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ..plugins import SBTStrategyConfig
from .base import SBTStrategy


class BitcoinIntradayMomentumConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    # Retained for run provenance; sizing compounds off live account equity.
    capital: Decimal
    leverage: float
    risk_percent: float = 1.0

    backtest_start_date: str = "2020-01-01"

    onfh_close_time: str = "08:00"
    slh_open_time: str = "16:00"
    slh_close_time: str = "16:30"

    plugins: tuple[str, ...] = ("vol_scaling",)
    # Returns are fed manually at the 17:00 US/Eastern close so sampling is
    # tied to that timezone (DST-aware), not fixed UTC boundaries.
    vol_track_daily: bool = False
    rv_lookback: int = 30
    vol_max_scale: float = 2.0


class BitcoinIntradayMomentum(SBTStrategy):
    def __init__(self, config: BitcoinIntradayMomentumConfig) -> None:
        super().__init__(config)
        self.prev_close: float | None = None
        self.onfh_close: float | None = None
        self.slh_open: float | None = None

        self.r_onfh: float | None = None
        self.r_slh: float | None = None

    def on_trading_bar(self, bar) -> None:
        dt_utc = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        dt_est = dt_utc.tz_convert("US/Eastern")
        time_str = dt_est.strftime("%H:%M")

        if time_str == self.config.onfh_close_time:
            self.onfh_close = bar.close.as_double()
            if self.prev_close:
                self.r_onfh = self.onfh_close / self.prev_close - 1.0

        if time_str == self.config.slh_open_time:
            self.slh_open = bar.close.as_double()

        if time_str == self.config.slh_close_time:
            if self.slh_open:
                slh_close = bar.close.as_double()
                self.r_slh = slh_close / self.slh_open - 1.0
            self.evaluate_signal_and_trade(bar.close.as_double())

        if time_str == "17:00":
            self.close_positions()
            close_val = bar.close.as_double()
            scaler = self.plugins.get("vol_scaling")
            if scaler is not None and self.prev_close is not None:
                daily_ret = close_val / self.prev_close - 1.0
                scaler.add_return(daily_ret)
            self.prev_close = close_val

    def evaluate_signal_and_trade(self, price: float) -> None:
        if self.r_onfh is None or self.r_slh is None:
            return

        if self.r_onfh <= 0 and self.r_slh >= 0:
            self._open_trade(OrderSide.SELL, price)
        elif self.r_onfh > 0 and self.r_slh < 0:
            self._open_trade(OrderSide.BUY, price)

    def close_positions(self) -> None:
        self.exit_market()

    def _open_trade(self, order_side: OrderSide, price: float) -> None:
        notional = (
            self.equity()
            * self.config.risk_percent
            * self.config.leverage
            * self.vol_multiplier()
        )
        self.enter_market(order_side, self.sized_quantity(notional / price))
