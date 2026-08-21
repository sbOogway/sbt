from decimal import Decimal

import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from ..plugins import PluginHost, SBTStrategyConfig


class BitcoinIntradayMomentumConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    capital: Decimal
    leverage: float

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


class BitcoinIntradayMomentum(Strategy):
    def __init__(self, config: BitcoinIntradayMomentumConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.prev_close: Decimal | None = None
        self.onfh_close: Decimal | None = None
        self.slh_open: Decimal | None = None

        self.r_onfh: float | None = None
        self.r_slh: float | None = None

        self.current_position_side: OrderSide | None = None
        self._open_qty: Quantity | None = None

        self.plugins = PluginHost.from_config(config)

    def on_start(self) -> None:
        self.plugins.on_start(self)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        dt_utc = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        dt_est = dt_utc.tz_convert("US/Eastern")
        time_str = dt_est.strftime("%H:%M")

        if time_str == self.config.onfh_close_time:
            self.onfh_close = Decimal(bar.close.as_double())
            if self.prev_close:
                self.r_onfh = float(self.onfh_close / self.prev_close) - 1.0

        if time_str == self.config.slh_open_time:
            self.slh_open = Decimal(bar.close.as_double())

        if time_str == self.config.slh_close_time:
            if self.slh_open:
                slh_close = Decimal(bar.close.as_double())
                self.r_slh = float(slh_close / self.slh_open) - 1.0
            self.evaluate_signal_and_trade(Decimal(bar.close.as_double()))

        if time_str == "17:00":
            self.close_positions()
            close_val = Decimal(bar.close.as_double())
            scaler = self.plugins.get("vol_scaling")
            if scaler is not None and self.prev_close is not None:
                daily_ret = float(close_val / self.prev_close) - 1.0
                scaler.add_return(daily_ret)
            self.prev_close = close_val

    def evaluate_signal_and_trade(self, price: Decimal) -> None:
        if self.r_onfh is None or self.r_slh is None:
            return

        if self.r_onfh <= 0 and self.r_slh >= 0:
            self._open_trade(OrderSide.SELL, price)
        elif self.r_onfh > 0 and self.r_slh < 0:
            self._open_trade(OrderSide.BUY, price)

    def close_positions(self) -> None:
        # if self.current_position_side == OrderSide.BUY:
        # self._close_trade(OrderSide.BUY)
        # elif self.current_position_side == OrderSide.SELL:
        self._close_trade(self.current_position_side)

        self.current_position_side = None
        self._open_qty = None

    def _open_trade(self, order_side: OrderSide, price: Decimal) -> None:
        weight = Decimal(str(self.plugins.size_multiplier()))
        notional = self.config.capital * Decimal(self.config.leverage) * weight
        raw_size = notional / price
        self._open_qty = Quantity(round(float(raw_size), 3), precision=3)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=self._open_qty,
        )
        self.submit_order(order)
        self.current_position_side = order_side

    def _close_trade(self, order_side: OrderSide) -> None:
        if self._open_qty is None:
            return
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=self._open_qty,
        )
        self.submit_order(order)
