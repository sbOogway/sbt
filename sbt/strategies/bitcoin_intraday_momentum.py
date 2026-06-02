from decimal import Decimal
from typing import Optional

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from ..volatility import VolatilityScaler


class BitcoinIntradayMomentumConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    capital: Decimal
    leverage: float

    backtest_start_date: str = "2020-01-01"

    onfh_close_time: str = "08:00"
    slh_open_time: str = "16:00"
    slh_close_time: str = "16:30"

    vol_scaling: bool = True
    rv_lookback: int = 30
    max_leverage: float = 0.0


class BitcoinIntradayMomentum(Strategy):
    def __init__(self, config: BitcoinIntradayMomentumConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.prev_close: Optional[Decimal] = None
        self.onfh_close: Optional[Decimal] = None
        self.slh_open: Optional[Decimal] = None

        self.r_onfh: Optional[float] = None
        self.r_slh: Optional[float] = None

        self.current_position_side: Optional[OrderSide] = None
        self._open_qty: Optional[Quantity] = None

        self._vol_scaler = VolatilityScaler(
            rv_lookback=config.rv_lookback,
            max_leverage=config.max_leverage,
        ) if config.vol_scaling else None

    def on_start(self) -> None:
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
            if self.config.vol_scaling and self.prev_close is not None and self._vol_scaler is not None:
                daily_ret = float(close_val / self.prev_close) - 1.0
                self._vol_scaler.add_return(daily_ret)
                dt_today = dt_est.date()
                dt_tomorrow = dt_today + pd.Timedelta(days=1)
                if dt_tomorrow.month != dt_today.month:
                    self._vol_scaler.rebalance(dt_tomorrow.month)
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
        weight = Decimal(self._vol_scaler.weight) if self._vol_scaler is not None else Decimal(1.0)
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


