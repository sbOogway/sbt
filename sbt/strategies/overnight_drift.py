from decimal import Decimal
from typing import Optional

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, FundingRateUpdate
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from ..volatility import VolatilityScaler


class OvernightDriftConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    capital: Decimal
    leverage: float

    backtest_start_date: str = "2020-01-01"

    nyse_open_time: str = "09:00"
    nyse_close_time: str = "16:00"
    europe_open_time: str = "02:00"

    vol_scaling: bool = True
    rv_lookback: int = 22
    max_leverage: float = 0.0


class OvernightDrift(Strategy):
    def __init__(self, config: OvernightDriftConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.prev_close: Optional[Decimal] = None
        self.current_position_side: Optional[OrderSide] = None
        self._open_qty: Optional[Quantity] = None
        self._latest_price: Optional[Decimal] = None
        self._open_funding_cost: Decimal = Decimal("0")
        self._trade_funding_costs: list[Decimal] = []

        self._vol_scaler = (
            VolatilityScaler(
                rv_lookback=config.rv_lookback,
                max_leverage=config.max_leverage,
            )
            if config.vol_scaling
            else None
        )

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)
        self.subscribe_funding_rates(self.instrument_id)

    def on_bar(self, bar: Bar) -> None:
        dt_utc = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        dt_et = dt_utc.tz_convert("US/Eastern")
        time_str = dt_et.strftime("%H:%M")

        close_price = Decimal(bar.close.as_double())
        self._latest_price = close_price

        if time_str == self.config.nyse_close_time:
            if self.prev_close is not None and close_price < self.prev_close:
                self._open_trade(OrderSide.BUY, close_price)

            if (
                self.config.vol_scaling
                and self.prev_close is not None
                and self._vol_scaler is not None
            ):
                daily_ret = float(close_price / self.prev_close) - 1.0
                self._vol_scaler.add_return(daily_ret)
                dt_today = dt_et.date()
                dt_tomorrow = dt_today + pd.Timedelta(days=1)
                if dt_tomorrow.month != dt_today.month:
                    self._vol_scaler.rebalance(dt_tomorrow.month)

            self.prev_close = close_price

        if time_str == self.config.europe_open_time:
            self.close_positions()

    def on_funding_rate(self, funding_rate: FundingRateUpdate) -> None:
        if self._open_qty is not None and self._latest_price is not None:
            qty = Decimal(str(self._open_qty))
            rate = Decimal(str(funding_rate.rate))
            payment = qty * self._latest_price * rate
            self._open_funding_cost += payment

    def close_positions(self) -> None:
        if self.current_position_side is not None:
            close_side = (
                OrderSide.SELL
                if self.current_position_side == OrderSide.BUY
                else OrderSide.BUY
            )
            self._close_trade(close_side)
            self._trade_funding_costs.append(self._open_funding_cost)
        self.current_position_side = None
        self._open_qty = None
        self._open_funding_cost = Decimal("0")

    def _open_trade(self, order_side: OrderSide, price: Decimal) -> None:
        weight = (
            Decimal(self._vol_scaler.weight)
            if self._vol_scaler is not None
            else Decimal(1.0)
        )
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

    def _close_trade(self, order_side: Optional[OrderSide]) -> None:
        if self._open_qty is None or order_side is None:
            return
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=self._open_qty,
        )
        self.submit_order(order)
