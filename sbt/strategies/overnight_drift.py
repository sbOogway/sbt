from decimal import Decimal

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
    risk_percent: float = 1.0

    backtest_start_date: str = "2020-01-01"

    entry_time: str = "20:00"
    exit_time: str = "14:00"

    vol_scaling: bool = True
    rv_lookback: int = 5
    vol_max_scale: float = 2.0
    weekdays_only: bool = True
    funding_enabled: bool = False


class OvernightDrift(Strategy):
    def __init__(self, config: OvernightDriftConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.prev_close: Decimal | None = None
        self.current_position_side: OrderSide | None = None
        self._open_qty: Quantity | None = None
        self._latest_price: Decimal | None = None
        self._open_funding_cost: Decimal = Decimal(0)
        self._trade_funding_costs: list[Decimal] = []

        self._vol_scaler = (
            VolatilityScaler(
                rv_lookback=config.rv_lookback,
                vol_max_scale=config.vol_max_scale,
            )
            if config.vol_scaling
            else None
        )

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)
        if self.config.funding_enabled:
            self.subscribe_funding_rates(self.instrument_id)

    def on_bar(self, bar: Bar) -> None:
        dt_utc = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        time_str = dt_utc.strftime("%H:%M")

        close_price = Decimal(bar.close.as_double())
        self._latest_price = close_price

        if time_str == self.config.entry_time:
            is_friday = dt_utc.weekday() == 4
            should_trade = not (self.config.weekdays_only and is_friday)
            if (
                self.prev_close is not None
                and close_price < self.prev_close
                and should_trade
            ):
                self._open_trade(OrderSide.BUY, close_price)

            if (
                self.config.vol_scaling
                and self.prev_close is not None
                and self._vol_scaler is not None
            ):
                daily_ret = float(close_price / self.prev_close) - 1.0
                self._vol_scaler.add_return(daily_ret)

            self.prev_close = close_price

        if time_str == self.config.exit_time:
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
        self._open_funding_cost = Decimal(0)

    def _open_trade(self, order_side: OrderSide, price: Decimal) -> None:
        weight = (
            Decimal(self._vol_scaler.weight)
            if self._vol_scaler is not None
            else Decimal(1.0)
        )
        account = self.cache.account_for_venue(self.instrument_id.venue)
        bal = account.balance_total()
        if bal is None:
            return
        current_equity = Decimal(str(bal.as_double()))
        if current_equity <= 0:
            return
        notional = (
            current_equity
            * Decimal(self.config.risk_percent)
            * Decimal(self.config.leverage)
            * weight
        )
        raw_size = notional / price
        self._open_qty = Quantity(round(float(raw_size), 3), precision=3)

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=self._open_qty,
        )
        self.submit_order(order)
        self.current_position_side = order_side

    def _close_trade(self, order_side: OrderSide | None) -> None:
        if self._open_qty is None or order_side is None:
            return
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=self._open_qty,
        )
        self.submit_order(order)
