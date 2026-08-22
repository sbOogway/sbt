"""Shared base class and funding tracking for SBT strategies."""

from decimal import Decimal

import pandas as pd
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from ..plugins import PluginHost, SBTStrategyConfig


class FundingTracker:
    """Accrues signed funding payments per open position.

    Payments are signed from the holder's perspective: a long paying a
    positive rate accrues a positive cost, a short receives (negative
    cost). :attr:`total_paid` therefore follows the convention
    *positive = strategy paid*, matching ``BacktestResult.funding_pnl``.
    """

    def __init__(self) -> None:
        self._open_accrual: Decimal = Decimal(0)
        self._settled: list[Decimal] = []

    def accrue(
        self,
        side: OrderSide | None,
        qty: Quantity | None,
        price: float | None,
        rate: float,
    ) -> None:
        """Accrue one funding payment for the currently open position."""
        if side is None or qty is None or price is None:
            return
        sign = 1.0 if side == OrderSide.BUY else -1.0
        self._open_accrual += Decimal(str(sign * float(qty) * float(price) * rate))

    def settle_position(self) -> None:
        """Freeze the accrual of the position being flattened."""
        if self._open_accrual:
            self._settled.append(self._open_accrual)
        self._open_accrual = Decimal(0)

    @property
    def total_paid(self) -> float:
        """Net funding paid across settled + open positions (>0 = paid)."""
        return float(sum(self._settled, Decimal(0)) + self._open_accrual)


class SBTStrategy(Strategy):
    """Common plumbing for SBT bar strategies.

    Provides compounding live-equity sizing helpers, market-order entry/
    exit state tracking, a typed funding side-channel, and optional
    trading-window gating via ``config.active_from``.

    Subclasses implement :meth:`on_trading_bar`; it receives **every**
    bar so indicators and plugins warm up continuously, while order
    submission is suppressed until the first bar at/after ``active_from``.
    Runner-level windows (train/val holdout) exploit this to preload
    lookback bars without trading on them.
    """

    def __init__(self, config: SBTStrategyConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        bar_type = getattr(config, "bar_type", None)
        if bar_type is not None:
            self.bar_type = bar_type
        self.plugins = PluginHost.from_config(config)
        self.funding = FundingTracker()
        self.position_side: OrderSide | None = None
        self._open_qty: Quantity | None = None
        self._active_from_ns: int | None = _parse_active_from_ns(
            getattr(config, "active_from", None)
        )
        self._current_ts_ns: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.plugins.on_start(self)
        bar_type = getattr(self.config, "bar_type", None)
        if bar_type is not None:
            self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        self._current_ts_ns = bar.ts_event
        self.plugins.on_bar(self, bar)
        self.on_trading_bar(bar)

    def on_trading_bar(self, bar: Bar) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Trading-window gating
    # ------------------------------------------------------------------

    @property
    def trading_active(self) -> bool:
        """False while processing warm-up bars (before ``active_from``)."""
        return (
            self._active_from_ns is None
            or self._current_ts_ns >= self._active_from_ns
        )

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def equity(self) -> float:
        """Live total account balance in settlement currency (compounds)."""
        account = self.cache.account_for_venue(self.instrument_id.venue)
        if account is None:
            return 0.0
        bal = account.balance_total()
        return float(bal.as_double()) if bal is not None else 0.0

    def vol_multiplier(self) -> float:
        """Product of all sizing-plugin multipliers (1.0 when none)."""
        return self.plugins.size_multiplier()

    def sized_quantity(self, size: float) -> Quantity | None:
        qty = round(size, 3)
        if qty <= 0:
            return None
        return Quantity(qty, precision=3)

    # ------------------------------------------------------------------
    # Order plumbing
    # ------------------------------------------------------------------

    def submit_market(self, side: OrderSide, qty: Quantity | None) -> bool:
        """Submit a market order; suppressed during warm-up. Returns filled intent."""
        if qty is None or not self.trading_active:
            return False
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
        )
        self.submit_order(order)
        return True

    def enter_market(self, side: OrderSide, qty: Quantity | None) -> bool:
        """Open the tracked position with a market order."""
        if not self.submit_market(side, qty):
            return False
        self.position_side = side
        self._open_qty = qty
        return True

    def exit_market(self) -> bool:
        """Flatten the tracked position and settle its funding accrual."""
        if self.position_side is None or self._open_qty is None:
            self.position_side = None
            self._open_qty = None
            return False
        close_side = (
            OrderSide.SELL
            if self.position_side == OrderSide.BUY
            else OrderSide.BUY
        )
        submitted = self.submit_market(close_side, self._open_qty)
        self.funding.settle_position()
        self.position_side = None
        self._open_qty = None
        return submitted


def _parse_active_from_ns(raw: str | None) -> int | None:
    if not raw:
        return None
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.value)
