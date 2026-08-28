"""Shared base class and funding tracking for sbt strategies."""

from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd
from nautilus_trader.model.data import Bar, BarType, FundingRateUpdate
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from ..plugins import (
    PluginHost,
    SBTStrategyConfig,
    SBTPortfolioStrategyConfig,
)
from ..utils import make_instrument_id, parse_interval


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
    """Common plumbing for sbt bar strategies.

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
        self._latest_price: float | None = None
        self._active_from_ns: int | None = _parse_active_from_ns(
            getattr(config, "active_from", None)
        )
        self._current_ts_ns: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.plugins.on_start(self)
        if self.config.subscribe_funding:
            self.subscribe_funding_rates(self.instrument_id)
        bar_type = getattr(self.config, "bar_type", None)
        if bar_type is not None:
            self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        self._current_ts_ns = bar.ts_event
        self._latest_price = bar.close.as_double()
        self.plugins.on_bar(self, bar)
        self.on_trading_bar(bar)

    def on_trading_bar(self, bar: Bar) -> None:
        raise NotImplementedError

    def on_funding_rate(self, funding_rate: FundingRateUpdate) -> None:
        """Accrue one funding payment against the open position.

        Strategies opt in via ``subscribe_funding=True`` on their config
        (the base subscribes in ``on_start``); accrual prices at the
        latest bar close.
        """
        self.funding.accrue(
            self.position_side,
            self._open_qty,
            self._latest_price,
            float(funding_rate.rate),
        )

    # ------------------------------------------------------------------
    # Trading-window gating
    # ------------------------------------------------------------------

    @property
    def trading_active(self) -> bool:
        """False while processing warm-up bars (before ``active_from``)."""
        return (
            self._active_from_ns is None or self._current_ts_ns >= self._active_from_ns
        )

    # ------------------------------------------------------------------
    # Position / sizing
    # ------------------------------------------------------------------

    @property
    def in_position(self) -> bool:
        """True while the tracked position is open."""
        return self.position_side is not None

    def equity(self) -> float:
        """Live total account balance in settlement currency (compounds)."""
        account = self.cache.account_for_venue(self.instrument_id.venue)
        if account is None:
            return 0.0
        bal = account.balance_total()
        return float(bal.as_double()) if bal is not None else 0.0

    def open_position(self, side: OrderSide, price: float) -> bool:
        """Enter at market, sized off live equity (the canonical formula).

        ``notional = equity * risk_percent * leverage
        * plugins.size_multiplier()``; quantity = notional / *price*.
        Stop-distance strategies size via :meth:`risk_quantity` instead.
        """
        if price <= 0:
            return False
        notional = (
            self.equity()
            * self.config.risk_percent
            * self.config.leverage
            * self.plugins.size_multiplier()
        )
        return self.enter_market(side, self.sized_quantity(notional / price))

    def sized_quantity(self, size: float) -> Quantity | None:
        qty = round(size, 3)
        if qty <= 0:
            return None
        return Quantity(qty, precision=3)

    def risk_quantity(
        self, stop_distance: float, risk_fraction: float
    ) -> Quantity | None:
        """Quantity risking ``risk_fraction`` of live equity over the stop."""
        if stop_distance <= 0:
            return None
        risk_amount = (
            self.equity()
            * self.config.leverage
            * risk_fraction
            * self.plugins.size_multiplier()
        )
        return self.sized_quantity(risk_amount / stop_distance)

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
            OrderSide.SELL if self.position_side == OrderSide.BUY else OrderSide.BUY
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


@dataclass
class LegState:
    """State of one portfolio leg (one instrument's tracked position)."""

    side: OrderSide | None = None
    qty: Quantity | None = None
    price: float | None = None
    funding: FundingTracker = field(default_factory=FundingTracker)


class _LegFundingAggregate:
    """Exposes ``total_paid`` = sum of all leg funding trackers.

    Lets ``BacktestResult._collect_result`` read ``strategy.funding.total_paid``
    unchanged, matching the single-instrument ``SBTStrategy.funding`` contract.
    """

    def __init__(self, legs: dict[InstrumentId, LegState]) -> None:
        self._legs = legs

    @property
    def total_paid(self) -> float:
        return sum(leg.funding.total_paid for leg in self._legs.values())


class SBTPortfolioStrategy(Strategy):
    """Multi-instrument (portfolio) base: per-leg positions on a shared account.

    The single-position ``SBTStrategy`` stays untouched (backward compat for
    the ~70 single-instrument strategies). This subclass instead manages a
    dict of per-instrument :class:`LegState` — side, quantity, latest price and
    a personal ``FundingTracker`` per leg.

    The runner injects ``instrument_id`` = the primary (first) symbol's
    instrument and builds one perpetual + bar stream per ``config.symbols``
    entry on a shared margin account. Subclasses implement
    :meth:`on_instrument_bar` (the per-leg analog of ``on_trading_bar``) and
    size via :meth:`open_position` (equal-weight ``leg_quantity``) or an
    explicit per-leg quantity.

    ``bar`` is routed to the owning leg by ``bar.bar_type.instrument_id``;
    funding updates are routed per leg by ``funding_rate.instrument_id``.
    The synthetic ``funding`` property sums every leg's accrual, so
    ``_collect_result`` needs no change.
    """

    def __init__(self, config: SBTPortfolioStrategyConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.plugins = PluginHost.from_config(config)
        self._legs: dict[InstrumentId, LegState] = {}
        self._init_legs()
        self._active_from_ns: int | None = _parse_active_from_ns(config.active_from)
        self._current_ts_ns: int = 0
        # Backward-compatible attributes: position_side = primary leg's side,
        # in_position = any leg open; funding = aggregate.
        self._primary_iid = config.instrument_id

    # ------------------------------------------------------------------
    # Legs
    # ------------------------------------------------------------------

    def _init_legs(self) -> None:
        venue = self.instrument_id.venue
        for sym in getattr(self.config, "symbols", ()) or ():
            self._legs[make_instrument_id(venue.value, sym)] = LegState()
        if not self._legs:
            self._legs[self.instrument_id] = LegState()

    @property
    def legs(self) -> dict[InstrumentId, LegState]:
        """Map of instrument id -> current leg state (read-only view)."""
        return self._legs

    @property
    def position_map(self) -> dict[InstrumentId, OrderSide | None]:
        """Map of instrument id -> current tracked side (None = flat)."""
        return {iid: leg.side for iid, leg in self._legs.items()}

    def _leg(self, instrument_id: InstrumentId | None = None) -> LegState:
        iid = instrument_id or self._primary_iid
        leg = self._legs.setdefault(iid, LegState())
        return leg

    def leg_side(self, instrument_id: InstrumentId | None = None) -> OrderSide | None:
        return self._leg(instrument_id).side

    def leg_in_position(self, instrument_id: InstrumentId | None = None) -> bool:
        leg = self._leg(instrument_id)
        return leg.side is not None and leg.qty is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.plugins.on_start(self)
        interval_nt = parse_interval(getattr(self.config, "interval", "1d"))
        for iid in self._legs:
            bar_type = BarType.from_str(f"{iid.value}-{interval_nt}-LAST-EXTERNAL")
            self.subscribe_bars(bar_type)
            if getattr(self.config, "subscribe_funding", False):
                self.subscribe_funding_rates(iid)

    def on_bar(self, bar: Bar) -> None:
        self._current_ts_ns = bar.ts_event
        iid = bar.bar_type.instrument_id
        leg = self._leg(iid)
        leg.price = bar.close.as_double()
        self.plugins.on_bar(self, bar)
        self.on_instrument_bar(iid, bar)

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        """Per-leg trading hook, the portfolio analog of ``on_trading_bar``."""
        raise NotImplementedError

    def on_funding_rate(self, funding_rate: FundingRateUpdate) -> None:
        leg = self._leg(funding_rate.instrument_id)
        leg.funding.accrue(
            leg.side,
            leg.qty,
            leg.price,
            float(funding_rate.rate),
        )

    # ------------------------------------------------------------------
    # Trading-window gating
    # ------------------------------------------------------------------

    @property
    def trading_active(self) -> bool:
        return (
            self._active_from_ns is None or self._current_ts_ns >= self._active_from_ns
        )

    # ------------------------------------------------------------------
    # Position / sizing
    # ------------------------------------------------------------------

    @property
    def in_position(self) -> bool:
        """True while any leg is open."""
        return any(leg.side is not None for leg in self._legs.values())

    @property
    def position_side(self) -> OrderSide | None:
        """Primary leg's side (backward-compatible accessor)."""
        return self._leg(self._primary_iid).side

    @property
    def funding(self) -> _LegFundingAggregate:
        """Synthetic funding: net of all leg trackers (>0 = strategy paid)."""
        return _LegFundingAggregate(self._legs)

    def equity(self) -> float:
        """Live whole-portfolio balance in settlement currency (compounds)."""
        account = self.cache.account_for_venue(self.instrument_id.venue)
        if account is None:
            return 0.0
        bal = account.balance_total()
        return float(bal.as_double()) if bal is not None else 0.0

    def leg_quantity(self, price: float, n_legs: int) -> Quantity | None:
        """Equal-weight quantity: whole-portfolio notional split across n legs."""
        if price <= 0 or n_legs <= 0:
            return None
        notional = (
            self.equity()
            * getattr(self.config, "risk_percent", 1.0)
            * getattr(self.config, "leverage", 1.0)
            * self.plugins.size_multiplier()
        )
        return self.sized_quantity(notional / n_legs / price)

    def sized_quantity(self, size: float) -> Quantity | None:
        qty = round(size, 3)
        if qty <= 0:
            return None
        return Quantity(qty, precision=3)

    # ------------------------------------------------------------------
    # Order plumbing
    # ------------------------------------------------------------------

    def submit_market(
        self,
        side: OrderSide,
        qty: Quantity | None,
        instrument_id: InstrumentId | None = None,
    ) -> bool:
        """Submit a market order on the given leg; suppressed during warm-up."""
        iid = instrument_id or self._primary_iid
        if qty is None or not self.trading_active:
            return False
        order = self.order_factory.market(
            instrument_id=iid,
            order_side=side,
            quantity=qty,
        )
        self.submit_order(order)
        return True

    def open_position(
        self,
        side: OrderSide,
        price: float,
        instrument_id: InstrumentId | None = None,
    ) -> bool:
        """Enter one leg at market, equal-weighted across all legs."""
        if price <= 0:
            return False
        qty = self.leg_quantity(price, len(self._legs))
        return self._open_leg(side, qty, instrument_id)

    def _open_leg(
        self,
        side: OrderSide,
        qty: Quantity | None,
        instrument_id: InstrumentId | None = None,
    ) -> bool:
        iid = instrument_id or self._primary_iid
        if not self.submit_market(side, qty, iid):
            return False
        leg = self._leg(iid)
        leg.side = side
        leg.qty = qty
        return True

    def exit_market(self, instrument_id: InstrumentId | None = None) -> bool:
        """Flatten one leg and settle its funding accrual."""
        iid = instrument_id or self._primary_iid
        leg = self._leg(iid)
        if leg.side is None or leg.qty is None:
            leg.side = None
            leg.qty = None
            return False
        close_side = (
            OrderSide.SELL if leg.side == OrderSide.BUY else OrderSide.BUY
        )
        submitted = self.submit_market(close_side, leg.qty, iid)
        leg.funding.settle_position()
        leg.side = None
        leg.qty = None
        return submitted

    def exit_all_legs(self) -> int:
        """Flatten every open leg; returns the number closed."""
        closed = 0
        for iid in list(self._legs):
            leg = self._leg(iid)
            if leg.side is not None:
                self.exit_market(iid)
                closed += 1
        return closed

    def apply_targets(
        self, targets: dict[InstrumentId, OrderSide | None]
    ) -> None:
        """Reconcile each leg's position to its target side.

        For every (iid, target) in *targets*:

        - flat leg + non-None target + price => open at target
        - open leg + None target => flatten
        - open leg + different target => flatten and reopen at target
          (reopen guarded by *leg.price*; exit is unconditional)

        Legs not in *targets* are left untouched.
        """
        for iid, target in targets.items():
            leg = self._leg(iid)
            if leg.side is None:
                if target is not None and leg.price:
                    self.open_position(target, leg.price, iid)
            elif target is None:
                self.exit_market(iid)
            elif leg.side != target:
                self.exit_market(iid)
                if leg.price:
                    self.open_position(target, leg.price, iid)
