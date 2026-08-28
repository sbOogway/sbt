"""Cointegrated-crypto statistical arbitrage, Leung & Nguyen (2019, SEF).

Build a mean-reverting linear combination (spread) of the basket's log-prices
via Johansen or Engle-Granger, normalized so the primary leg has weight 1.
Trade the deviation of the spread from its rolling equilibrium with an
entry/exit z-score discipline and an optional stop-loss, holding the long/short
leg basket (long positive-weight legs, short negative-weight legs under a LONG
spread; reversed under SHORT) sized via equal-notional ``leg_quantity``.

State machine: FLAT -> (enter on |z| > entry_z) -> LONG/SHORT -> (exit on
|z| < exit_z, or stop-loss) -> FLAT. The spread vector is re-estimated on a
``reestimate_every`` cadence over a rolling ``estimation_window``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from ....plugins import SBTPortfolioStrategyConfig
from ...base import SBTPortfolioStrategy
from .cointegration import fit_weights

_DAY_NS = 86_400_000_000_000

# position state
_FLAT, _LONG, _SHORT = 0, 1, -1


class CointegratedArbConfig(SBTPortfolioStrategyConfig, kw_only=True, frozen=True):
    method: str = "engle_granger"   # "johansen" | "engle_granger"
    estimation_window: int = 60     # rolling days used to fit the spread vector
    reestimate_every: int = 20      # days between refits of the cointegration vector
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_loss_pct: float | None = None


class CointegratedArb(SBTPortfolioStrategy):
    _METHODS = ("johansen", "engle_granger")

    def __init__(self, config: CointegratedArbConfig) -> None:
        super().__init__(config)
        if config.method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}, got {config.method!r}")
        if config.estimation_window <= 0 or config.reestimate_every <= 0:
            raise ValueError("estimation_window and reestimate_every must be positive")
        if not (config.entry_z > config.exit_z >= -config.entry_z):
            raise ValueError("need entry_z > exit_z with entry_z > 0")
        if config.stop_loss_pct is not None and config.stop_loss_pct <= 0:
            raise ValueError(f"stop_loss_pct must be > 0 when set, got {config.stop_loss_pct!r}")
        # Stable iteration order matching config.symbols (or just the primary
        # leg when the basket is empty); mirrors self._legs in the base class.
        self._order: list[InstrumentId] = list(self._legs)
        # Per-leg (ts_ns, close)
        self._series: dict[InstrumentId, list[tuple[int, float]]] = {
            iid: [] for iid in self._order
        }
        self._state: int = _FLAT
        self._weights: np.ndarray | None = None
        self._last_fit_ns: int | None = None
        self._entry_price: dict[InstrumentId, float] = {}

    def on_instrument_bar(self, instrument_id: InstrumentId, bar: Bar) -> None:
        self._series[instrument_id].append(
            (bar.ts_event, float(bar.close.as_double()))
        )
        if instrument_id != self._primary_iid:
            return
        day_ns = int(pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").normalize().value)
        if not self.trading_active:
            return
        self._step(day_ns)

    # ------------------------------------------------------------------ #

    def _spread_frame(self, day_ns: int) -> pd.DataFrame | None:
        lo = day_ns - self.config.estimation_window * _DAY_NS
        rows: list[dict] = []
        for iid, pairs in self._series.items():
            for t, c in pairs:
                if lo < t <= day_ns and c > 0:
                    rows.append({"t": t, "iid": iid, "lp": float(np.log(c))})
        if not rows:
            return None
        df = pd.DataFrame(rows).pivot(index="t", columns="iid", values="lp")
        df = df.reindex(columns=[iid for iid in self._order if iid in df.columns])
        if len(df) < max(2, self.config.estimation_window // 4):
            return None
        df = df.dropna()
        if len(df) < 2 or self._primary_iid not in df.columns:
            return None
        return df

    def _step(self, day_ns: int) -> None:
        frame = self._spread_frame(day_ns)
        if frame is None:
            return

        # Refit the cointegrating vector on cadence.
        if (
            self._weights is None
            or self._last_fit_ns is None
            or (day_ns - self._last_fit_ns) >= self.config.reestimate_every * _DAY_NS
        ):
            try:
                self._weights = fit_weights(frame, self.config.method)
            except Exception:
                return
            self._last_fit_ns = day_ns

        w = self._weights
        cols = [iid for iid in self._order if iid in frame.columns]
        if len(cols) != len(w):
            return
        spread = (frame[cols].values @ w)
        mean = float(spread.mean())
        std = float(spread.std())
        if std <= 0:
            return
        z = float((spread[-1] - mean) / std)

        if self._state == _FLAT:
            if z <= -self.config.entry_z:
                self._enter(_LONG, day_ns)
            elif z >= self.config.entry_z:
                self._enter(_SHORT, day_ns)
        else:
            if self._stop_hit():
                self._flatten()
                return
            # Exit when the spread reverts back across the exit threshold.
            if self._state == _LONG and z > -self.config.exit_z:
                self._flatten()
            elif self._state == _SHORT and z < self.config.exit_z:
                self._flatten()

    # ------------------------------------------------------------------ #

    def _enter(self, state: int, _day_ns: int) -> None:
        self._state = state
        self._entry_price.clear()
        for iid in self._order:
            leg = self._leg(iid)
            if leg.price is None:
                continue
            w = self._weights[self._order.index(iid)]
            # LONG spread buys positive-weight legs, sells negative; SHORT flips.
            if (state == _LONG and w > 0) or (state == _SHORT and w < 0):
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL
            self._entry_price[iid] = leg.price
            self.open_position(side, leg.price, iid)

    def _stop_hit(self) -> bool:
        if self.config.stop_loss_pct is None:
            return False
        for iid in self._order:
            leg = self._leg(iid)
            if leg.side is None or iid not in self._entry_price:
                continue
            ref = self._entry_price[iid]
            if ref <= 0:
                continue
            move = (leg.price - ref) / ref
            if leg.side == OrderSide.BUY and move <= -self.config.stop_loss_pct:
                return True
            if leg.side == OrderSide.SELL and move >= self.config.stop_loss_pct:
                return True
        return False

    def _flatten(self) -> None:
        for iid in self._order:
            if self._leg(iid).side is not None:
                self.exit_market(iid)
        self._state = _FLAT
        self._entry_price.clear()
