"""Rolling volatility-scaling plugin (Moreira & Muir).

Scales position size by the realized-volatility regime: each observation day
computes realized variance (RV) over the trailing ``rv_lookback`` days,
compares it against the average of all trailing RVs seen so far, and sets
``weight = min(vol_max_scale, C / RV)``.

Feeding modes:

- **automatic** (``vol_track_daily=True``): tracks daily close-to-close
  returns across date rollovers from the forwarded bar stream.
- **manual** (``vol_track_daily=False``): the host strategy calls
  :meth:`add_return` itself whenever it has a completed-period return — for
  sampling schemes tied to a specific time of day or timezone.

Weight refresh frequency:

- ``daily``: weight updates after every fed return.
- ``monthly``: weight refresh is deferred to the first day completion in a
  new calendar month and held constant within the month (requires automatic
  tracking so month boundaries are known).
"""

import pandas as pd

from .base import SBTStrategyConfig, SizingPlugin


class VolScalingPlugin(SizingPlugin):
    name = "vol_scaling"

    def __init__(self, config: SBTStrategyConfig) -> None:
        super().__init__(config)
        self.rv_lookback = int(getattr(config, "rv_lookback", 22))
        self.vol_max_scale = float(getattr(config, "vol_max_scale", 2.0))
        self.vol_rebalance_freq = str(
            getattr(config, "vol_rebalance_freq", "daily")
        ).lower()
        self.vol_track_daily = bool(getattr(config, "vol_track_daily", True))

        if self.vol_rebalance_freq not in ("daily", "monthly"):
            raise ValueError(
                f"Invalid vol_rebalance_freq: {self.vol_rebalance_freq!r}. "
                "Expected 'daily' or 'monthly'."
            )

        self.daily_returns: list[float] = []
        self._rv_history: list[float] = []
        self._weight: float = 1.0

        # Automatic daily-tracking state
        self._last_date = None
        self._last_close: float | None = None
        self._prev_day_close: float | None = None
        self._rebalance_month: int | None = None

    # ------------------------------------------------------------------
    # Feeding
    # ------------------------------------------------------------------

    def on_bar(self, strategy, bar) -> None:
        if not self.vol_track_daily:
            return
        date = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").date()
        close = bar.close.as_double()

        if self._last_date is not None and date != self._last_date:
            self._on_day_complete(self._last_date, self._last_close, new_date=date)

        self._last_date = date
        self._last_close = close

    def add_return(self, daily_return: float) -> None:
        """Feed one completed-period return manually."""
        self.daily_returns.append(daily_return)
        if len(self.daily_returns) < self.rv_lookback:
            return

        rv = sum(r * r for r in self.daily_returns[-self.rv_lookback :])
        self._rv_history.append(rv)
        c = sum(self._rv_history) / len(self._rv_history)
        new_weight = (
            min(self.vol_max_scale, c / rv) if rv > 0 else self.vol_max_scale
        )
        if self.vol_rebalance_freq == "monthly":
            self._pending_weight = new_weight
        else:
            self._weight = new_weight

    def _on_day_complete(self, day_date, day_close: float | None, new_date=None) -> None:
        """A trading day ended: feed its close-to-close return, maybe rebalance.

        *new_date* is the incoming day; monthly weight refresh takes effect on
        the first day of a new calendar month (weight held constant within it).
        """
        if day_close is not None and self._prev_day_close not in (None, 0):
            self.add_return(day_close / self._prev_day_close - 1.0)

        if (
            self.vol_rebalance_freq == "monthly"
            and getattr(self, "_pending_weight", None) is not None
        ):
            gate_date = new_date if new_date is not None else day_date
            if self._rebalance_month is None or gate_date.month != self._rebalance_month:
                self._weight = self._pending_weight
                self._pending_weight = None
                self._rebalance_month = gate_date.month

        self._prev_day_close = day_close

    # ------------------------------------------------------------------
    # SizingPlugin interface
    # ------------------------------------------------------------------

    def size_multiplier(self) -> float:
        return self._weight

    @property
    def weight(self) -> float:
        return self._weight
