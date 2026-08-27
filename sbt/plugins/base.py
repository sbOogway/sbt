"""Plugin infrastructure for sbt strategies and runners.

Strategy-level plugins receive forwarded lifecycle events from their host
strategy and may influence sizing. Runner-level plugins expand one job into
multiple execution windows (e.g. in-sample / out-of-sample holdout).

Plugins are enabled per strategy through the ``plugins`` config tuple::

    class MyConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
        plugins: tuple[str, ...] = ("vol_scaling",)

Plugin parameters stay as flat fields on the strategy config so optimizer
specs (``rv_lookback=int(3,30)``) keep working unchanged.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId

from sbt.core.job import BacktestResult

if TYPE_CHECKING:
    from sbt.core.config import RunConfig


class SBTStrategyConfig(StrategyConfig, kw_only=True, frozen=True):
    """Base config for all sbt strategies.

    Carries the runner-injected fields — ``instrument_id``, ``capital``,
    ``leverage``, ``backtest_start_date``, ``active_from`` — so concrete
    strategies declare only their signal parameters. The runner always
    supplies these on construction; the defaults exist for direct use in
    tests. Also holds the shared sizing/trading knobs — ``risk_percent``
    (full-notional entries via :meth:`SBTStrategy.open_position`) and
    ``subscribe_funding`` (funding-rate side-channel opt-in) — and the
    ``plugins`` opt-in tuple plus ``active_from`` for trading-window
    gating: bars before this timestamp warm up indicators/plugins but
    cannot produce orders.
    ``kw_only=True`` matches nautilus ``StrategyConfig`` so subclasses may
    freely mix required and defaulted fields.
    """

    instrument_id: InstrumentId
    capital: Decimal = Decimal("1000")
    leverage: float = 1.0
    backtest_start_date: str = "2020-01-01"
    # Full-notional entries: notional = equity * risk_percent * leverage
    # * plugin multiplier (stop-distance sizing ignores it).
    risk_percent: float = 1.0
    subscribe_funding: bool = False
    plugins: tuple[str, ...] = ()
    active_from: str | None = None


class SBTBarStrategyConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Config tier for bar-driven strategies: adds the ``bar_type``.

    Bar-mode strategies subclass this; order-book (L2) strategies subclass
    :class:`SBTStrategyConfig` directly and never receive a ``bar_type``.
    """

    bar_type: BarType


class SBTPortfolioStrategyConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Config tier for multi-instrument (portfolio) bar strategies.

    Adds the ``symbols`` basket and the shared ``interval``. The runner
    injects ``instrument_id`` (the primary / first symbol's instrument) and
    builds one perpetual + bar stream per symbol on a shared margin account;
    the strategy derives each leg's instrument id and bar type from
    ``symbols`` + ``interval``. No single ``bar_type`` — each leg subscribes
    its own. ``symbols`` order is significant: index 0 is the primary leg.
    """

    symbols: tuple[str, ...] = ()
    interval: str = "1d"


class StrategyPlugin(ABC):
    """Base class for strategy-level plugins.

    Plugins are constructed with the full strategy config and read their own
    parameters from it via :func:`getattr` defaults.

    ClassVars:
        required_config_fields: config field names that MUST exist on the
            host strategy config; ``PluginHost.from_config`` refuses to
            construct the plugin otherwise (guards optimizer typos from
            silently falling back to defaults).
        optional_config_fields: names read with getattr defaults; documented
            for discoverability, not enforced.
    """

    name: ClassVar[str]
    required_config_fields: ClassVar[tuple[str, ...]] = ()
    optional_config_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: SBTStrategyConfig) -> None:
        self.config = config

    def on_start(self, strategy) -> None:
        pass

    def on_bar(self, strategy, bar: Bar) -> None:
        pass

    def on_stop(self, strategy) -> None:
        pass


class SizingPlugin(StrategyPlugin):
    """A plugin that contributes a multiplier to position sizing."""

    @abstractmethod
    def size_multiplier(self) -> float:
        """Return the current sizing multiplier (1.0 = no adjustment)."""
        raise NotImplementedError


class Window(NamedTuple):
    """One execution window derived by a runner plugin.

    Attributes:
        label: Human-readable name for logs/reports.
        start: Trading start (orders gated at this instant via strategy
            ``active_from``; earlier rows in *df* are warm-up only).
        end: Trading end (inclusive upper bound).
        df: Pre-sliced OHLCV frame including warm-up bars ahead of
            *start*, or ``None`` to have the runner load by date (L2 mode).
    """

    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    df: pd.DataFrame | None = None


class RunnerPlugin(ABC):
    """Base class for runner-level plugins that expand one job into windows.

    Contract: :meth:`expand` derives named execution windows from the run
    config (+ the loaded bar frame when available); the runner executes each
    window through the normal path and merges with :meth:`combine`.
    """

    name: ClassVar[str]

    @abstractmethod
    def expand(self, cfg: "RunConfig", df: pd.DataFrame | None) -> dict[str, Window]:
        """Return named windows to execute for this run.

        Raises ``ValueError`` on invalid configuration or unusable data —
        the runner converts that into a FAILED result.
        """
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        job_id: str,
        results: dict[str, BacktestResult],
        windows: dict[str, Window],
    ) -> BacktestResult:
        """Merge per-window BacktestResults into one combined result."""
        raise NotImplementedError

    def summarize(self, results: dict[str, BacktestResult]) -> None:
        """Optional per-window console summary after a successful combine."""


def _config_field_names(config) -> set[str]:
    """Field names of a msgspec Struct or dataclass config."""
    struct_fields = getattr(type(config), "__struct_fields__", None)
    if struct_fields is not None:
        return set(struct_fields)
    import dataclasses

    return {f.name for f in dataclasses.fields(config)}


class PluginHost:
    """Owns the plugin instances attached to a strategy.

    Strategies instantiate once in ``__init__`` and forward lifecycle events::

        self.plugins = PluginHost.from_config(config)
        ...
        self.plugins.on_bar(self, bar)
        notional *= self.plugins.size_multiplier()
    """

    def __init__(self, plugins: list[StrategyPlugin]) -> None:
        self._plugins = plugins

    @classmethod
    def from_config(cls, config: SBTStrategyConfig) -> "PluginHost":
        from . import get_plugin_class  # deferred: avoids circular import

        names = getattr(config, "plugins", ()) or ()
        available = _config_field_names(config)
        plugins = []
        for name in names:
            plugin_cls = get_plugin_class(name)
            missing = [
                f for f in plugin_cls.required_config_fields if f not in available
            ]
            if missing:
                raise ValueError(
                    f"Plugin '{name}' requires config fields missing from "
                    f"{type(config).__name__}: {missing}. Silent defaults are "
                    "disabled to keep optimizer trials honest."
                )
            plugins.append(plugin_cls(config))
        return cls(plugins)

    def get(self, name: str) -> StrategyPlugin | None:
        """Return the plugin registered under *name*, or None."""
        for p in self._plugins:
            if p.name == name:
                return p
        return None

    def on_start(self, strategy) -> None:
        for p in self._plugins:
            p.on_start(strategy)

    def on_bar(self, strategy, bar: Bar) -> None:
        for p in self._plugins:
            p.on_bar(strategy, bar)

    def on_stop(self, strategy) -> None:
        for p in self._plugins:
            p.on_stop(strategy)

    def size_multiplier(self) -> float:
        """Product of all sizing-plugin multipliers (1.0 when none)."""
        m = 1.0
        for p in self._plugins:
            if isinstance(p, SizingPlugin):
                m *= p.size_multiplier()
        return m
