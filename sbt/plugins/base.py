"""Plugin infrastructure for SBT strategies and runners.

Strategy-level plugins receive forwarded lifecycle events from their host
strategy and may influence sizing. Runner-level plugins expand one job into
multiple execution windows (e.g. in-sample / out-of-sample holdout).

Plugins are enabled per strategy through the ``plugins`` config tuple::

    class MyConfig(SBTStrategyConfig, frozen=True):
        plugins: tuple[str, ...] = ("vol_scaling",)

Plugin parameters stay as flat fields on the strategy config so optimizer
specs (``rv_lookback=int(3,30)``) keep working unchanged.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar

from sbt.core.job import BacktestResult


class SBTStrategyConfig(StrategyConfig, kw_only=True, frozen=True):
    """Base config for all SBT strategies.

    Adds the ``plugins`` tuple used to opt in to pluggable behaviour.
    ``kw_only=True`` matches nautilus ``StrategyConfig`` so subclasses may
    freely mix required and defaulted fields.
    """

    plugins: tuple[str, ...] = ()


class StrategyPlugin(ABC):
    """Base class for strategy-level plugins.

    Plugins are constructed with the full strategy config and read their own
    parameters from it via :func:`getattr` defaults.
    """

    name: ClassVar[str]

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


class RunnerPlugin(ABC):
    """Base class for runner-level plugins that split a job into windows."""

    name: ClassVar[str]

    @abstractmethod
    def combine(self, job_id: str, results: dict) -> "BacktestResult":
        """Merge per-window BacktestResults into one combined result."""
        raise NotImplementedError


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
        return cls([get_plugin_class(name)(config) for name in names])

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
