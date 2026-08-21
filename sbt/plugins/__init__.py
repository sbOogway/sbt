"""SBT plugin registry."""

from .base import (
    PluginHost,
    RunnerPlugin,
    SBTStrategyConfig,
    SizingPlugin,
    StrategyPlugin,
)
from .train_val_split import IN_SAMPLE, OUT_OF_SAMPLE, TrainValSplit
from .vol_scaling import VolScalingPlugin

_PLUGIN_REGISTRY = {
    VolScalingPlugin.name: VolScalingPlugin,
}


def get_plugin_class(name: str) -> type:
    cls = _PLUGIN_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown plugin: {name!r}. Available: {sorted(_PLUGIN_REGISTRY)}"
        )
    return cls


def plugin_names() -> list[str]:
    return list(_PLUGIN_REGISTRY)


__all__ = [
    "IN_SAMPLE",
    "OUT_OF_SAMPLE",
    "PluginHost",
    "RunnerPlugin",
    "SBTStrategyConfig",
    "SizingPlugin",
    "StrategyPlugin",
    "TrainValSplit",
    "VolScalingPlugin",
    "get_plugin_class",
    "plugin_names",
]
