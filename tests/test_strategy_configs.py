"""Strategy-config construction contract.

The runner injects a fixed set of kwargs into every strategy config; those
fields live on ``SBTStrategyConfig`` (plus ``bar_type`` on the bar tier).
These tests pin that contract for every registered strategy so config drift
fails loudly here instead of at backtest time.
"""

from decimal import Decimal

import pytest
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

from sbt.core.runner import _build_strategy_config
from sbt.plugins import SBTBarStrategyConfig, SBTStrategyConfig
from sbt.utils import get_strategy_class, get_strategy_names
from sbt.utils import make_perpetual


def _injected_kwargs():
    return {
        "instrument_id": InstrumentId.from_str("BTCUSDT-PERP.TESTEX"),
        "capital": Decimal("1000"),
        "leverage": 1.0,
        "backtest_start_date": "2024-01-01",
        "active_from": "2024-01-02T00:00:00+00:00",
    }


def test_every_registered_config_accepts_injected_kwargs():
    instrument = make_perpetual("TESTEX", "BTC/USDT:USDT")
    bar_type = BarType.from_str(f"{instrument.id.value}-1-HOUR-LAST-EXTERNAL")

    assert get_strategy_names(), "strategy registry must not be empty"
    for name in get_strategy_names():
        _, ConfigClass = get_strategy_class(name)
        kwargs = _injected_kwargs()
        if issubclass(ConfigClass, SBTBarStrategyConfig):
            kwargs["bar_type"] = bar_type
        elif issubclass(ConfigClass, SBTStrategyConfig) is False:
            pytest.fail(f"{ConfigClass.__name__} does not subclass SBTStrategyConfig")
        config = _build_strategy_config(ConfigClass, **kwargs)
        assert config.instrument_id == kwargs["instrument_id"]
        assert config.capital == Decimal("1000")


def test_bar_configs_declare_bar_type_l2_configs_do_not():
    bar_names, l2_names = [], []
    for name in get_strategy_names():
        _, ConfigClass = get_strategy_class(name)
        if issubclass(ConfigClass, SBTBarStrategyConfig):
            bar_names.append(name)
        else:
            l2_names.append(name)

    assert bar_names, "expected at least one bar-driven strategy"
    assert l2_names, "expected at least one L2 strategy"


def test_unknown_param_key_raises_listing_valid_fields():
    _, ConfigClass = get_strategy_class("orb")
    kwargs = _injected_kwargs()
    kwargs["bar_type"] = BarType.from_str("BTCUSDT-PERP.TESTEX-1-HOUR-LAST-EXTERNAL")
    kwargs["totally_bogus_param"] = 3

    with pytest.raises(ValueError) as excinfo:
        _build_strategy_config(ConfigClass, **kwargs)

    message = str(excinfo.value)
    assert "totally_bogus_param" in message
    assert "ORBConfig" in message


def test_injected_fields_have_defaults_except_instrument_id():
    import msgspec

    fields = msgspec.structs.fields(SBTStrategyConfig)
    by_name = {f.name: f for f in fields}
    assert by_name["instrument_id"].required
    for name in (
        "capital",
        "leverage",
        "backtest_start_date",
        "active_from",
        "plugins",
    ):
        assert not by_name[name].required


def test_tier_defaults_match_settled_values():
    import msgspec

    def defaults(config_cls):
        return {
            f.name: f.default
            for f in msgspec.structs.fields(config_cls)
            if not f.required
        }

    base_defaults = defaults(SBTStrategyConfig)
    assert base_defaults["capital"] == Decimal("1000")
    assert base_defaults["leverage"] == 1.0
    assert base_defaults["backtest_start_date"] == "2020-01-01"
    assert base_defaults["active_from"] is None
    assert base_defaults["plugins"] == ()

    bar_fields = msgspec.structs.fields(SBTBarStrategyConfig)
    assert [f.name for f in bar_fields if f.required] == ["instrument_id", "bar_type"]
