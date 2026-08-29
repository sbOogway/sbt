"""RunConfig merge semantics: TOML values survive absent CLI overrides.

Note: ``exchange``, ``symbol``, and ``interval`` are no longer loaded
from TOML — they must come from the CLI (or ``--feather`` inference).
These tests pass them as ``cli_overrides`` to simulate the CLI path.
"""

import argparse
import textwrap
from pathlib import Path

import pytest

from sbt.core.config import (
    RunConfig,
    cli_overrides_from_args,
    param_overrides_from_args,
)


def _write_toml(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(body))
    return path


def test_toml_open_report_false_survives_empty_overrides(tmp_path):
    path = _write_toml(
        tmp_path,
        """
        [run]
        open_report = false
        """,
    )
    cfg = RunConfig.from_toml(
        path,
        strategy_name="orb",
        cli_overrides={"exchange": "hyperliquid", "symbol": "BTC/USDC:USDC"},
    )
    assert cfg.open_report is False


def test_cli_no_open_beats_toml_true(tmp_path):
    path = _write_toml(
        tmp_path,
        """
        [run]
        open_report = true
        """,
    )
    cfg = RunConfig.from_toml(
        path,
        strategy_name="orb",
        cli_overrides={
            "exchange": "hyperliquid",
            "symbol": "BTC/USDC:USDC",
            "open_report": False,
        },
    )
    assert cfg.open_report is False


def test_absent_keys_keep_defaults_and_toml_values(tmp_path):
    path = _write_toml(
        tmp_path,
        """
        [run]
        leverage = 2.5
        open_report = false
        """,
    )
    cfg = RunConfig.from_toml(
        path,
        strategy_name="orb",
        cli_overrides={"exchange": "hyperliquid", "symbol": "BTC/USDC:USDC"},
    )
    assert cfg.leverage == 2.5
    assert cfg.capital == RunConfig.capital  # default untouched


def test_legacy_toml_fields_are_dropped(tmp_path):
    """Old config.toml files with exchange/symbol/interval are silently dropped.

    Ticket #58 Part 2: these fields are now CLI-only. The TOML
    should not carry them. If an old TOML does, they're dropped.
    """
    path = _write_toml(
        tmp_path,
        """
        [run]
        exchange = "hyperliquid"
        symbol = "BTC/USDC:USDC"
        interval = "1h"
        leverage = 2.5
        """,
    )
    # CLI passes exchange/symbol/interval as overrides (as a real CLI would).
    cfg = RunConfig.from_toml(
        path,
        strategy_name="orb",
        cli_overrides={"exchange": "bybit", "symbol": "ETH/USDT:USDT", "interval": "1d"},
    )
    # CLI values win, not the TOML values.
    assert cfg.exchange == "bybit"
    assert cfg.symbol == "ETH/USDT:USDT"
    assert cfg.interval == "1d"
    assert cfg.leverage == 2.5


def test_legacy_db_row_drops_removed_fields(tmp_path):
    """Old BacktestResult rows have exchange/symbol/interval in their config blob.

    ``from_dict`` ignores unknown keys, so they drop silently (per
    the Part 2 grilling decision). The result is still loadable.
    """
    cfg = RunConfig.from_dict(
        {
            "exchange": "bybit",
            "symbol": "BTC/USDT:USDT",
            "interval": "1d",
            "capital": "1000",
            "leverage": 1.0,
            "strategy_name": "orb",
            "exchange": "hyperliquid",  # duplicate, last wins
        }
    )
    # Unknown fields are dropped; known fields are coerced.
    assert cfg.exchange == "hyperliquid"
    assert cfg.symbol == "BTC/USDT:USDT"
    assert cfg.interval == "1d"


# ---------------------------------------------------------------------
# from_cli_args / cli_overrides_from_args / param_overrides_from_args
# ---------------------------------------------------------------------


def _ns(**overrides) -> argparse.Namespace:
    """Build a minimal argparse Namespace mimicking the backtest CLI.

    All optional fields default to None / empty so each test fills in
    only the bits it cares about. ``no_open`` is included because
    :func:`cli_overrides_from_args` reads it.
    """
    base = dict(
        config="config.toml",
        strategy="orb",
        exchange=None,
        symbol=None,
        symbols=None,
        interval=None,
        leverage=None,
        start=None,
        end=None,
        feather=None,
        warmup_bars=None,
        data_type=None,
        l2_max_files=None,
        train_val_split=None,
        no_open=False,
        param=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_overrides_picks_up_required_triplet():
    ns = _ns(exchange="bybit", symbol="BTC/USDT:USDT", interval="1d")
    out = cli_overrides_from_args(ns)
    assert out["exchange"] == "bybit"
    assert out["symbol"] == "BTC/USDT:USDT"
    assert out["symbols"] is None
    assert out["interval"] == "1d"


def test_cli_overrides_flattens_repeated_symbols_specs():
    """`--symbols A,B --symbols C` flattens to [A, B, C]."""
    ns = _ns(
        exchange="bybit",
        symbols=["BTC/USDT:USDT,ETH/USDT:USDT", "SOL/USDT:USDT"],
        interval="1d",
    )
    out = cli_overrides_from_args(ns)
    assert out["symbols"] == ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    assert out["symbol"] is None  # multi-instrument mode


def test_cli_overrides_prefers_symbols_over_symbol():
    """When both --symbol and --symbols are set, --symbols wins (portfolio mode)."""
    ns = _ns(
        exchange="bybit",
        symbol="BTC/USDT:USDT",
        symbols=["ETH/USDT:USDT"],
        interval="1d",
    )
    out = cli_overrides_from_args(ns)
    assert out["symbol"] is None
    assert out["symbols"] == ["ETH/USDT:USDT"]


def test_cli_overrides_no_open_disables_open_report():
    ns = _ns(
        exchange="bybit", symbol="BTC/USDT:USDT", interval="1d", no_open=True
    )
    out = cli_overrides_from_args(ns)
    assert out["open_report"] is False


def test_cli_overrides_missing_args_lists_all():
    """All three missing required args appear in the error."""
    ns = _ns()
    with pytest.raises(ValueError, match="--exchange") as exc:
        cli_overrides_from_args(ns)
    msg = str(exc.value)
    assert "--exchange" in msg
    assert "--symbol" in msg
    assert "--interval" in msg


def test_cli_overrides_partial_missing_lists_only_missing():
    ns = _ns(exchange="bybit", symbol="BTC/USDT:USDT")  # interval missing
    with pytest.raises(ValueError, match="--interval") as exc:
        cli_overrides_from_args(ns)
    assert "--exchange" not in str(exc.value)
    assert "--symbol" not in str(exc.value)
    assert "--interval" in str(exc.value)


def test_cli_overrides_feather_inference_fills_missing(tmp_path):
    """--feather PATH infers exchange/symbol/interval when not set explicitly."""
    feather = tmp_path / "bybit_BTCUSDT:USDT_1d_20240101_20240131.feather"
    feather.write_bytes(b"")
    ns = _ns(feather=str(feather))
    out = cli_overrides_from_args(ns)
    assert out["exchange"] == "bybit"
    assert out["symbol"] == "BTC/USDT:USDT"
    assert out["interval"] == "1d"
    assert out["feather"] == str(feather)


def test_cli_overrides_cli_wins_over_feather_inference(tmp_path):
    """Explicit CLI args override --feather inference."""
    feather = tmp_path / "bybit_BTCUSDT:USDT_1d_20240101_20240131.feather"
    feather.write_bytes(b"")
    ns = _ns(feather=str(feather), exchange="hyperliquid", interval="1h")
    out = cli_overrides_from_args(ns)
    assert out["exchange"] == "hyperliquid"
    assert out["interval"] == "1h"
    # Symbol was not given explicitly -> still inferred from the feather.
    assert out["symbol"] == "BTC/USDT:USDT"


def test_cli_overrides_unparseable_feather_raises_value_error():
    """A non-feather --feather path raises ValueError (CLI layer translates to exit)."""
    ns = _ns(feather="/tmp/random_file.feather")
    with pytest.raises(ValueError, match="Could not infer"):
        cli_overrides_from_args(ns)


def test_param_overrides_parses_each_spec():
    ns = _ns(
        param=[
            "entry_threshold=0.6",
            "lookback=14",
            "use_filter=true",
            "name=foo",
        ]
    )
    out = param_overrides_from_args(ns)
    assert out == {
        "entry_threshold": 0.6,
        "lookback": 14,
        "use_filter": True,
        "name": "foo",
    }


def test_param_overrides_int_wins_over_float():
    """int is tried first; an int-shaped value stays int (not coerced to float)."""
    ns = _ns(param=["size=42"])
    assert param_overrides_from_args(ns) == {"size": 42}


def test_param_overrides_none_when_absent():
    assert param_overrides_from_args(_ns()) == {}


def test_param_overrides_malformed_spec_raises():
    ns = _ns(param=["bad_spec_no_equals"])
    with pytest.raises(ValueError, match="Invalid --param"):
        param_overrides_from_args(ns)


def test_from_cli_args_builds_runconfig(tmp_path):
    """End-to-end: from_cli_args returns a RunConfig with the right fields."""
    path = tmp_path / "config.toml"
    path.write_text(
        textwrap.dedent(
            """
            [run]
            leverage = 2.0
            open_report = true
            """
        )
    )
    ns = _ns(
        config=str(path),
        exchange="bybit",
        symbol="BTC/USDT:USDT",
        interval="1d",
        param=["lookback=21"],
    )
    cfg = RunConfig.from_cli_args(ns)
    assert cfg.exchange == "bybit"
    assert cfg.symbol == "BTC/USDT:USDT"
    assert cfg.interval == "1d"
    # TOML value survives when CLI doesn't override.
    assert cfg.leverage == 2.0
    # --param lands in strategy_params.
    assert cfg.strategy_params == {"lookback": 21}
    # Default open_report from TOML (no --no-open).
    assert cfg.open_report is True


def test_from_cli_args_no_open_overrides_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        textwrap.dedent(
            """
            [run]
            open_report = true
            """
        )
    )
    ns = _ns(
        config=str(path),
        exchange="bybit",
        symbol="BTC/USDT:USDT",
        interval="1d",
        no_open=True,
    )
    cfg = RunConfig.from_cli_args(ns)
    assert cfg.open_report is False


def test_from_cli_args_propagates_value_error(tmp_path):
    """The CLI layer translates this to an exit message; from_cli_args itself raises."""
    ns = _ns()  # no exchange/symbol/interval
    with pytest.raises(ValueError, match="Missing required CLI args"):
        RunConfig.from_cli_args(ns)


def test_from_cli_args_requires_existing_toml(tmp_path):
    ns = _ns(
        config=str(tmp_path / "missing.toml"),
        exchange="bybit",
        symbol="BTC/USDT:USDT",
        interval="1d",
    )
    with pytest.raises(FileNotFoundError):
        RunConfig.from_cli_args(ns)
