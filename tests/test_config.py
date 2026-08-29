"""RunConfig merge semantics: TOML values survive absent CLI overrides.

Note: ``exchange``, ``symbol``, and ``interval`` are no longer loaded
from TOML — they must come from the CLI (or ``--feather`` inference).
These tests pass them as ``cli_overrides`` to simulate the CLI path.
"""

import textwrap

from sbt.core.config import RunConfig


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
