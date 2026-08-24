"""RunConfig merge semantics: TOML values survive absent CLI overrides."""

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
        exchange = "hyperliquid"
        symbol = "BTC/USDC:USDC"
        open_report = false
        """,
    )
    cfg = RunConfig.from_toml(path, strategy_name="orb", cli_overrides={})
    assert cfg.open_report is False


def test_cli_no_open_beats_toml_true(tmp_path):
    path = _write_toml(
        tmp_path,
        """
        [run]
        exchange = "hyperliquid"
        symbol = "BTC/USDC:USDC"
        open_report = true
        """,
    )
    cfg = RunConfig.from_toml(
        path, strategy_name="orb", cli_overrides={"open_report": False}
    )
    assert cfg.open_report is False


def test_absent_keys_keep_defaults_and_toml_values(tmp_path):
    path = _write_toml(
        tmp_path,
        """
        [run]
        exchange = "hyperliquid"
        symbol = "BTC/USDC:USDC"
        leverage = 2.5
        open_report = false
        """,
    )
    cfg = RunConfig.from_toml(path, strategy_name="orb", cli_overrides={})
    assert cfg.leverage == 2.5
    assert cfg.capital == RunConfig.capital  # default untouched
