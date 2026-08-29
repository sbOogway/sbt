"""Tests for walk-forward validation and rolling optimization."""

import argparse
import textwrap
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sbt.core.config import RunConfig
from sbt.optimize.walk_forward import (
    WalkForwardResult,
    _generate_windows,
    run_walk_forward,
    run_walk_forward_from_args,
)
from tests.conftest import make_synthetic_bars


def test_generate_windows():
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2021-01-01", tz="UTC")

    windows = _generate_windows(start, end, is_months=6, oos_months=2, step_months=2)
    assert len(windows) >= 3

    # First window: IS 6 months, OOS 2 months
    is_start, is_end, oos_start, oos_end = windows[0]
    assert is_start == start
    assert (is_end - is_start).days in (181, 182)
    assert oos_start == is_end + pd.Timedelta(days=1)
    assert oos_end <= end

    # Second window advances by step_months (2 months)
    is_start2, _, _, _ = windows[1]
    assert (is_start2 - is_start).days in (59, 60, 61)


def test_walk_forward_result_summary():
    windows = [
        {"status": "done", "oos_sharpe": 1.5, "oos_return_pct": 5.0, "oos_trades": 10, "oos_pnl": 50.0},
        {"status": "done", "oos_sharpe": -0.5, "oos_return_pct": -2.0, "oos_trades": 8, "oos_pnl": -20.0},
        {"status": "done", "oos_sharpe": 2.0, "oos_return_pct": 8.0, "oos_trades": 12, "oos_pnl": 80.0},
    ]
    res = WalkForwardResult(
        strategy="orb",
        windows=windows,
        mean_oos_sharpe=1.0,
        mean_oos_return_pct=3.67,
        consistency=2 / 3,
        worst_oos_sharpe=-0.5,
        total_oos_trades=30,
        total_windows=3,
        profitable_windows=2,
    )
    summary = res.summary_line()
    assert "Strategy: orb" in summary
    assert "Windows: 3" in summary
    assert "Mean OOS Sharpe: +1.000" in summary
    assert "Consistency: 67%" in summary
    assert "Worst OOS Sharpe: -0.500" in summary
    assert "Total OOS Trades: 30" in summary


def test_run_walk_forward_end_to_end():
    # 120 days of hourly bars gives ~4 months of data
    bars = make_synthetic_bars(days=120, base=100.0)
    cfg = RunConfig(
        exchange="TESTEX",
        symbol="BTC/USDT:USDT",
        interval="1h",
        strategy_name="orb",
        start="2024-01-01",
        end="2024-05-01",
        open_report=False,
    )

    result = run_walk_forward(
        cfg,
        is_months=2,
        oos_months=1,
        step_months=1,
        trials=2,
        param_space=[
            "orb_period=int(1,3)",
            "atr_period=int(7,14)",
        ],
        bars=bars,
    )

    assert isinstance(result, WalkForwardResult)
    assert result.strategy == "orb"
    assert result.total_windows >= 1
    assert len(result.windows) >= 1
    completed = [w for w in result.windows if w["status"] == "done"]
    assert len(completed) >= 1
    assert result.total_oos_trades >= 0


def test_run_walk_forward_unregistered_strategy_raises():
    bars = make_synthetic_bars(days=60)
    cfg = RunConfig(
        exchange="TESTEX",
        symbol="BTC/USDT:USDT",
        interval="1h",
        strategy_name="totally_unknown_strategy_xyz",
        start="2024-01-01",
        end="2024-03-01",
        open_report=False,
    )
    with pytest.raises(ValueError, match="No param space registered"):
        run_walk_forward(cfg, bars=bars)


# ---------------------------------------------------------------------
# run_walk_forward_from_args — CLI glue
# ---------------------------------------------------------------------


def _wf_args(
    config_path: str,
    *,
    exchange: str = "bybit",
    symbol: str = "BTC/USDT:USDT",
    interval: str = "1h",
    is_months: int = 2,
    oos_months: int = 1,
    step_months: int = 1,
    trials: int = 2,
    param: list[str] | None = None,
) -> argparse.Namespace:
    """Build a Namespace shaped like the ``sbt backtest --walk-forward`` CLI."""
    return argparse.Namespace(
        config=config_path,
        strategy="orb",
        exchange=exchange,
        symbol=symbol,
        symbols=None,
        interval=interval,
        leverage=None,
        start=None,
        end=None,
        feather=None,
        warmup_bars=None,
        data_type=None,
        l2_max_files=None,
        train_val_split=None,
        no_open=True,
        param=param,
        walk_forward=True,
        wf_is_months=is_months,
        wf_oos_months=oos_months,
        wf_step_months=step_months,
        wf_trials=trials,
    )


def _write_orb_toml(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            [run]
            open_report = true
            """
        )
    )


def test_run_walk_forward_from_args_end_to_end(tmp_path, monkeypatch):
    """Full flow: builds cfg from args, runs walk-forward, prints summary."""
    cfg_path = tmp_path / "config.toml"
    _write_orb_toml(cfg_path)
    bars = make_synthetic_bars(days=120, base=100.0)

    args = _wf_args(
        str(cfg_path),
        param=["orb_period=int(1,3)", "atr_period=int(7,14)"],
    )
    monkeypatch.setattr(
        "sbt.core.config.RunConfig.from_toml",
        lambda *_a, **_kw: RunConfig(
            exchange="bybit",
            symbol="BTC/USDT:USDT",
            interval="1h",
            strategy_name="orb",
            start="2024-01-01",
            end="2024-05-01",
            open_report=False,
        ),
    )

    with patch(
        "sbt.core.runner._discover_bars",
        return_value=(bars, None),
    ):
        result = run_walk_forward_from_args(args)

    assert isinstance(result, WalkForwardResult)
    assert result.strategy == "orb"
    assert result.total_windows >= 1
    completed = [w for w in result.windows if w["status"] == "done"]
    assert len(completed) >= 1


def test_run_walk_forward_from_args_propagates_value_error(tmp_path):
    """Missing required CLI args (no exchange/symbol/interval) raise ValueError.

    The CLI layer translates this to an exit message; the helper itself
    is pure and propagates.
    """
    cfg_path = tmp_path / "config.toml"
    _write_orb_toml(cfg_path)
    args = _wf_args(
        str(cfg_path),
        exchange="",  # missing
        symbol="",    # missing
        interval="",  # missing
    )
    with pytest.raises(ValueError, match="Missing required CLI args"):
        run_walk_forward_from_args(args)


def test_run_walk_forward_from_args_uses_arg_knobs(tmp_path, monkeypatch):
    """The four --wf-* knobs on the Namespace flow into run_walk_forward."""
    cfg_path = tmp_path / "config.toml"
    _write_orb_toml(cfg_path)
    bars = make_synthetic_bars(days=180, base=100.0)
    args = _wf_args(
        str(cfg_path),
        is_months=3,
        oos_months=1,
        step_months=1,
        trials=2,
        param=["orb_period=int(1,3)", "atr_period=int(7,14)"],
    )

    captured: dict = {}

    def fake_run_walk_forward(cfg, **_kwargs):
        captured["cfg"] = cfg
        captured["kwargs"] = _kwargs
        return WalkForwardResult(
            strategy=cfg.strategy_name, windows=[], total_windows=0
        )

    monkeypatch.setattr(
        "sbt.core.config.RunConfig.from_toml",
        lambda *_a, **_kw: RunConfig(
            exchange="bybit",
            symbol="BTC/USDT:USDT",
            interval="1h",
            strategy_name="orb",
            start="2024-01-01",
            end="2024-07-01",
            open_report=False,
        ),
    )
    monkeypatch.setattr(
        "sbt.core.runner._discover_bars",
        lambda *_a, **_kw: (bars, None),
    )
    monkeypatch.setattr(
        "sbt.optimize.walk_forward.run_walk_forward", fake_run_walk_forward
    )

    run_walk_forward_from_args(args)

    assert captured["kwargs"]["is_months"] == 3
    assert captured["kwargs"]["oos_months"] == 1
    assert captured["kwargs"]["step_months"] == 1
    assert captured["kwargs"]["trials"] == 2
    assert captured["kwargs"]["param_space"] == [
        "orb_period=int(1,3)",
        "atr_period=int(7,14)",
    ]


def test_run_walk_forward_from_args_prints_summary_line(tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "config.toml"
    _write_orb_toml(cfg_path)
    args = _wf_args(
        str(cfg_path),
        param=["orb_period=int(1,3)", "atr_period=int(7,14)"],
    )
    monkeypatch.setattr(
        "sbt.core.config.RunConfig.from_toml",
        lambda *_a, **_kw: RunConfig(
            exchange="bybit",
            symbol="BTC/USDT:USDT",
            interval="1h",
            strategy_name="orb",
            start="2024-01-01",
            end="2024-05-01",
            open_report=False,
        ),
    )
    monkeypatch.setattr(
        "sbt.core.runner._discover_bars",
        lambda *_a, **_kw: (make_synthetic_bars(days=120, base=100.0), None),
    )

    run_walk_forward_from_args(args)
    out = capsys.readouterr().out
    assert "Strategy: orb" in out
    assert "Mean OOS Sharpe" in out
    assert "Consistency" in out
