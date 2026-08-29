"""Tests for walk-forward validation and rolling optimization."""

from decimal import Decimal
import pandas as pd
import pytest

from sbt.core.config import RunConfig
from sbt.optimize.walk_forward import (
    WalkForwardResult,
    _generate_windows,
    run_walk_forward,
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
