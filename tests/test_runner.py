"""Headless runner tests through the explicit-frame data seam.

These exercise the real engine end-to-end on synthetic bars with zero
filesystem access: ``pd.read_feather`` is monkeypatched to explode so any
accidental fall back to feather discovery fails loudly.
"""

from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
import numpy as np
import pandas as pd
import pytest

import sbt.core.runner as runner_mod
from sbt.core.config import RunConfig
from sbt.core.job import JobStatus
from sbt.core.runner import (
    BacktestRunner,
    _resolve_currency,
    _slice_frame,
    load_bars,
)
from sbt.strategies.base import (
    SBTStrategy,
    SBTPortfolioStrategy,
    SBTPortfolioStrategyConfig,
)
from sbt.strategies.ohlc.orb import ORBConfig
from sbt.utils import make_perpetual

from tests.conftest import make_synthetic_bars


@pytest.fixture(autouse=True)
def no_feather_reads(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("runner attempted a filesystem data read")

    monkeypatch.setattr(pd, "read_feather", _boom)


def test_explicit_bars_end_to_end(synthetic_bars, orb_config):
    result = BacktestRunner(orb_config).run(bars=synthetic_bars)

    assert result.status == JobStatus.DONE, result.error
    assert result.error is None
    assert result.num_trades >= 5, "deterministic ramp should produce breakouts"
    assert isinstance(result.pnl, float)
    assert result.duration_seconds > 0
    assert result.stats


def test_explicit_bars_with_split(orb_config, make_bars):
    cfg = RunConfig(
        **{
            **orb_config.__dict__,
            "train_val_split": 0.7,
            "warmup_bars": 48,
        }
    )
    result = BacktestRunner(cfg).run(bars=make_bars())

    assert result.status == JobStatus.DONE, result.error
    # Per-window metrics are first-class columns, not NULL.
    assert result.in_sample_num_trades is not None
    assert result.out_of_sample_num_trades is not None
    assert result.in_sample_sharpe_ratio is not None
    assert result.out_of_sample_sharpe_ratio is not None
    # OOS metrics are promoted to the top level.
    assert result.num_trades == result.out_of_sample_num_trades


@pytest.mark.parametrize("factor", ["momentum", "volume"])
def test_factor_long_short_portfolio(factor):
    """Registered factor_long_short runs as a weekly rank-sort long-short.

    A basket whose symbols trend at different daily steps produces a clear
    cross-sectional split; the strategy should open long (winners) and short
    (losers) legs. Runs over both supported factors.
    """
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    bars = {
        sym: make_synthetic_bars(days=35, base=100.0, step=s) for sym, s in zip(symbols, steps)
    }
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1h",
        strategy_name="factor_long_short",
        strategy_params={"factor": factor, "lookback_weeks": 1},
        start="2024-01-01",
        end="2024-02-05",
        open_report=False,
    )

    result = BacktestRunner(cfg).run(bars=bars)

    assert result.status == JobStatus.DONE, result.error
    assert result.num_trades >= 2, "expected at least one long and one short fill"
    assert isinstance(result.pnl, float)


def test_factor_long_short_rejects_bad_factor():
    """An unknown factor name fails loudly during strategy construction."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    bars = {sym: make_synthetic_bars(days=10) for sym in symbols}
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1h",
        strategy_name="factor_long_short",
        strategy_params={"factor": "bogus"},
        start="2024-01-01",
        end="2024-01-10",
        open_report=False,
    )

    with pytest.raises(ValueError, match="factor must be one of"):
        BacktestRunner(cfg).run(bars=bars)


def make_daily_trend_bars(days: int, base: float, growth: float, volume: float = 1000.0):
    """One daily OHLC bar per day, closing at ``base*(1+g)^d`` (per-day drift)."""
    rows = []
    for d in range(days):
        ts = pd.Timestamp("2024-01-01 00:00", tz="UTC") + pd.Timedelta(days=d)
        c = round(base * (1 + growth) ** d, 2)
        o = round(c / (1 + growth), 2)
        rows.append({
            "timestamp": ts,
            "open": o,
            "high": round(max(o, c) * 1.01, 2),
            "low": round(min(o, c) * 0.99, 2),
            "close": c,
            "volume": volume,
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "params,reverse",
    [({"reverse": True}, True), ({"reverse": False}, False)],
)
def test_zaremba_reversal_daily(params, reverse):
    """Registered zaremba_reversal builds a daily quintile long-short.

    Symbols with distinct per-day growth give a clear cross-sectional split in
    the lagged daily return; both reversal and momentum directions should open
    a long+short book and rebalance daily.
    """
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    bars = {
        sym: make_daily_trend_bars(days=30, base=100.0, growth=g)
        for sym, g in zip(symbols, growths)
    }
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1d",
        strategy_name="zaremba_reversal",
        strategy_params=params,
        start="2024-01-01",
        end="2024-01-30",
        open_report=False,
    )

    result = BacktestRunner(cfg).run(bars=bars)

    assert result.status == JobStatus.DONE, result.error
    assert result.num_trades >= 2, "expected at least one long and one short fill"
    assert isinstance(result.pnl, float)


def test_zaremba_reversal_liquidity_momentum_subset():
    """Restricting to the top liquid coins (momentum role) runs as a subset."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    # Higher-traffic symbols are the "liquid" ones.
    volumes = [5000.0, 5000.0, 1500.0, 1000.0, 800.0, 500.0]
    bars = {
        sym: make_daily_trend_bars(days=30, base=100.0, growth=0.01, volume=v)
        for sym, v in zip(symbols, volumes)
    }
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1d",
        strategy_name="zaremba_reversal",
        strategy_params={"reverse": False, "liquidity_top_quantile": 0.5},
        start="2024-01-01",
        end="2024-01-30",
        open_report=False,
    )

    result = BacktestRunner(cfg).run(bars=bars)

    assert result.status == JobStatus.DONE, result.error
    assert isinstance(result.pnl, float)


def test_zaremba_reversal_rejects_bad_quantile():
    """An out-of-range liquidity_top_quantile fails loudly during construction."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    bars = {sym: make_daily_trend_bars(days=10, base=100.0, growth=0.01) for sym in symbols}
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1d",
        strategy_name="zaremba_reversal",
        strategy_params={"liquidity_top_quantile": 1.5},
        start="2024-01-01",
        end="2024-01-10",
        open_report=False,
    )

    with pytest.raises(ValueError, match="liquidity_top_quantile"):
        BacktestRunner(cfg).run(bars=bars)


@pytest.mark.parametrize(
    "reverse,expected_long,expected_short",
    [(True, "BTC/USDT:USDT", "LINK/USDT:USDT"),
     (False, "LINK/USDT:USDT", "BTC/USDT:USDT")],
)
def test_zaremba_reversal_direction(reverse, expected_long, expected_short):
    """Reversal longs the worst daily performer and shorts the best; momentum flips it.

    Growths: BTC is the biggest daily loser (-0.02), LINK the biggest winner
    (+0.03); with 6 symbols and top_fraction 0.2 -> one leg on each tail.
    """
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    bars = {
        sym: make_daily_trend_bars(days=30, base=100.0, growth=g)
        for sym, g in zip(symbols, growths)
    }
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1d",
        strategy_name="zaremba_reversal",
        strategy_params={"reverse": reverse},
        start="2024-01-01",
        end="2024-01-30",
        open_report=False,
    )

    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)

    assert result.status == JobStatus.DONE, result.error
    strat = runner.strategy
    sides = {
        iid.value.split(":")[0].replace("USDT", ""): side
        for iid, side in strat.position_map.items()
    }
    assert sides[expected_long.replace("/USDT:USDT", "")] == OrderSide.BUY
    assert sides[expected_short.replace("/USDT:USDT", "")] == OrderSide.SELL


def test_momentum_reversal_jk():
    """Registered momentum_reversal forms a J/K winner-minus-loser book.

    Distinct per-day growth gives a stable J-week formation ranking; the
    (non-overlapping) WML should open long winners and short losers.
    """
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    bars = {
        sym: make_daily_trend_bars(days=60, base=100.0, growth=g)
        for sym, g in zip(symbols, growths)
    }
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1d",
        strategy_name="momentum_reversal",
        strategy_params={"formation_weeks": 2, "holding_weeks": 2},
        start="2024-01-01",
        end="2024-03-01",
        open_report=False,
    )

    result = BacktestRunner(cfg).run(bars=bars)

    assert result.status == JobStatus.DONE, result.error
    assert result.num_trades >= 2, "expected at least one long and one short fill"
    assert isinstance(result.pnl, float)


def test_momentum_reversal_direction():
    """The winner is long and the loser short after formation (2/2)."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    bars = {
        sym: make_daily_trend_bars(days=60, base=100.0, growth=g)
        for sym, g in zip(symbols, growths)
    }
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1d",
        strategy_name="momentum_reversal",
        strategy_params={"formation_weeks": 2, "holding_weeks": 2, "top_bottom": 0.30},
        start="2024-01-01",
        end="2024-03-01",
        open_report=False,
    )

    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)

    assert result.status == JobStatus.DONE, result.error
    sides = {
        iid.value.split(":")[0].replace("USDT", ""): side
        for iid, side in runner.strategy.position_map.items()
    }
    assert sides["LINK"] == OrderSide.BUY   # biggest winner
    assert sides["BTC"] == OrderSide.SELL   # biggest loser


def test_momentum_reversal_overlap_unsupported():
    """Overlapping 1/K tranches are rejected, not silently approximated."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    bars = {sym: make_daily_trend_bars(days=20, base=100.0, growth=0.01) for sym in symbols}
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1d",
        strategy_name="momentum_reversal",
        strategy_params={"overlap": True},
        start="2024-01-01",
        end="2024-01-20",
        open_report=False,
    )

    with pytest.raises(NotImplementedError, match="overlapping"):
        BacktestRunner(cfg).run(bars=bars)


def test_momentum_reversal_rejects_bad_fraction():
    """An out-of-range top_bottom fails loudly during construction."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    bars = {sym: make_daily_trend_bars(days=20, base=100.0, growth=0.01) for sym in symbols}
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1d",
        strategy_name="momentum_reversal",
        strategy_params={"top_bottom": 0.8},
        start="2024-01-01",
        end="2024-01-20",
        open_report=False,
    )

    with pytest.raises(ValueError, match="top_bottom"):
        BacktestRunner(cfg).run(bars=bars)


def test_size_volume_momentum_large_direction():
    """group=large, signal=momentum: momentum on the liquid subset (long best)."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    volumes = [5000.0, 5000.0, 5000.0, 100.0, 100.0, 100.0]  # BTC/ETH/SOL liquid
    bars = {
        sym: make_daily_trend_bars(days=60, base=100.0, growth=g, volume=v)
        for sym, g, v in zip(symbols, growths, volumes)
    }
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="size_volume_momentum",
        strategy_params={"signal": "momentum", "group": "large", "liquid_fraction": 0.5},
        start="2024-01-01", end="2024-03-01", open_report=False,
    )
    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)
    assert result.status == JobStatus.DONE, result.error

    sides = {iid.value.split(":")[0].replace("USDT", ""): side
             for iid, side in runner.strategy.position_map.items()}
    # Among liquid {BTC,ETH,SOL}, SOL is best (long), BTC worst (short); illiquid flat.
    assert sides["SOL"] == OrderSide.BUY
    assert sides["BTC"] == OrderSide.SELL
    assert sides["DOT"] is None


def test_size_volume_momentum_small_direction():
    """group=small, signal=momentum: reversal on the illiquid subset (long worst)."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    volumes = [5000.0, 5000.0, 5000.0, 100.0, 100.0, 100.0]  # ADA/DOT/LINK illiquid
    bars = {
        sym: make_daily_trend_bars(days=60, base=100.0, growth=g, volume=v)
        for sym, g, v in zip(symbols, growths, volumes)
    }
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="size_volume_momentum",
        strategy_params={"signal": "momentum", "group": "small", "liquid_fraction": 0.5},
        start="2024-01-01", end="2024-03-01", open_report=False,
    )
    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)
    assert result.status == JobStatus.DONE, result.error

    sides = {iid.value.split(":")[0].replace("USDT", ""): side
             for iid, side in runner.strategy.position_map.items()}
    # Among illiquid {ADA,DOT,LINK}, reversal longs worst (ADA) and shorts best (LINK).
    assert sides["ADA"] == OrderSide.BUY
    assert sides["LINK"] == OrderSide.SELL
    assert sides["ETH"] is None


def test_size_volume_momentum_high_momentum_runs():
    """signal=high_momentum runs and produces a long+short book."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    bars = {
        sym: make_daily_trend_bars(days=60, base=100.0, growth=g)
        for sym, g in zip(symbols, growths)
    }
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="size_volume_momentum",
        strategy_params={"signal": "high_momentum", "group": "both", "hk_weeks": 2},
        start="2024-01-01", end="2024-03-01", open_report=False,
    )
    result = BacktestRunner(cfg).run(bars=bars)
    assert result.status == JobStatus.DONE, result.error
    assert result.num_trades >= 2
    assert isinstance(result.pnl, float)


def test_size_volume_momentum_rejects_bad_signal():
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    bars = {sym: make_daily_trend_bars(days=20, base=100.0, growth=0.01) for sym in symbols}
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="size_volume_momentum",
        strategy_params={"signal": "bogus"},
        start="2024-01-01", end="2024-01-20", open_report=False,
    )
    with pytest.raises(ValueError, match="signal must be one of"):
        BacktestRunner(cfg).run(bars=bars)


def test_ts_xs_momentum_cs_direction():
    """mode=cs: long top winners / short top losers by trailing return."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    bars = {sym: make_daily_trend_bars(days=60, base=100.0, growth=g)
            for sym, g in zip(symbols, growths)}
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="ts_xs_momentum",
        strategy_params={"mode": "cs", "lookback_days": 14, "holding_days": 7,
                         "top_fraction": 0.3},
        start="2024-01-01", end="2024-03-01", open_report=False,
    )
    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)
    assert result.status == JobStatus.DONE, result.error

    sides = {iid.value.split(":")[0].replace("USDT", ""): side
             for iid, side in runner.strategy.position_map.items()}
    # top 2 winners long (LINK, DOT), bottom 2 losers short (BTC, ETH), middle flat.
    assert sides["LINK"] == OrderSide.BUY
    assert sides["DOT"] == OrderSide.BUY
    assert sides["BTC"] == OrderSide.SELL
    assert sides["ETH"] == OrderSide.SELL
    assert sides["SOL"] is None
    assert sides["ADA"] is None


def test_ts_xs_momentum_ts_long_when_market_positive():
    """mode=ts, rising basket: whole market long."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    bars = {sym: make_daily_trend_bars(days=60, base=100.0, growth=0.02)
            for sym in symbols}
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="ts_xs_momentum",
        strategy_params={"mode": "ts", "lookback_days": 28, "holding_days": 5},
        start="2024-01-01", end="2024-03-01", open_report=False,
    )
    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)
    assert result.status == JobStatus.DONE, result.error

    for iid, side in runner.strategy.position_map.items():
        assert side == OrderSide.BUY


def test_ts_xs_momentum_ts_flat_when_market_negative():
    """mode=ts, falling basket: flat (no long)."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    bars = {sym: make_daily_trend_bars(days=60, base=100.0, growth=-0.02)
            for sym in symbols}
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="ts_xs_momentum",
        strategy_params={"mode": "ts", "lookback_days": 28, "holding_days": 5},
        start="2024-01-01", end="2024-03-01", open_report=False,
    )
    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)
    assert result.status == JobStatus.DONE, result.error
    assert all(side is None for side in runner.strategy.position_map.values())


def test_ts_xs_momentum_rejects_bad_mode():
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    bars = {sym: make_daily_trend_bars(days=20, base=100.0, growth=0.01) for sym in symbols}
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="ts_xs_momentum",
        strategy_params={"mode": "bogus"},
        start="2024-01-01", end="2024-01-20", open_report=False,
    )
    with pytest.raises(ValueError, match="mode must be one of"):
        BacktestRunner(cfg).run(bars=bars)


def test_momentum_winners_long_only():
    """mode long_only: only the top-quintile winner is long; rest flat."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    bars = {sym: make_daily_trend_bars(days=60, base=100.0, growth=g)
            for sym, g in zip(symbols, growths)}
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="momentum_winners",
        strategy_params={"formation_days": 30, "continuation_days": 7,
                         "top_fraction": 0.2, "long_only": True},
        start="2024-01-01", end="2024-03-01", open_report=False,
    )
    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)
    assert result.status == JobStatus.DONE, result.error

    sides = {iid.value.split(":")[0].replace("USDT", ""): side
             for iid, side in runner.strategy.position_map.items()}
    assert sides["LINK"] == OrderSide.BUY      # top quintile (1 of 6) winner
    assert all(sides[s] is None for s in ("BTC", "ETH", "SOL", "ADA", "DOT"))


def test_momentum_winners_wml():
    """mode long_only=False: long top quintile, short bottom quintile."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "ADA/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"]
    growths = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    bars = {sym: make_daily_trend_bars(days=60, base=100.0, growth=g)
            for sym, g in zip(symbols, growths)}
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="momentum_winners",
        strategy_params={"formation_days": 30, "continuation_days": 7,
                         "top_fraction": 0.2, "long_only": False},
        start="2024-01-01", end="2024-03-01", open_report=False,
    )
    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)
    assert result.status == JobStatus.DONE, result.error

    sides = {iid.value.split(":")[0].replace("USDT", ""): side
             for iid, side in runner.strategy.position_map.items()}
    assert sides["LINK"] == OrderSide.BUY      # top winner
    assert sides["BTC"] == OrderSide.SELL      # bottom loser
    assert all(sides[s] is None for s in ("ETH", "SOL", "ADA", "DOT"))


def test_momentum_winners_rejects_bad_top_fraction():
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    bars = {sym: make_daily_trend_bars(days=20, base=100.0, growth=0.01) for sym in symbols}
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="momentum_winners",
        strategy_params={"top_fraction": 0.8},
        start="2024-01-01", end="2024-01-20", open_report=False,
    )
    with pytest.raises(ValueError, match="top_fraction must be"):
        BacktestRunner(cfg).run(bars=bars)


def make_cointegrated_bars(n=160, seed=3, base=100.0):
    """BTC/ETH/BCH/LTC with a clean cointegrated system and a negative tail.

    ETH/BCH/LTC are independent random-walk common trends; BTC = trend1 +
    trend2 + a small persistent mean-reverting residual (so the combo
    BTC-ETH-BCH is stationary). The residual's strongly negative final stretch
    forces a final LONG spread book for the stat-arb test.
    """
    rng = np.random.default_rng(seed)
    g1 = np.cumsum(rng.normal(0, 0.05, n))
    g2 = np.cumsum(rng.normal(0, 0.05, n))
    g3 = np.cumsum(rng.normal(0, 0.05, n))

    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = 0.9 * resid[i - 1] + rng.normal(0, 0.08)
    resid -= resid.mean()
    resid[-15:] -= 2.0  # strongly negative final spread -> LONG book

    logs = {
        "BTC": (g1 + g2 + resid).tolist(),
        "ETH": g1.tolist(),
        "BCH": g2.tolist(),
        "LTC": g3.tolist(),
    }
    start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    bars = {}
    for sym, lp in logs.items():
        rows = []
        for d, v in enumerate(lp):
            c = round(base * np.exp(v), 2)
            o = c * (1 + rng.normal(0, 0.01))
            rows.append({
                "timestamp": start + pd.Timedelta(days=d),
                "open": round(o, 2),
                "high": round(max(o, c) * 1.01, 2),
                "low": round(min(o, c) * 0.99, 2),
                "close": c,
                "volume": 1000.0,
            })
        bars[f"{sym}/USDT:USDT"] = pd.DataFrame(rows)
    return bars


@pytest.mark.parametrize("method", ["engle_granger", "johansen"])
def test_cointegrated_arb_builds_book(method):
    """Stat-arb enters a balanced long/short book when the spread deviates."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "BCH/USDT:USDT", "LTC/USDT:USDT"]
    bars = make_cointegrated_bars()
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="cointegrated_arb",
        strategy_params={"method": method, "estimation_window": 90,
                         "reestimate_every": 20, "entry_z": 0.5, "exit_z": 0.0},
        start="2024-01-01", end="2024-06-15", open_report=False,
    )
    runner = BacktestRunner(cfg)
    result = runner.run(bars=bars)
    assert result.status == JobStatus.DONE, result.error
    assert result.num_trades >= 2

    sides = {iid.value.split(":")[0].replace("USDT", ""): side
             for iid, side in runner.strategy.position_map.items()}
    # Final spread strongly negative => LONG book: primary (weight +1) is long,
    # the negative-weight hedge BCH is short, and the book is balanced (both
    # sides present). Coeff estimates carry short-sample noise, so only assert
    # the robust balanced-book invariants.
    assert sides["BTC"] == OrderSide.BUY
    assert sides["BCH"] == OrderSide.SELL
    longs = [s for s in sides.values() if s == OrderSide.BUY]
    shorts = [s for s in sides.values() if s == OrderSide.SELL]
    assert longs and shorts


def test_cointegrated_arb_rejects_bad_method():
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "BCH/USDT:USDT", "LTC/USDT:USDT"]
    bars = make_cointegrated_bars()
    cfg = RunConfig(
        exchange="TESTEX", symbol=symbols[0], symbols=symbols, interval="1d",
        strategy_name="cointegrated_arb",
        strategy_params={"method": "bogus"},
        start="2024-01-01", end="2024-02-01", open_report=False,
    )
    with pytest.raises(ValueError, match="method must be one of"):
        BacktestRunner(cfg).run(bars=bars)


def test_portfolio_end_to_end(monkeypatch):
    """Multi-symbol portfolio mode runs one engine with N legs on a shared account.

    A probe portfolio strategy opens a long on each leg as its first bar
    arrives; the assertion is that N instruments trade on a single engine
    that shares one venue/account and yields a portfolio-level result.
    """

    class PortfolioLongAll(SBTPortfolioStrategy):
        def __init__(self, config: SBTPortfolioStrategyConfig) -> None:
            super().__init__(config)
            self._entered: set[InstrumentId] = set()

        def on_instrument_bar(self, instrument_id: InstrumentId, bar) -> None:
            if instrument_id not in self._entered and self.trading_active:
                self._entered.add(instrument_id)
                self.open_position(OrderSide.BUY, bar.close.as_double(), instrument_id)

    class PortfolioLongAllConfig(
        SBTPortfolioStrategyConfig, kw_only=True, frozen=True
    ):
        pass

    monkeypatch.setattr(
        runner_mod,
        "get_strategy_class",
        lambda name: (PortfolioLongAll, PortfolioLongAllConfig),
    )

    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    bars = {
        sym: make_synthetic_bars(days=10, base=100.0 + i * 10) for i, sym in enumerate(symbols)
    }
    cfg = RunConfig(
        exchange="TESTEX",
        symbol=symbols[0],
        symbols=symbols,
        interval="1h",
        strategy_name="portfolio_probe",
        start="2024-01-01",
        end="2024-01-10",
        open_report=False,
    )

    result = BacktestRunner(cfg).run(bars=bars)

    assert result.status == JobStatus.DONE, result.error
    assert result.num_trades >= 3, "expected at least one fill per leg"
    assert isinstance(result.pnl, float)


def test_injected_funding_reaches_engine(synthetic_bars, orb_config, monkeypatch):
    """A funding frame provided at run() flows into the engine side-channel.

    A probe strategy opts in via ``subscribe_funding`` on its config (the
    base subscribes in ``on_start``) and holds a synthetic long, accruing
    deterministically on every bar while a position is notionally open.
    """

    class FundingProbe(SBTStrategy):
        def on_start(self) -> None:
            super().on_start()
            self.position_side = OrderSide.BUY
            self._open_qty = Quantity(1.0, precision=3)

        def on_trading_bar(self, bar) -> None:
            if self.position_side is not None:
                self.funding.accrue(
                    self.position_side,
                    self._open_qty,
                    bar.close.as_double(),
                    0.0001,
                )

    monkeypatch.setattr(
        runner_mod, "get_strategy_class", lambda name: (FundingProbe, ORBConfig)
    )
    cfg = RunConfig(
        **{
            **orb_config.__dict__,
            "strategy_params": {
                **orb_config.strategy_params,
                "subscribe_funding": True,
            },
        }
    )

    funding = pd.DataFrame(
        {
            "timestamp": synthetic_bars["timestamp"],
            "funding_rate": 0.0001,
        }
    )
    result = BacktestRunner(cfg).run(bars=synthetic_bars, funding=funding)

    assert result.status == JobStatus.DONE, result.error
    assert result.funding_pnl > 0


def test_too_few_bars_fails_cleanly(orb_config, make_bars):
    tiny = make_bars(days=1).iloc[:1]
    result = BacktestRunner(orb_config).run(bars=tiny)

    assert result.status == JobStatus.FAILED
    assert "Not enough bars" in result.error


def test_missing_column_fails_cleanly(orb_config, make_bars):
    bad = make_bars(days=2).drop(columns=["volume"])
    result = BacktestRunner(orb_config).run(bars=bad)

    assert result.status == JobStatus.FAILED
    assert "missing expected columns" in result.error


def test_l2_mode_still_requires_catalog(orb_config, make_bars):
    cfg = RunConfig(
        **{**orb_config.__dict__, "data_type": "l2", "data_dir": "/nonexistent"}
    )
    result = BacktestRunner(cfg).run(bars=make_bars(days=2))

    assert result.status == JobStatus.FAILED
    assert "L2" in result.error


# ---------------------------------------------------------------------
# Pure helper units
# ---------------------------------------------------------------------


def test_resolve_currency_known_and_unknown():
    usdt = _resolve_currency("USDT")
    assert usdt.code == "USDT"
    fake = _resolve_currency("FAKE")
    assert fake.code == "FAKE"


def test_slice_frame_bounds(make_bars):
    df = make_bars(days=3)
    start = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    end = pd.Timestamp("2024-01-02 05:00", tz="UTC")
    out = _slice_frame(df, start, end)

    assert len(out) == 6
    assert out["timestamp"].iloc[0] == start
    assert out["timestamp"].iloc[-1] == end


def test_load_bars_conversions(make_bars):
    instrument = make_perpetual("TESTEX", "BTC/USDT:USDT")
    bar_type = BarType.from_str(f"{instrument.id.value}-1-HOUR-LAST-EXTERNAL")
    df = make_bars(days=1)

    bars = load_bars(df, bar_type, instrument)

    assert len(bars) == len(df)
    assert bars[0].close.as_double() == pytest.approx(df["close"].iloc[0], abs=0.06)


# ---------------------------------------------------------------------
# Result persistence via db_path
# ---------------------------------------------------------------------


class TestRunnerPersistence:
    """BacktestRunner persists BacktestResult when db_path is set."""

    def test_persists_successful_result(self, synthetic_bars, orb_config, tmp_path):
        db = str(tmp_path / "test.db")
        result = BacktestRunner(orb_config, db_path=db).run(
            job_id="persist-001", bars=synthetic_bars
        )

        assert result.status == JobStatus.DONE
        from sbt.core.db import ResultStore

        store = ResultStore(db)
        stored = store.get_result("persist-001")
        store.close()

        assert stored is not None
        assert stored.job_id == "persist-001"
        assert stored.status == JobStatus.DONE
        assert stored.sharpe_ratio == pytest.approx(result.sharpe_ratio)
        assert stored.pnl == pytest.approx(result.pnl)
        assert stored.num_trades == result.num_trades

    def test_persists_failed_result(self, orb_config, tmp_path):
        db = str(tmp_path / "test.db")
        tiny = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
                "volume": [1.0],
            }
        )
        result = BacktestRunner(orb_config, db_path=db).run(
            job_id="fail-001", bars=tiny
        )

        assert result.status == JobStatus.FAILED

        from sbt.core.db import ResultStore

        store = ResultStore(db)
        stored = store.get_result("fail-001")
        store.close()

        assert stored is not None
        assert stored.status == JobStatus.FAILED
        assert stored.error is not None

    def test_no_persist_when_db_path_none(self, synthetic_bars, orb_config, tmp_path):
        db = str(tmp_path / "test.db")
        result = BacktestRunner(orb_config, db_path=None).run(
            job_id="no-persist-001", bars=synthetic_bars
        )

        assert result.status == JobStatus.DONE

        from sbt.core.db import ResultStore

        store = ResultStore(db)
        stored = store.get_result("no-persist-001")
        store.close()

        assert stored is None

    def test_persists_with_train_val_split(self, orb_config, tmp_path):
        db = str(tmp_path / "test.db")
        cfg = RunConfig(
            **{
                **orb_config.__dict__,
                "train_val_split": 0.7,
                "warmup_bars": 48,
            }
        )
        result = BacktestRunner(cfg, db_path=db).run(
            job_id="split-001",
            bars=make_synthetic_bars(),
        )

        assert result.status == JobStatus.DONE

        from sbt.core.db import ResultStore

        store = ResultStore(db)
        stored = store.get_result("split-001")
        store.close()

        assert stored is not None
        assert stored.num_trades == result.num_trades
        assert stored.pnl == pytest.approx(result.pnl)
