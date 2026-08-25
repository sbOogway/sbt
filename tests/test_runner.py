"""Headless runner tests through the explicit-frame data seam.

These exercise the real engine end-to-end on synthetic bars with zero
filesystem access: ``pd.read_feather`` is monkeypatched to explode so any
accidental fall back to feather discovery fails loudly.
"""

from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
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
from sbt.strategies.base import SBTStrategy
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
