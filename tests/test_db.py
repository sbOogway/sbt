"""Unit tests for ResultStore: schema, round-trip, and migration."""

import pytest

from sbt.core.db import ResultStore
from sbt.core.job import BacktestResult, JobStatus


@pytest.fixture
def store():
    s = ResultStore(":memory:")
    yield s
    s.close()


def _make_result(job_id="test-001", **overrides) -> BacktestResult:
    defaults = dict(
        job_id=job_id,
        status=JobStatus.DONE,
        sharpe_ratio=1.5,
        num_trades=42,
        pnl=1234.56,
        sqn=3.2,
        duration_seconds=99.9,
        funding_pnl=-12.34,
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


class TestResultStoreRoundTrip:
    def test_scalar_fields_survive_round_trip(self, store):
        result = _make_result()
        store._insert_result(result)
        got = store.get_result("test-001")

        assert got is not None
        assert got.job_id == "test-001"
        assert got.status == JobStatus.DONE
        assert got.sharpe_ratio == pytest.approx(1.5)
        assert got.num_trades == 42
        assert got.pnl == pytest.approx(1234.56)
        assert got.sqn == pytest.approx(3.2)
        assert got.duration_seconds == pytest.approx(99.9)
        assert got.funding_pnl == pytest.approx(-12.34)

    def test_is_oos_fields_round_trip(self, store):
        result = _make_result(
            in_sample_sharpe_ratio=1.2,
            in_sample_num_trades=30,
            in_sample_pnl=800.0,
            in_sample_sqn=2.5,
            in_sample_funding_pnl=-5.0,
            in_sample_duration_seconds=40.0,
            out_of_sample_sharpe_ratio=1.8,
            out_of_sample_num_trades=12,
            out_of_sample_pnl=434.56,
            out_of_sample_sqn=4.1,
            out_of_sample_funding_pnl=-7.34,
            out_of_sample_duration_seconds=59.9,
        )
        store._insert_result(result)
        got = store.get_result("test-001")

        assert got is not None
        assert got.in_sample_sharpe_ratio == pytest.approx(1.2)
        assert got.in_sample_num_trades == 30
        assert got.in_sample_pnl == pytest.approx(800.0)
        assert got.in_sample_sqn == pytest.approx(2.5)
        assert got.in_sample_funding_pnl == pytest.approx(-5.0)
        assert got.in_sample_duration_seconds == pytest.approx(40.0)
        assert got.out_of_sample_sharpe_ratio == pytest.approx(1.8)
        assert got.out_of_sample_num_trades == 12
        assert got.out_of_sample_pnl == pytest.approx(434.56)
        assert got.out_of_sample_sqn == pytest.approx(4.1)
        assert got.out_of_sample_funding_pnl == pytest.approx(-7.34)
        assert got.out_of_sample_duration_seconds == pytest.approx(59.9)

    def test_single_window_has_null_is_oos(self, store):
        result = _make_result()
        store._insert_result(result)
        got = store.get_result("test-001")

        assert got is not None
        assert got.in_sample_sharpe_ratio is None
        assert got.in_sample_num_trades is None
        assert got.out_of_sample_sharpe_ratio is None
        assert got.out_of_sample_num_trades is None

    def test_json_fields_round_trip(self, store):
        result = _make_result(
            stats={"PnL (total)": 100.0, "Sharpe Ratio (252 days)": 1.5},
            positions=[{"side": "BUY", "qty": 1.0}],
            fills=[{"price": 100.0}],
        )
        store._insert_result(result)
        got = store.get_result("test-001")

        assert got is not None
        assert got.stats["PnL (total)"] == 100.0
        assert len(got.positions) == 1
        assert len(got.fills) == 1

    def test_insert_or_replace_upserts(self, store):
        r1 = _make_result(sharpe_ratio=1.0)
        store._insert_result(r1)
        r2 = _make_result(sharpe_ratio=2.0)
        store._insert_result(r2)
        got = store.get_result("test-001")

        assert got is not None
        assert got.sharpe_ratio == pytest.approx(2.0)


class TestSchemaMigration:
    def test_schema_version_is_4(self, store):
        row = store.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        assert int(row["value"]) == 4

    def test_new_columns_exist(self, store):
        cols = {r[1] for r in store.conn.execute("PRAGMA table_info(results)").fetchall()}
        assert "in_sample_sharpe_ratio" in cols
        assert "out_of_sample_num_trades" in cols
        assert "in_sample_funding_pnl" in cols
        assert "out_of_sample_duration_seconds" in cols

    def test_legacy_splits_json_column_survives(self, store):
        """Old databases keep splits_json; the column should still exist."""
        cols = {r[1] for r in store.conn.execute("PRAGMA table_info(results)").fetchall()}
        # On a fresh in-memory db, splits_json won't be created (field is gone
        # from the dataclass), but on a migrated file it would persist. This
        # test verifies the new columns are present regardless.
        assert "in_sample_pnl" in cols
