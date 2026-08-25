"""Tests for sbt.web: DB query functions and HTTP routes."""

import json
import datetime

import pytest

from sbt.core.db import ResultStore
from sbt.core.job import BacktestJob, BacktestResult, JobStatus
from sbt.core.config import RunConfig


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_config(**overrides) -> RunConfig:
    defaults = dict(
        exchange="HYPERLIQUID",
        symbol="BTC/USDC:USDC",
        interval="1h",
        strategy_name="overnight_drift",
        strategy_params={},
        start="2024-01-01",
        end="2024-02-01",
        open_report=False,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def _make_job(job_id: str = "abc123", **overrides) -> BacktestJob:
    config = overrides.pop("config", _make_config())
    defaults = dict(
        config=config,
        id=job_id,
        status=JobStatus.DONE,
        submitted_at=datetime.datetime(2024, 2, 1, 12, 0, 0),
    )
    defaults.update(overrides)
    return BacktestJob(**defaults)


def _make_result(job_id: str = "abc123", **overrides) -> BacktestResult:
    defaults = dict(
        job_id=job_id,
        status=JobStatus.DONE,
        sharpe_ratio=1.5,
        num_trades=42,
        pnl=1234.56,
        sqn=3.2,
        duration_seconds=99.9,
        funding_pnl=-12.34,
        stats={"PnL (total)": 1234.56, "Sharpe Ratio (252 days)": 1.5},
        positions=[{"side": "BUY", "qty": 1.0}],
        fills=[{"price": 100.0}],
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


@pytest.fixture
def populated_store():
    """An in-memory ResultStore with two jobs + results."""
    store = ResultStore(":memory:")
    job1 = _make_job("aaa111", config=_make_config(strategy_name="overnight_drift"))
    job2 = _make_job("bbb222", config=_make_config(strategy_name="orb"))
    store.save_job(job1)
    store.save_job(job2)
    store._insert_result(_make_result("aaa111", sharpe_ratio=1.5, pnl=1234.56, num_trades=42, sqn=3.2))
    store._insert_result(_make_result("bbb222", sharpe_ratio=0.8, pnl=-200.0, num_trades=10, sqn=1.1))
    yield store
    store.close()


@pytest.fixture
def empty_store():
    store = ResultStore(":memory:")
    yield store
    store.close()


# ── DB query function tests ───────────────────────────────────────────


class TestListResultsForDashboard:
    def test_returns_all_results(self, populated_store):
        from sbt.web.app import list_results_for_dashboard

        rows = list_results_for_dashboard(populated_store.conn)
        assert len(rows) == 2

    def test_row_has_expected_keys(self, populated_store):
        from sbt.web.app import list_results_for_dashboard

        rows = list_results_for_dashboard(populated_store.conn)
        row = rows[0]
        expected_keys = {"job_id", "strategy", "symbol", "sharpe", "pnl", "trades", "sqn", "duration", "date", "status"}
        assert expected_keys == set(row.keys())

    def test_values_match_db(self, populated_store):
        from sbt.web.app import list_results_for_dashboard

        rows = list_results_for_dashboard(populated_store.conn)
        by_id = {r["job_id"]: r for r in rows}
        assert by_id["aaa111"]["strategy"] == "overnight_drift"
        assert by_id["aaa111"]["sharpe"] == pytest.approx(1.5)
        assert by_id["aaa111"]["pnl"] == pytest.approx(1234.56)
        assert by_id["aaa111"]["trades"] == 42
        assert by_id["aaa111"]["sqn"] == pytest.approx(3.2)
        assert by_id["aaa111"]["symbol"] == "BTC/USDC:USDC"

    def test_empty_db_returns_empty_list(self, empty_store):
        from sbt.web.app import list_results_for_dashboard

        rows = list_results_for_dashboard(empty_store.conn)
        assert rows == []

    def test_failed_jobs_excluded(self, populated_store):
        from sbt.web.app import list_results_for_dashboard

        populated_store._insert_result(
            _make_result("ccc333", status=JobStatus.FAILED, sharpe_ratio=None, pnl=None, num_trades=None)
        )
        rows = list_results_for_dashboard(populated_store.conn)
        ids = {r["job_id"] for r in rows}
        assert "ccc333" not in ids


class TestGetResultDetail:
    def test_returns_dict_for_existing_job(self, populated_store):
        from sbt.web.app import get_result_detail

        detail = get_result_detail(populated_store.conn, "aaa111")
        assert detail is not None
        assert detail["job_id"] == "aaa111"
        assert detail["strategy"] == "overnight_drift"
        assert detail["sharpe"] == pytest.approx(1.5)

    def test_returns_none_for_missing_job(self, populated_store):
        from sbt.web.app import get_result_detail

        assert get_result_detail(populated_store.conn, "nonexistent") is None

    def test_includes_stats_dict(self, populated_store):
        from sbt.web.app import get_result_detail

        detail = get_result_detail(populated_store.conn, "aaa111")
        assert "stats" in detail
        assert detail["stats"]["PnL (total)"] == pytest.approx(1234.56)

    def test_includes_positions_and_fills(self, populated_store):
        from sbt.web.app import get_result_detail

        detail = get_result_detail(populated_store.conn, "aaa111")
        assert len(detail["positions"]) == 1
        assert len(detail["fills"]) == 1

    def test_includes_config_info(self, populated_store):
        from sbt.web.app import get_result_detail

        detail = get_result_detail(populated_store.conn, "aaa111")
        assert detail["exchange"] == "HYPERLIQUID"
        assert detail["symbol"] == "BTC/USDC:USDC"
        assert detail["interval"] == "1h"

    def test_has_tearsheet_flag(self, populated_store, tmp_path):
        from sbt.web.app import get_result_detail

        # No tearsheet file exists
        detail = get_result_detail(populated_store.conn, "aaa111", reports_dir=tmp_path)
        assert detail["has_tearsheet"] is False

        # Create a fake tearsheet
        (tmp_path / "tearsheet_aaa111.html").write_text("<html></html>")
        detail = get_result_detail(populated_store.conn, "aaa111", reports_dir=tmp_path)
        assert detail["has_tearsheet"] is True


# ── HTTP route tests ──────────────────────────────────────────────────


class TestRoutes:
    @pytest.fixture
    def client(self, tmp_path):
        """A FastAPI TestClient wired to a temp file DB with test data."""
        from sbt.web.app import create_app

        db_file = tmp_path / "test.db"
        store = ResultStore(str(db_file))
        job1 = _make_job("aaa111", config=_make_config(strategy_name="overnight_drift"))
        job2 = _make_job("bbb222", config=_make_config(strategy_name="orb"))
        store.save_job(job1)
        store.save_job(job2)
        store._insert_result(_make_result("aaa111", sharpe_ratio=1.5, pnl=1234.56, num_trades=42, sqn=3.2))
        store._insert_result(_make_result("bbb222", sharpe_ratio=0.8, pnl=-200.0, num_trades=10, sqn=1.1))
        store.close()

        app = create_app(db_path=str(db_file), reports_dir=tmp_path)
        from starlette.testclient import TestClient

        return TestClient(app)

    def test_list_page_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_list_page_shows_strategies(self, client):
        resp = client.get("/")
        assert "overnight_drift" in resp.text
        assert "orb" in resp.text

    def test_detail_page_returns_200(self, client):
        resp = client.get("/results/aaa111")
        assert resp.status_code == 200

    def test_detail_page_shows_stats(self, client):
        resp = client.get("/results/aaa111")
        assert "1.5" in resp.text  # sharpe ratio

    def test_detail_page_404_for_missing(self, client):
        resp = client.get("/results/nonexistent")
        assert resp.status_code == 404

    def test_reports_static_serves_file(self, client, tmp_path):
        (tmp_path / "tearsheet_aaa111.html").write_text("<html>tearsheet</html>")
        resp = client.get("/reports/tearsheet_aaa111.html")
        assert resp.status_code == 200
        assert "tearsheet" in resp.text
