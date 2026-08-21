"""SQLite-backed store for backtest jobs and results."""

import json
import sqlite3
from pathlib import Path

from .config import RunConfig
from .job import BacktestJob, BacktestResult, JobStatus


class ResultStore:
    """Thin wrapper around a SQLite database for persisting jobs and results.

    The same database file can double as an Optuna storage backend
    (``sqlite:///sbt.db``).
    """

    def __init__(self, db_path: str | Path = "sbt.db") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id             TEXT PRIMARY KEY,
                status         TEXT NOT NULL DEFAULT 'pending',
                strategy_name  TEXT NOT NULL,
                config_json    TEXT NOT NULL,
                worker_id      TEXT,
                study_name     TEXT,
                submitted_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS results (
                job_id            TEXT PRIMARY KEY REFERENCES jobs(id),
                status            TEXT NOT NULL,
                sharpe_ratio      REAL,
                num_trades        INTEGER,
                pnl               REAL,
                sqn               REAL,
                stats_json        TEXT,
                equity_curve_json TEXT,
                positions_json    TEXT,
                fills_json        TEXT,
                tearsheet_path    TEXT,
                error             TEXT,
                duration_seconds  REAL DEFAULT 0.0,
                funding_pnl       REAL DEFAULT 0.0
            );
        """)
        existing_cols = {
            row[1] for row in self.conn.execute("PRAGMA table_info(results)").fetchall()
        }
        if "sqn" not in existing_cols:
            self.conn.execute("ALTER TABLE results ADD COLUMN sqn REAL")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def save_job(self, job: BacktestJob) -> None:
        """Insert or replace a job record."""
        self.conn.execute(
            """INSERT OR REPLACE INTO jobs
               (id, status, strategy_name, config_json, worker_id, study_name, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                job.id,
                job.status.value,
                job.config.strategy_name,
                json.dumps(job.config.to_dict()),
                job.worker_id,
                job.study_name,
                job.submitted_at.isoformat(),
            ),
        )
        self.conn.commit()

    def update_job_status(
        self, job_id: str, status: JobStatus, worker_id: str | None = None
    ) -> None:
        self.conn.execute(
            "UPDATE jobs SET status = ?, worker_id = ? WHERE id = ?",
            (status.value, worker_id, job_id),
        )
        self.conn.commit()

    def get_job(self, job_id: str) -> BacktestJob | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def list_jobs(self, status: JobStatus | None = None) -> list[BacktestJob]:
        if status is not None:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY submitted_at DESC",
                (status.value,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY submitted_at DESC",
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def _row_to_job(self, row: sqlite3.Row) -> BacktestJob:
        cfg_dict = json.loads(row["config_json"])
        import datetime

        return BacktestJob(
            config=RunConfig.from_dict(cfg_dict),
            id=row["id"],
            status=JobStatus(row["status"]),
            submitted_at=datetime.datetime.fromisoformat(row["submitted_at"]),
            worker_id=row["worker_id"],
            study_name=row["study_name"],
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def save_result(self, result: BacktestResult) -> None:
        """Insert or replace a result record."""
        self.conn.execute(
            """INSERT OR REPLACE INTO results
               (job_id, status, sharpe_ratio, num_trades, pnl, sqn,
                stats_json, equity_curve_json, positions_json, fills_json,
                tearsheet_path, error, duration_seconds, funding_pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.job_id,
                result.status.value,
                result.sharpe_ratio,
                result.num_trades,
                result.pnl,
                result.sqn,
                json.dumps(result.stats, default=str),
                json.dumps(result.equity_curve, default=str),
                json.dumps(result.positions, default=str),
                json.dumps(result.fills, default=str),
                result.tearsheet_path,
                result.error,
                result.duration_seconds,
                result.funding_pnl,
            ),
        )
        self.conn.commit()

    def get_result(self, job_id: str) -> BacktestResult | None:
        row = self.conn.execute(
            "SELECT * FROM results WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_result(row)

    def get_study_results(self, study_name: str) -> list[BacktestResult]:
        """Fetch all results for jobs belonging to a given Optuna study."""
        rows = self.conn.execute(
            """SELECT r.* FROM results r
               JOIN jobs j ON r.job_id = j.id
               WHERE j.study_name = ?
               ORDER BY r.job_id""",
            (study_name,),
        ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def _row_to_result(self, row: sqlite3.Row) -> BacktestResult:
        return BacktestResult(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            sharpe_ratio=row["sharpe_ratio"],
            num_trades=row["num_trades"],
            pnl=row["pnl"],
            sqn=row["sqn"],
            stats=json.loads(row["stats_json"] or "{}"),
            equity_curve=json.loads(row["equity_curve_json"] or "[]"),
            positions=json.loads(row["positions_json"] or "[]"),
            fills=json.loads(row["fills_json"] or "[]"),
            tearsheet_path=row["tearsheet_path"],
            error=row["error"],
            duration_seconds=row["duration_seconds"] or 0.0,
            funding_pnl=row["funding_pnl"] or 0.0,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()
