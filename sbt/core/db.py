"""SQLite-backed store for backtest jobs and results."""

import json
import sqlite3
from pathlib import Path

from .config import RunConfig
from .job import BacktestJob, BacktestResult, JobStatus, result_field_specs

# Bump when introducing new migrations; see ResultStore._migrate().
_SCHEMA_VERSION = 5


class ResultStore:
    """Thin wrapper around a SQLite database for persisting jobs and results.

    The same database file can double as an Optuna storage backend
    (``sqlite:///sbt.db``).
    """

    def __init__(self, db_path: str | Path = "sbt.db") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL lets readers (client status/list queries, Optuna storage) work
        # while the scheduler writes; busy_timeout avoids spurious
        # "database is locked" under concurrent access.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._create_tables()
        self._migrate()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        result_cols = ",\n                ".join(
            f"{col} {aff}"
            + (" PRIMARY KEY REFERENCES jobs(id)" if name == "job_id" else "")
            + (" NOT NULL" if name == "status" else "")
            for name, col, _kind, aff in result_field_specs()
        )
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id              TEXT PRIMARY KEY,
                status          TEXT NOT NULL DEFAULT 'pending',
                strategy_name   TEXT NOT NULL,
                config_json     TEXT NOT NULL,
                worker_id       TEXT,
                study_name      TEXT,
                submitted_at    TEXT NOT NULL,
                timeout_seconds INTEGER NOT NULL DEFAULT 3600,
                attempts        INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS results (
                {result_cols}
            );
        """)
        self.conn.commit()

    def _migrate(self) -> None:
        """Versioned, idempotent schema migrations.

        v1 -> v2: jobs.timeout_seconds / jobs.attempts (durability).
        v2 -> v3: results columns are derived from the BacktestResult
        dataclass (``core.job.result_field_specs``); any column a new
        field introduces is added generically here, so adding a metric
        field needs no hand-written migration. Legacy columns that no longer
        map to a field (e.g. ``equity_curve_json``, ``tearsheet_path``)
        are left in place on old databases and simply never written.
        v3 -> v4: splits dict replaced by 12 first-class in_sample_*
        and out_of_sample_* scalar columns on results. Legacy
        ``splits_json`` column is left in place on old databases.
        v4 -> v5: results.run_id added (nautilus engine run UUID).
        """
        row = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        version = int(row["value"]) if row else 1

        if version < 2:
            job_cols = {
                r[1] for r in self.conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for col in ("timeout_seconds", "attempts"):
                if col not in job_cols:
                    default = 3600 if col == "timeout_seconds" else 0
                    self.conn.execute(
                        f"ALTER TABLE jobs ADD COLUMN {col} INTEGER NOT NULL DEFAULT {default}"
                    )

        res_cols = {
            r[1] for r in self.conn.execute("PRAGMA table_info(results)").fetchall()
        }
        if res_cols:
            for _name, col, _kind, aff in result_field_specs():
                if col not in res_cols:
                    self.conn.execute(f"ALTER TABLE results ADD COLUMN {col} {aff}")

        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def save_job(self, job: BacktestJob) -> None:
        """Insert or replace a job record."""
        self.conn.execute(
            """INSERT OR REPLACE INTO jobs
               (id, status, strategy_name, config_json, worker_id, study_name,
                submitted_at, timeout_seconds, attempts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.id,
                job.status.value,
                job.config.strategy_name,
                json.dumps(job.config.to_dict()),
                job.worker_id,
                job.study_name,
                job.submitted_at.isoformat(),
                job.timeout_seconds,
                job.attempts,
            ),
        )
        self.conn.commit()

    def save_standalone_job(
        self, job_id: str, strategy_name: str, config_json: str
    ) -> None:
        """Insert a minimal job row for runner-level persistence."""
        from datetime import datetime, timezone

        self.conn.execute(
            """INSERT OR REPLACE INTO jobs
               (id, status, strategy_name, config_json, submitted_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                job_id,
                JobStatus.DONE.value,
                strategy_name,
                config_json,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def reconcile_backlog(self) -> list[BacktestJob]:
        """Startup recovery: return unfinished jobs in FIFO order.

        Jobs stuck in RUNNING from a previous scheduler life are reset to
        PENDING with their worker cleared; they will be re-dispatched like
        any other pending job.
        """
        stale = self.conn.execute(
            "SELECT id FROM jobs WHERE status = ?", (JobStatus.RUNNING.value,)
        ).fetchall()
        for row in stale:
            self.conn.execute(
                "UPDATE jobs SET status = ?, worker_id = NULL WHERE id = ?",
                (JobStatus.PENDING.value, row["id"]),
            )
        self.conn.commit()
        rows = self.conn.execute(
            """SELECT * FROM jobs WHERE status IN (?, ?)
               ORDER BY submitted_at ASC""",
            (JobStatus.PENDING.value, JobStatus.RUNNING.value),
        ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def update_job_status(
        self, job_id: str, status: JobStatus, worker_id: str | None = None
    ) -> None:
        self.conn.execute(
            "UPDATE jobs SET status = ?, worker_id = ? WHERE id = ?",
            (status.value, worker_id, job_id),
        )
        self.conn.commit()

    def update_job_dispatch(self, job: BacktestJob) -> None:
        """Persist dispatch bookkeeping (status, worker, attempt count)."""
        self.conn.execute(
            "UPDATE jobs SET status = ?, worker_id = ?, attempts = ? WHERE id = ?",
            (job.status.value, job.worker_id, job.attempts, job.id),
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
            timeout_seconds=row["timeout_seconds"]
            if "timeout_seconds" in row.keys()
            else 3600,
            attempts=row["attempts"] if "attempts" in row.keys() else 0,
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_result(self, job_id: str) -> BacktestResult | None:
        row = self.conn.execute(
            "SELECT * FROM results WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_result(row)

    def complete_job(
        self, job_id: str, result: BacktestResult, worker_id: str | None = None
    ) -> None:
        """Persist a terminal result and the job's final status atomically."""
        with self.conn:
            self._insert_result(result)
            self.conn.execute(
                "UPDATE jobs SET status = ?, worker_id = ? WHERE id = ?",
                (result.status.value, worker_id, job_id),
            )

    def list_results(self, study_name: str | None = None) -> list[BacktestResult]:
        """All results in one query; optionally only those of an Optuna study."""
        if study_name:
            rows = self.conn.execute(
                """SELECT r.* FROM results r
                   JOIN jobs j ON r.job_id = j.id
                   WHERE j.study_name = ?
                   ORDER BY r.job_id""",
                (study_name,),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM results ORDER BY job_id").fetchall()
        return [self._row_to_result(r) for r in rows]

    def _insert_result(self, result: BacktestResult) -> None:
        specs = result_field_specs()
        cols_sql = ", ".join(col for _, col, _, _ in specs)
        placeholders = ", ".join("?" for _ in specs)
        values = [self._encode_result_field(result, spec) for spec in specs]
        self.conn.execute(
            f"INSERT OR REPLACE INTO results ({cols_sql}) VALUES ({placeholders})",
            tuple(values),
        )
        self.conn.commit()

    @staticmethod
    def _encode_result_field(result: BacktestResult, spec) -> object:
        name, _col, kind, _aff = spec
        value = getattr(result, name)
        if kind == "enum":
            return value.value
        if kind == "json":
            return json.dumps(value, default=str)
        return value

    def _row_to_result(self, row: sqlite3.Row) -> BacktestResult:
        kwargs: dict = {}
        for name, col, kind, _aff in result_field_specs():
            raw = row[col]
            if raw is None or name == "job_id":
                continue
            if kind == "enum":
                kwargs[name] = JobStatus(raw)
            elif kind == "json":
                # Empty/NULL -> leave absent so the dataclass default
                # ({} or [] per annotation) applies.
                if raw:
                    kwargs[name] = json.loads(raw)
            else:
                kwargs[name] = raw
        kwargs["job_id"] = row["job_id"]
        kwargs.setdefault("status", JobStatus.DONE)
        return BacktestResult(**kwargs)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()
