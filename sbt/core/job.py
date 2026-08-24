"""Job and result models for the backtest scheduler."""

import datetime
import dataclasses
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import get_args, get_origin, get_type_hints, Union

from .config import RunConfig


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class BacktestJob:
    """A unit of work submitted to the scheduler."""

    config: RunConfig
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.PENDING
    submitted_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
    )
    worker_id: str | None = None
    study_name: str | None = None
    # Hard wall-clock budget for one execution attempt; the scheduler's
    # reaper kills the worker and requeues/fails the job past this point.
    timeout_seconds: int = 3600
    # Number of times this job has been handed to a worker. Requeues
    # (ACK timeout, worker death) increment it; jobs exceeding the
    # scheduler's max-attempts policy are failed instead of retried.
    attempts: int = 0

    def to_dict(self) -> dict:
        """Convert BacktestJob to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "worker_id": self.worker_id,
            "study_name": self.study_name,
            "timeout_seconds": self.timeout_seconds,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BacktestJob:
        """Reconstruct BacktestJob from a dictionary."""
        sub_at = d.get("submitted_at")
        if isinstance(sub_at, str):
            submitted_at = datetime.datetime.fromisoformat(sub_at)
        elif isinstance(sub_at, datetime.datetime):
            submitted_at = sub_at
        else:
            submitted_at = datetime.datetime.now(datetime.UTC)

        return cls(
            config=RunConfig.from_dict(d["config"]),
            id=d["id"],
            status=JobStatus(d.get("status", "pending")),
            submitted_at=submitted_at,
            worker_id=d.get("worker_id"),
            study_name=d.get("study_name"),
            timeout_seconds=int(d.get("timeout_seconds", 3600)),
            attempts=int(d.get("attempts", 0)),
        )


@dataclass
class BacktestResult:
    """Structured output of a single backtest run.

    The optimisation objectives (sharpe_ratio, num_trades, pnl, sqn)
    are promoted to top-level fields for Optuna multi-objective studies.
    The full engine stats dict is kept in *stats* for reporting.

    Serialization is fields-derived (same contract as ``RunConfig``):
    :meth:`to_dict` / :meth:`from_dict` walk ``dataclasses.fields`` and
    coerce the annotated types — adding a metric field never requires
    touching the codec, the results DDL, or the row mappers.
    """

    job_id: str
    status: JobStatus
    # --- optimisation objectives ---
    sharpe_ratio: float | None = None
    num_trades: int | None = None
    pnl: float | None = None
    sqn: float | None = None
    # --- full engine output ---
    stats: dict = field(default_factory=dict)
    positions: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0
    funding_pnl: float = 0.0
    # Per-window metrics when a runner plugin split the job into windows
    # (e.g. train/val holdout): {"in_sample": {...}, "out_of_sample": {...}}.
    splits: dict = field(default_factory=dict)
    # When positions/fills exceed the inline-row budget they are spilled to
    # parquet under reports/artifacts/{job_id}/ and carried by path + count;
    # the inline lists stay empty in that case.
    positions_path: str | None = None
    fills_path: str | None = None
    positions_count: int | None = None
    fills_count: int | None = None

    def to_dict(self) -> dict:
        """Convert BacktestResult to a JSON-serializable dictionary."""
        hints = get_type_hints(type(self))
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if hints[f.name] is JobStatus:
                value = value.value
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, d: dict) -> BacktestResult:
        """Reconstruct BacktestResult from a dictionary."""
        hints = get_type_hints(cls)
        kwargs: dict = {"job_id": d["job_id"]}
        for f in dataclasses.fields(cls):
            if f.name == "job_id" or f.name not in d or d[f.name] is None:
                continue
            tp = hints[f.name]
            kwargs[f.name] = JobStatus(d[f.name]) if tp is JobStatus else d[f.name]
        kwargs.setdefault("status", JobStatus.DONE)
        return cls(**kwargs)


def result_field_specs() -> list[tuple[str, str, str, str]]:
    """Storage layout derived from :class:`BacktestResult` fields.

    Yields ``(field_name, column_name, kind, affinity)`` where kind is
    ``"enum"``/``"json"``/``"scalar"`` and affinity is the SQLite type.
    dict/list fields persist as ``<name>_json`` TEXT columns; scalars get
    one queryable column each. This is the single source of truth shared
    by the DDL, migrations, insert, and row decode in ``core.db``.
    """
    hints = get_type_hints(BacktestResult)
    specs = []
    for f in dataclasses.fields(BacktestResult):
        name = f.name
        tp = hints[name]
        origin = get_origin(tp)
        if tp is JobStatus:
            specs.append((name, name, "enum", "TEXT"))
            continue
        if tp in (dict, list) or origin in (dict, list):
            specs.append((name, f"{name}_json", "json", "TEXT"))
            continue
        if origin is Union:
            args = [a for a in get_args(tp) if a is not type(None)]
            if len(args) == 1:
                tp = args[0]
        affinity = {float: "REAL", int: "INTEGER", str: "TEXT"}.get(tp, "TEXT")
        specs.append((name, name, "scalar", affinity))
    return specs
