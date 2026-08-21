"""Job and result models for the backtest scheduler."""

import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum

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

    def to_dict(self) -> dict:
        """Convert BacktestJob to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "worker_id": self.worker_id,
            "study_name": self.study_name,
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
        )


@dataclass
class BacktestResult:
    """Structured output of a single backtest run.

    The three optimisation objectives (sharpe_ratio, num_trades, pnl)
    are promoted to top-level fields for Optuna multi-objective studies.
    The full engine stats dict is kept in *stats* for reporting.
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
    equity_curve: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    tearsheet_path: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    funding_pnl: float = 0.0
    # Per-window metrics when a runner plugin split the job into windows
    # (e.g. train/val holdout): {"in_sample": {...}, "out_of_sample": {...}}.
    splits: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert BacktestResult to a JSON-serializable dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "sharpe_ratio": self.sharpe_ratio,
            "num_trades": self.num_trades,
            "pnl": self.pnl,
            "sqn": self.sqn,
            "stats": self.stats,
            "equity_curve": self.equity_curve,
            "positions": self.positions,
            "fills": self.fills,
            "tearsheet_path": self.tearsheet_path,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "funding_pnl": self.funding_pnl,
            "splits": self.splits,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BacktestResult:
        """Reconstruct BacktestResult from a dictionary."""
        return cls(
            job_id=d["job_id"],
            status=JobStatus(d.get("status", "done")),
            sharpe_ratio=d.get("sharpe_ratio"),
            num_trades=d.get("num_trades"),
            pnl=d.get("pnl"),
            sqn=d.get("sqn"),
            stats=d.get("stats", {}),
            equity_curve=d.get("equity_curve", []),
            positions=d.get("positions", []),
            fills=d.get("fills", []),
            tearsheet_path=d.get("tearsheet_path"),
            error=d.get("error"),
            duration_seconds=d.get("duration_seconds", 0.0),
            funding_pnl=d.get("funding_pnl", 0.0),
            splits=d.get("splits", {}),
        )
