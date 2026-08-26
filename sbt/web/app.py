"""FastAPI app for browsing backtest results and tearsheets."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_TEMPLATE_DIR = Path(__file__).parent / "templates"


# ── DB query helpers ──────────────────────────────────────────────────


def list_results_for_dashboard(conn: sqlite3.Connection) -> list[dict]:
    """Return one dict per completed result for the list view.

    Excludes failed jobs (NULL sharpe/pnl) and joins config metadata
    from the jobs table.
    """
    rows = conn.execute(
        """
        SELECT r.job_id, r.status, r.sharpe_ratio, r.pnl, r.num_trades,
               r.sqn, r.duration_seconds, r.funding_pnl, r.run_id,
               j.strategy_name, j.config_json, j.submitted_at
        FROM results r
        JOIN jobs j ON r.job_id = j.id
        WHERE r.sharpe_ratio IS NOT NULL
        ORDER BY j.submitted_at DESC
        """
    ).fetchall()
    results = []
    for row in rows:
        cfg = json.loads(row["config_json"])
        capital = float(cfg.get("capital", 0) or 0)
        pnl = row["pnl"]
        pnl_pct = (pnl / capital * 100) if capital and pnl is not None else None
        results.append(
            {
                "job_id": row["job_id"],
                "run_id": row["run_id"],
                "display_id": row["run_id"] or row["job_id"],
                "strategy": row["strategy_name"],
                "symbol": cfg.get("symbol", ""),
                "sharpe": row["sharpe_ratio"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "trades": row["num_trades"],
                "sqn": row["sqn"],
                "duration": row["duration_seconds"],
                "date": row["submitted_at"][:10],
                "status": row["status"],
            }
        )
    return results


def _find_tearsheet(reports_dir: Path, tearsheet_id: str) -> str | None:
    """Return the tearsheet filename if it exists, else None."""
    name = f"tearsheet_{tearsheet_id}.html"
    if (reports_dir / name).exists():
        return name
    return None


def _find_tearsheet_by_job(
    reports_dir: Path, submitted_at: str
) -> str | None:
    """Fallback: scan reports dir for a tearsheet matching the job timestamp.

    When ``run_id`` is NULL (stale row from before the run_id feature),
    the UUID-based filename is lost.  This scans for tearsheets whose
    mtime falls within ±5 minutes of the job's ``submitted_at`` and
    returns the closest match.  Returns ``None`` when no match is found.
    """
    from datetime import datetime, timedelta, timezone

    try:
        job_ts = datetime.fromisoformat(submitted_at)
    except (ValueError, TypeError):
        return None
    if job_ts.tzinfo is None:
        job_ts = job_ts.replace(tzinfo=timezone.utc)

    candidates: list[tuple[float, str]] = []
    for f in reports_dir.glob("tearsheet_*.html"):
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        diff = abs((mtime - job_ts).total_seconds())
        if diff <= 300:  # ±5 minutes
            candidates.append((diff, f.name))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def get_result_detail(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    reports_dir: Path | None = None,
) -> dict | None:
    """Return a rich dict for the detail view, or None if not found."""
    row = conn.execute(
        """
        SELECT r.*, j.strategy_name, j.config_json, j.submitted_at
        FROM results r
        JOIN jobs j ON r.job_id = j.id
        WHERE r.job_id = ?
        """,
        (job_id,),
    ).fetchone()
    if row is None:
        return None

    cfg = json.loads(row["config_json"])
    stats_raw = row["stats_json"]
    stats = json.loads(stats_raw) if stats_raw else {}
    positions_raw = row["positions_json"]
    positions = json.loads(positions_raw) if positions_raw else []
    fills_raw = row["fills_json"]
    fills = json.loads(fills_raw) if fills_raw else []

    tearsheet_id = row["run_id"] or row["job_id"]
    tearsheet_file = None
    if reports_dir is not None:
        tearsheet_file = _find_tearsheet(reports_dir, tearsheet_id)
        if tearsheet_file is None and not row["run_id"]:
            tearsheet_file = _find_tearsheet_by_job(reports_dir, row["submitted_at"])

    capital = float(cfg.get("capital", 0) or 0)
    pnl = row["pnl"]
    pnl_pct = (pnl / capital * 100) if capital and pnl is not None else None

    return {
        "job_id": row["job_id"],
        "run_id": row["run_id"],
        "display_id": row["run_id"] or row["job_id"],
        "strategy": row["strategy_name"],
        "exchange": cfg.get("exchange", ""),
        "symbol": cfg.get("symbol", ""),
        "interval": cfg.get("interval", ""),
        "sharpe": row["sharpe_ratio"],
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "trades": row["num_trades"],
        "sqn": row["sqn"],
        "duration": row["duration_seconds"],
        "funding_pnl": row["funding_pnl"],
        "date": row["submitted_at"][:10],
        "status": row["status"],
        "stats": stats,
        "positions": positions,
        "fills": fills,
        "has_tearsheet": tearsheet_file is not None,
        "tearsheet_file": tearsheet_file,
    }


# ── FastAPI app factory ───────────────────────────────────────────────


def create_app(
    db_path: str | Path = "sbt.db",
    reports_dir: str | Path = "reports",
) -> FastAPI:
    """Build the FastAPI application wired to a specific DB and reports dir."""
    from sbt.core.db import ResultStore

    ResultStore(db_path).close()

    app = FastAPI(title="SBT Results")
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    _db_path = str(db_path)
    _reports_dir = Path(reports_dir)

    def _conn() -> sqlite3.Connection:
        conn = sqlite3.connect(_db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @app.get("/", response_class=HTMLResponse)
    def list_results(request: Request):
        conn = _conn()
        try:
            rows = list_results_for_dashboard(conn)
        finally:
            conn.close()
        return templates.TemplateResponse(request, "list.html", {"results": rows})

    @app.get("/results/{job_id}", response_class=HTMLResponse)
    def detail(request: Request, job_id: str):
        conn = _conn()
        try:
            detail_data = get_result_detail(conn, job_id, reports_dir=_reports_dir)
        finally:
            conn.close()
        if detail_data is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return templates.TemplateResponse(
            request, "detail.html", {"result": detail_data}
        )

    @app.get("/reports/{path:path}")
    def serve_report(path: str):
        file = _reports_dir / path
        if not file.is_file():
            raise HTTPException(status_code=404, detail="Report not found")
        return FileResponse(str(file))

    return app
