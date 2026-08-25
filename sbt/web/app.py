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
               r.sqn, r.duration_seconds, r.funding_pnl,
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
        results.append(
            {
                "job_id": row["job_id"],
                "strategy": row["strategy_name"],
                "symbol": cfg.get("symbol", ""),
                "sharpe": row["sharpe_ratio"],
                "pnl": row["pnl"],
                "trades": row["num_trades"],
                "sqn": row["sqn"],
                "duration": row["duration_seconds"],
                "date": row["submitted_at"][:10],
                "status": row["status"],
            }
        )
    return results


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

    has_tearsheet = False
    if reports_dir is not None:
        has_tearsheet = (reports_dir / f"tearsheet_{job_id}.html").exists()

    return {
        "job_id": row["job_id"],
        "strategy": row["strategy_name"],
        "exchange": cfg.get("exchange", ""),
        "symbol": cfg.get("symbol", ""),
        "interval": cfg.get("interval", ""),
        "sharpe": row["sharpe_ratio"],
        "pnl": row["pnl"],
        "trades": row["num_trades"],
        "sqn": row["sqn"],
        "duration": row["duration_seconds"],
        "funding_pnl": row["funding_pnl"],
        "date": row["submitted_at"][:10],
        "status": row["status"],
        "stats": stats,
        "positions": positions,
        "fills": fills,
        "has_tearsheet": has_tearsheet,
    }


# ── FastAPI app factory ───────────────────────────────────────────────


def create_app(
    db_path: str | Path = "sbt.db",
    reports_dir: str | Path = "reports",
) -> FastAPI:
    """Build the FastAPI application wired to a specific DB and reports dir."""
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
        return templates.TemplateResponse(
            request, "list.html", {"results": rows}
        )

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
