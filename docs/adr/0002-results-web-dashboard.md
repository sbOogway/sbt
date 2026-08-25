# ADR-0002: Server-rendered web dashboard for results

Status: Accepted

We added a lightweight web UI (`sbt.web`) to browse backtest results
alongside their tearsheets. The project was CLI-only; a web view
simplifies result exploration and sharing.

Server-rendered HTML (FastAPI + Jinja2) over a JS SPA — no build step,
no client-side state management. The app reads directly from `sbt.db`
(SQLite, WAL mode) so it works standalone without the scheduler running.
Existing tearsheet HTML files are served as static assets and embedded
via iframe, avoiding any re-rendering. A light theme distinguishes the
dashboard from the dark-themed comparison/Optuna reports.

Considered alternatives: Streamlit (too opinionated, limited layout
control), Flask (synchronous, no async advantage), static SPA (build
complexity unjustified for two views). The FastAPI path was chosen for
its lightweight async support (useful if ZMQ live-updates are added
later) and native OpenAPI docs.
