"""Server CLI entry point.

Usage::

    uv run python3 -m sbt.server --workers 4 --port 5555 --worker-port 5556 --db sbt.db
"""

import argparse
from pathlib import Path
from .scheduler import Scheduler


def main():
    parser = argparse.ArgumentParser(description="SBT Backtesting Server Daemon")
    parser.add_argument("--workers", type=int, default=4, help="Fixed number of worker processes (default: 4)")
    parser.add_argument("--port", type=int, default=5555, help="Client request port (default: 5555)")
    parser.add_argument("--worker-port", type=int, default=5556, help="Worker communication port (default: 5556)")
    parser.add_argument("--db", default="sbt.db", help="SQLite database path (default: sbt.db)")
    parser.add_argument("--worktree-root", default=".worktrees", help="Worktree root directory (default: .worktrees)")
    parser.add_argument("--repo-root", default=".", help="Git repo root (default: .)")
    args = parser.parse_args()

    scheduler = Scheduler(
        num_workers=args.workers,
        client_endpoint=f"tcp://127.0.0.1:{args.port}",
        worker_endpoint=f"tcp://127.0.0.1:{args.worker_port}",
        db_path=args.db,
        worktree_root=args.worktree_root,
        repo_root=args.repo_root,
    )
    scheduler.start()


if __name__ == "__main__":
    main()
