"""Entry point: ``python -m sbt.web``."""

from __future__ import annotations

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="sbt results web dashboard")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    parser.add_argument(
        "--db", default="sbt.db", help="Path to SQLite database (default: sbt.db)"
    )
    parser.add_argument(
        "--reports",
        default="reports",
        help="Path to reports directory (default: reports)",
    )
    args = parser.parse_args()

    from sbt.web.app import create_app

    app = create_app(db_path=args.db, reports_dir=args.reports)
    print(f"Starting sbt results dashboard at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
