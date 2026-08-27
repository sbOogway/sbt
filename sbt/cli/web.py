"""`sbt web` subcommand."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``web`` subcommand to *subparsers*."""
    from .args import add_web_args

    p = subparsers.add_parser("web", help="Launch the results web dashboard")
    add_web_args(p)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Launch the web dashboard from parsed CLI args."""
    import uvicorn

    from ..web.app import create_app

    app = create_app(db_path=args.db, reports_dir=args.reports)
    print(f"Starting sbt results dashboard at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
