"""`sbt data` subcommand.

The orchestration lives in :func:`sbt.data.run`; this module is just the
argparse subcommand wiring so ``sbt data --foo`` is dispatched correctly.
"""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``data`` subcommand to *subparsers*."""
    from .args import add_data_args

    p = subparsers.add_parser("data", help="Download market data via ccxt")
    add_data_args(p)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Download market data from parsed CLI args."""
    from ..data import run as _run

    _run(args)
