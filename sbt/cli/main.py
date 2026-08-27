"""Top-level CLI entry point for sbt.

Usage::

    sbt backtest --config config.toml --strategy overnight_drift
    sbt data --exchange binance --symbol BTC/USDT --interval 5m --start 2024-01-01
    sbt web --port 8080
    sbt optimize --config config.toml --strategy overnight_drift --trials 50
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbt",
        description="sbt — sbOogway's backtest tool",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # backtest
    from .backtest import register as register_backtest
    from .data import register as register_data
    from .optimize import register as register_optimize
    from .web import register as register_web

    register_backtest(subparsers)
    register_data(subparsers)
    register_web(subparsers)
    register_optimize(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
