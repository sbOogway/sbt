"""Shared argparse argument groups for CLI subcommands."""

from __future__ import annotations

import argparse


def add_backtest_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by backtest and optimize subcommands."""
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to TOML config file (default: config.toml)",
    )
    parser.add_argument(
        "--strategy",
        default="bitcoin_intraday_momentum",
        help="Strategy name (default: bitcoin_intraday_momentum)",
    )
    parser.add_argument("--exchange", help="Override exchange from config")
    parser.add_argument("--symbol", help="Override trading pair from config")
    parser.add_argument(
        "--symbols",
        action="append",
        metavar="SYMBOL[,SYMBOL...]",
        help=(
            "Multi-instrument symbols (comma-separated or repeatable); "
            "enables portfolio mode when more than one is given"
        ),
    )
    parser.add_argument("--interval", help="Override candle interval from config")
    parser.add_argument("--leverage", help="Override leverage from config")
    parser.add_argument("--start", help="Override backtest start date from config")
    parser.add_argument("--end", help="Override backtest end date from config")
    parser.add_argument(
        "--feather", help="Path to feather file (auto-detect if omitted)"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the tearsheet in a browser after the run",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        metavar="FRACTION",
        help="Holdout split: in-sample fraction (e.g. 0.7)",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        help="Bars loaded before trading start for indicator warm-up",
    )
    parser.add_argument(
        "--data-type",
        choices=["bar", "l2"],
        help="Data type: 'bar' (OHLCV) or 'l2' (OrderBookDelta + TradeTicks)",
    )
    parser.add_argument(
        "--l2-max-files",
        type=int,
        help="Max L2 parquet files to load (for fast testing)",
    )
    parser.add_argument(
        "--param",
        action="append",
        metavar="NAME=VALUE",
        help="Override a strategy parameter (repeatable)",
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the data download subcommand."""
    parser.add_argument(
        "--exchange", default="binance", help="Exchange ID (default: binance)"
    )
    parser.add_argument(
        "--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)"
    )
    parser.add_argument(
        "--interval", default="5m", help="Candle interval (default: 5m)"
    )
    parser.add_argument(
        "--start", default="2015-01-01", help="Start date (default: 2015-01-01)"
    )
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--output", default=None, help="Output feather path")
    parser.add_argument(
        "--page-limit",
        type=int,
        default=None,
        help="Rows per request (default: 1000 ohlcv / 500 funding)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refetch from --start even if the output file already exists",
    )
    parser.add_argument(
        "--type",
        default="ohlcv",
        choices=["ohlcv", "funding"],
        help="Data type to fetch (default: ohlcv)",
    )


def add_web_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the web dashboard subcommand."""
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
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the browser automatically",
    )
