"""`sbt data` subcommand."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``data`` subcommand to *subparsers*."""
    from .args import add_data_args

    p = subparsers.add_parser("data", help="Download market data via ccxt")
    add_data_args(p)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Download market data from parsed CLI args."""
    from ..core.feather import actual_range_name, feather_path
    from ..data import fetch_funding_rates, fetch_ohlcv, _resume_start_ms

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end_dt = (
        datetime.fromisoformat(args.end).replace(tzinfo=UTC)
        if args.end
        else datetime.now(UTC)
    )

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    data_tag = args.type if args.type != "ohlcv" else args.interval
    output = Path(
        args.output
        or feather_path(args.exchange, args.symbol, data_tag, start_dt, end_dt)
    )

    prev: pd.DataFrame | None = None
    if not args.no_resume:
        resume_ms, prev = _resume_start_ms(output, args.type)
        if resume_ms is not None:
            print(
                f"Resuming {output} from {pd.Timestamp(resume_ms, unit='ms', tz='UTC')}"
            )
            start_ms = max(start_ms, resume_ms)

    if args.type == "funding":
        rows = fetch_funding_rates(
            args.exchange, args.symbol, start_ms, end_ms, page_limit=args.page_limit
        )
    else:
        rows = fetch_ohlcv(
            args.exchange,
            args.symbol,
            args.interval,
            start_ms,
            end_ms,
            page_limit=args.page_limit,
        )
    if not rows and prev is None:
        print("No data fetched.")
        return

    df = pd.DataFrame(rows)
    if prev is not None and len(df):
        df = pd.concat([prev, df], ignore_index=True)
    elif prev is not None:
        df = prev
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_feather(output)
    healed = actual_range_name(output, df["timestamp"].min(), df["timestamp"].max())
    if healed is not None and healed != output.name:
        output = output.rename(output.with_name(healed))
        print(f"Renamed to {output.name} to match the actual data range")
    print(f"Saved {len(df)} rows to {output}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
