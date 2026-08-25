"""Data downloader: ccxt -> feather.

Unified pagination with retries and adaptive page sizes for both OHLCV and
funding rates, plus incremental resume: re-running with an existing output
file extends it from its newest timestamp instead of refetching everything.
"""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import ccxt
import pandas as pd
from tqdm import tqdm

from .core.feather import actual_range_name, feather_path

_DEFAULT_PAGE_LIMITS = {"ohlcv": 1000, "funding": 500}


def _make_exchange(exchange_id: str):
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def _fetch_page_with_retry(fetch, *, retries=3, what="page"):
    last_err = None
    for attempt in range(retries):
        try:
            return fetch()
        except ccxt.NetworkError as e:
            last_err = e
            if attempt < retries - 1:
                wait = 2**attempt
                tqdm.write(f"Network error, retrying in {wait}s... ({e})")
                time.sleep(wait)
    raise last_err


def _paginate(fetch_page, start_ms, end_ms, *, label, initial_limit):
    """Yield raw pages walking [start_ms, end_ms); adapts page size down
    when the exchange rejects the limit."""
    limit = initial_limit
    since = start_ms
    pbar = tqdm(desc=label)
    while since < end_ms:
        try:
            rows = _fetch_page_with_retry(lambda: fetch_page(since, limit))
        except ccxt.ExchangeError as e:
            # Some exchanges cap results per call; shrink until accepted.
            if limit > 50 and ("limit" in str(e).lower() or "size" in str(e).lower()):
                limit = max(50, limit // 2)
                tqdm.write(f"{type(e).__name__}: lowering page size to {limit}")
                continue
            raise
        if not rows:
            break
        yield rows
        pbar.update(len(rows))
        last_ms = rows[-1][0] if isinstance(rows[0], list) else rows[-1]["timestamp"]
        since = last_ms + 1
    pbar.close()


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    page_limit: int | None = None,
):
    exchange = _make_exchange(exchange_id)
    limit = page_limit or _DEFAULT_PAGE_LIMITS["ohlcv"]

    def page(since, limit):
        return exchange.fetch_ohlcv(symbol, interval, since=since, limit=limit)

    def parse(k):
        return {
            "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        }

    out = []
    for raw in _paginate(
        page,
        start_ms,
        end_ms,
        label=f"Fetching {symbol} {interval} from {exchange_id}",
        initial_limit=limit,
    ):
        out.extend(parse(k) for k in raw)
    return out


def fetch_funding_rates(
    exchange_id: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    page_limit: int | None = None,
):
    exchange = _make_exchange(exchange_id)
    limit = page_limit or _DEFAULT_PAGE_LIMITS["funding"]

    def page(since, limit):
        return exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)

    def parse(r):
        return {
            "timestamp": pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"),
            "funding_rate": float(r["fundingRate"]),
        }

    out = []
    for raw in _paginate(
        page,
        start_ms,
        end_ms,
        label=f"Fetching {symbol} funding rates from {exchange_id}",
        initial_limit=limit,
    ):
        out.extend(parse(r) for r in raw)
    return out


def _resume_start_ms(path: Path, data_type: str) -> tuple[int, pd.DataFrame | None]:
    """Return (resume_since_ms, previous_frame) when *path* can be extended."""
    if not path.exists():
        return None, None
    try:
        prev = pd.read_feather(path)
        if prev.empty or "timestamp" not in prev.columns:
            return None, None
        max_ts = pd.to_datetime(prev["timestamp"], utc=True).max()
        return int(max_ts.value // 1_000_000) + 1, prev
    except Exception as e:
        tqdm.write(f"Could not read {path} for resume ({e}); refetching from scratch")
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download market data via ccxt")
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
