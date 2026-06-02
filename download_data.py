import argparse
import time
from datetime import datetime, timezone

import ccxt
from nautilus_trader.core.uuid import UUID4
import pandas as pd
from tqdm import tqdm


def fetch_ohlcv(exchange_id: str, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    all_bars: list[dict] = []
    since = start_ms
    retries = 3

    pbar = tqdm(desc=f"Fetching {symbol} {interval} from {exchange_id}")
    while since < end_ms:
        for attempt in range(retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, interval, since=since, limit=1000)
                break
            except ccxt.NetworkError as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    tqdm.write(f"Network error, retrying in {wait}s... ({e})")
                    time.sleep(wait)
                else:
                    raise
        if not ohlcv:
            break
        parsed = [
            {
                "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in ohlcv
        ]
        all_bars.extend(parsed)
        since = ohlcv[-1][0] + 1
        pbar.update(len(parsed))
    pbar.close()
    return all_bars


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV candles via ccxt")
    parser.add_argument("--exchange", default="binance", help="Exchange ID (default: binance)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--interval", default="5m", help="Candle interval (default: 5m)")
    parser.add_argument("--start", default="2015-01-01", help="Start date (default: 2015-01-01)")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--output", default=None, help="Output feather path")
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc)
    )

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    bars = fetch_ohlcv(args.exchange, args.symbol, args.interval, start_ms, end_ms)
    if not bars:
        print("No data fetched.")
        return

    df = pd.DataFrame(bars)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    safe_symbol = args.symbol.replace("/", "")
    session_id = UUID4()
    output = (
        args.output
        or f"data/{args.exchange}_{safe_symbol}_{args.interval}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}_{session_id}.feather"
    )
    df.to_feather(output)
    print(f"Saved {len(df)} rows to {output}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
