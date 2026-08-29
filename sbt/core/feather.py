"""Owner of the feather filename convention.

Data files follow ``{exchange}_{symbol}_{tag}_{YYYYMMDD}[_{YYYYMMDD}].feather``
where *tag* is the bar interval (``1h``, ``1d``, ...) for OHLCV data or
``funding`` for funding rates. The downloader names its output through
:func:`feather_path`; the runner discovers and ranks candidate files through
:func:`find_feather`. Writer and reader share this one home so the contract
cannot drift between them, and incremental resume heals stale range suffixes
through :func:`actual_range_name` instead of silently outgrowing the name.
"""

import glob
import re
from pathlib import Path

import pandas as pd

_RANGE_RE = re.compile(r"_(\d{8})_(\d{8})\.feather$")


def to_utc_ts(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalize a date string or Timestamp to a tz-aware UTC Timestamp."""
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
    return pd.Timestamp(value, tz="UTC")


def safe_symbol(symbol: str) -> str:
    """Strip the pair separator: ``BTC/USDC:USDC`` -> ``BTCUSDCUSDC``."""
    return symbol.replace("/", "")


def derive_tick_size(closes: pd.Series) -> float:
    """Best-effort price-tick from a close-price series.

    Strategy:
    1. Smallest non-zero abs diff between consecutive closes. Works for
       high-priced coins (BTC: 0.1, ETH: 0.01).
    2. If the smallest non-zero diff is below float resolution
       (sub-cent coins where daily changes are invisible to float64), fall
       back to a price-magnitude heuristic: ``ref_price * 1e-4`` capped
       to ``[1e-8, 1.0]``. This matches Bybit's typical micro-tick for
       altcoins.
    3. If the series is too short to compute a diff, return a default
       of 0.01 (the ETH-scale tick).
    """
    if len(closes) < 2:
        return 0.01
    diffs = closes.diff().dropna().abs()
    nonzero = diffs[diffs > 0]
    if len(nonzero) > 0:
        tick = float(nonzero.min())
        # If the tick is suspiciously large relative to the price
        # (e.g. a single volatile move between two bars), clamp to
        # 1% of the median price.
        median = float(closes.median())
        if median > 0 and tick > median * 0.01:
            return max(median * 1e-4, 1e-8)
        return tick
    # No non-zero diffs (constant prices or sub-float resolution).
    median = float(closes.median())
    if median <= 0:
        return 0.01
    # Price-magnitude heuristic: 1 bps of the median price, clamped.
    return min(max(median * 1e-4, 1e-8), 1.0)


def feather_path(
    exchange: str,
    symbol: str,
    tag: str,
    start,
    end,
) -> str:
    """Default feather path under ``data/`` following the naming convention.

    *start*/*end* are any tz-aware datetime or Timestamp; they become the
    ``YYYYMMDD`` range suffix.
    """
    return (
        f"data/{exchange}_{safe_symbol(symbol)}_{tag}_"
        f"{start:%Y%m%d}_{end:%Y%m%d}.feather"
    )


def parse_range(path: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Parse the ``_YYYYMMDD_YYYYMMDD`` suffix convention into (start, end)."""
    m = _RANGE_RE.search(str(path))
    if not m:
        return None
    s, e = m.group(1), m.group(2)
    return (
        pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:]}", tz="UTC"),
        pd.Timestamp(f"{e[:4]}-{e[4:6]}-{e[6:]} 23:59:59", tz="UTC"),
    )


def actual_range_name(
    path: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str | None:
    """Filename *path* would have if its range suffix matched [start, end].

    Returns None when the filename does not follow the convention (an
    explicit ``--output`` name), signalling callers to leave it untouched.
    """
    if not _RANGE_RE.search(Path(path).name):
        return None
    return _RANGE_RE.sub(f"_{start:%Y%m%d}_{end:%Y%m%d}.feather", Path(path).name)


def find_feather(
    exchange: str,
    symbol: str,
    tag: str,
    search_dirs: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> str | None:
    """Discover a feather data file by convention.

    Searches *search_dirs* (defaulting to ``["data", "."]``) for files
    matching ``{exchange}_{symbol}_{tag}_*.feather`` where *tag* is the bar
    interval or ``funding``. Unprefixed ``{symbol}_{tag}_*.feather`` files
    are considered only when they are the unique match in a directory
    (never preferred over prefixed ones). When several files match, the one
    best covering [start, end] wins (full coverage first, then max overlap,
    then newest range); the chosen file is printed.
    """
    if search_dirs is None:
        search_dirs = ["data", "."]

    raw_symbol = safe_symbol(symbol)
    req_start = to_utc_ts(start) if start else None
    req_end = to_utc_ts(end) if end else None

    for d in search_dirs:
        prefixed = sorted(
            glob.glob(f"{d}/{exchange.lower()}_{raw_symbol}_{tag}_*.feather")
        )
        bare = [
            f
            for f in sorted(glob.glob(f"{d}/{raw_symbol}_{tag}_*.feather"))
            if f not in prefixed
        ]
        candidates = prefixed or (bare if len(bare) == 1 else [])
        if not candidates:
            continue

        def rank(path: str):
            rng = parse_range(path)
            if rng is None:
                return (-1, pd.Timedelta(0), pd.Timestamp(0))
            fs, fe = rng
            if req_start is None:
                return (0, pd.Timedelta(0), fe)
            lo = req_start
            hi = req_end if req_end is not None else fe
            covers = fs <= lo and fe >= hi
            overlap = min(fe, hi) - max(fs, lo)
            return (1 if covers else 0, overlap, fe)

        choice = max(candidates, key=rank)
        if len(candidates) > 1:
            print(f"find_feather: {len(candidates)} matches; chose {choice}")
        return choice
    return None
