"""Bulk-download the Bybit USDT-linear universe via the single-symbol ``sbt data``.

Reads the pinned universe file (one ccxt ``symbol`` per line, ``#`` comments),
and for each symbol drives ``python -m sbt data --exchange bybit --symbol <s>
--interval 1d --start <start> --type ohlcv``. Each ``sbt data`` call is
incremental/resume internally (extends the existing ``data/bybit_<id>:USDT_1d_*``
feather from its newest timestamp), so re-runs cheaply top every symbol up to
today.

Usage:

    uv run python scripts/universe_pull.py --dry-run          # plan only
    uv run python scripts/universe_pull.py                     # fetch all
    uv run python scripts/universe_pull.py --skip-existing     # net-new only

    --universe file    default universe/bybit_usdt_linears.txt
    --start 2023-01-01 common window start (symbols' fetch start)
    --skip-existing    skip symbols whose on-disk _1d_ feather already covers --start
    --state file       persistent per-symbol status JSON (resume across reruns)
    --jobs N           parallel subprocesses (default 1; opt-in - no shared rate limit)
    --dry-run          print the plan, fetch nothing
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from sbt.core.feather import parse_range, safe_symbol


def _universe(file: Path) -> list[str]:
    syms = []
    for line in file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            syms.append(line)
    return syms


def _feather_path(exchange: str, symbol: str, tag: str, data_dir: Path) -> Path | None:
    """Locate the on-disk feather by convention (best match by start coverage)."""
    token = safe_symbol(symbol)
    best: Path | None = None
    best_start: date | None = None
    for p in data_dir.glob(f"{exchange}_{token}_{tag}_*.feather"):
        rng = parse_range(p.name)
        if rng is None:
            continue
        if best is None or rng[0].date() < best_start:
            best, best_start = p, rng[0].date()
    return best


def _covered(feather: Path | None, start: date) -> bool:
    if feather is None:
        return False
    rng = parse_range(feather.name)
    return rng is not None and rng[0].date() <= start


def _run_one(exchange, symbol, interval, start, data_dir, state):
    token = safe_symbol(symbol)
    cmd = [
        sys.executable, "-m", "sbt", "data",
        "--exchange", exchange, "--symbol", symbol,
        "--interval", interval, "--start", start, "--type", "ohlcv",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        if proc.returncode == 0 and ("Saved" in out or "No data fetched" in out):
            if "No data fetched" in out:
                state[token] = {"status": "EMPTY", "ts": datetime.now(UTC).isoformat()}
                return token, "EMPTY"
            state[token] = {"status": "DONE", "ts": datetime.now(UTC).isoformat()}
            return token, "DONE"
        state[token] = {"status": "ERROR", "ts": datetime.now(UTC).isoformat(),
                        "detail": out[-300:]}
        return token, f"ERROR -> {out.strip().splitlines()[-1][:120]}"
    except Exception as e:  # noqa: BLE001
        state[token] = {"status": "ERROR", "ts": datetime.now(UTC).isoformat(),
                        "detail": str(e)}
        return token, f"ERROR -> {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="universe/bybit_usdt_linears.txt")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--exchange", default="bybit")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--state", default="reports/universe_pull_state.json")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    state_file = Path(args.state)
    state_file.parent.mkdir(exist_ok=True, parents=True)
    state: dict = {}
    if state_file.exists():
        state = json.loads(state_file.read_text())

    syms = _universe(Path(args.universe))
    todo = []
    skipped = 0
    for s in syms:
        token = safe_symbol(s)
        if state.get(token, {}).get("status") == "DONE":
            continue  # resume: already fetched in a prior batch
        if args.skip_existing and _covered(_feather_path(args.exchange, s, args.interval, data_dir), date.fromisoformat(args.start)):
            skipped += 1
            continue
        todo.append(s)

    print(f"universe={len(syms)}  todo={len(todo)}  skipped_existing={skipped}  resumed_done={len(syms)-len(todo)-skipped}")
    if args.dry_run:
        for s in todo[:50]:
            print(f"  {s}")
        if len(todo) > 50:
            print(f"  ... and {len(todo)-50} more")
        print("dry-run: would fetch", len(todo), "symbols")
        return

    t0 = time.monotonic()
    counts = {"DONE": 0, "EMPTY": 0, "ERROR": 0}
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_run_one, args.exchange, s, args.interval, args.start, data_dir, state): s for s in todo}
            for fut in as_completed(futs):
                tok, status = fut.result()
                first = status.split(" ->")[0].strip()
                counts[first] = counts.get(first, 0) + 1
                print(f"{tok}: {status[:140]}")
    else:
        for s in todo:
            tok, status = _run_one(args.exchange, s, args.interval, args.start, data_dir, state)
            first = status.split(" ->")[0].strip()
            counts[first] = counts.get(first, 0) + 1
            print(f"{tok}: {status[:140]}")

    state_file.write_text(json.dumps(state, indent=2))
    dt = time.monotonic() - t0
    print(f"\nsummary: {counts}  ({len(todo)} symbols)  runtime={dt:.1f}s ({dt/60:.1f}min)")


if __name__ == "__main__":
    main()
