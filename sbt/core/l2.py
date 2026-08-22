"""Layer 2 (L2) Parquet data loader for Nautilus Trader."""

import glob
import logging
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nautilus_trader.model.data import BookOrder, OrderBookDelta, TradeTick
from nautilus_trader.model.enums import AggressorSide, BookAction, BookType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, TradeId
from nautilus_trader.model.instruments import CryptoPerpetual, Instrument
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.serialization.arrow.serializer import ArrowSerializer

logger = logging.getLogger("L2Data")

_ACTION_MAP = {
    1: BookAction.ADD,
    2: BookAction.UPDATE,
    3: BookAction.DELETE,
    4: BookAction.CLEAR,
}

_SIDE_MAP = {
    1: OrderSide.BUY,
    2: OrderSide.SELL,
}

_AGGRESSOR_SIDE_MAP = {
    0: AggressorSide.NO_AGGRESSOR,
    1: AggressorSide.BUYER,
    2: AggressorSide.SELLER,
}


def list_l2_instruments(catalog_dir: str = "data") -> list[str]:
    """List available instrument IDs in the L2 data catalog."""
    inst_dir = Path(catalog_dir) / "instruments"
    if not inst_dir.exists():
        return []
    return [p.name for p in inst_dir.iterdir() if p.is_dir()]


def load_l2_instrument(instrument_id: str, catalog_dir: str = "data") -> Instrument:
    """Load Instrument definition from the L2 Parquet catalog."""
    inst_dir = Path(catalog_dir) / "instruments" / instrument_id
    if not inst_dir.exists():
        raise FileNotFoundError(f"Instrument directory not found: {inst_dir}")

    files = sorted(glob.glob(str(inst_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No instrument parquet files found in {inst_dir}")

    table = pq.read_table(files[0])
    insts = ArrowSerializer.deserialize(CryptoPerpetual, table)
    if not insts:
        raise RuntimeError(f"Failed to deserialize instrument from {files[0]}")
    return insts[0]


_FILE_TS_PATTERN = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})-(\d+)Z$"
)


def _parse_file_timestamp(stem_part: str) -> pd.Timestamp | None:
    """Parse a filename timestamp like ``2026-07-28T14-53-47-373180579Z``."""
    m = _FILE_TS_PATTERN.match(stem_part)
    if not m:
        return None
    y, mo, d, h, mi, s, frac = m.groups()
    iso = f"{y}-{mo}-{d}T{h}:{mi}:{s}.{frac}"
    try:
        return pd.Timestamp(iso, tz="UTC")
    except ValueError:
        return None


def _filter_files_by_time(
    files: list[str],
    start: str | None = None,
    end: str | None = None,
) -> list[str]:
    """Filter files whose ISO timestamps overlap with [start, end]."""
    if not start and not end:
        return files

    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    filtered = []
    for f in files:
        parts = Path(f).stem.split("_")
        if len(parts) == 2:
            file_start = _parse_file_timestamp(parts[0])
            file_end = _parse_file_timestamp(parts[1])
            # Only apply overlap logic when both timestamps parsed cleanly;
            # otherwise fall open and let row-level filtering decide.
            if file_start is not None and file_end is not None:
                if start_ts and file_end < start_ts:
                    continue
                if end_ts and file_start > end_ts:
                    continue
        filtered.append(f)
    return filtered


def load_order_book_deltas(
    instrument: Instrument,
    catalog_dir: str = "data",
    start: str | None = None,
    end: str | None = None,
    max_files: int | None = None,
) -> list[OrderBookDelta]:
    """Load and parse OrderBookDelta records from Parquet files."""
    inst_str = instrument.id.value
    delta_dir = Path(catalog_dir) / "order_book_deltas" / inst_str
    if not delta_dir.exists():
        logger.warning("No order book deltas found for %s at %s", inst_str, delta_dir)
        return []

    files = sorted(glob.glob(str(delta_dir / "*.parquet")))
    files = _filter_files_by_time(files, start, end)
    if max_files:
        files = files[:max_files]

    if not files:
        return []

    logger.info("Loading %d order book delta files for %s...", len(files), inst_str)

    inst_id = instrument.id
    p_prec = instrument.price_precision
    s_prec = instrument.size_precision

    # Apply time filter if needed
    start_nanos = pd.Timestamp(start, tz="UTC").value if start else None
    end_nanos = pd.Timestamp(end, tz="UTC").value if end else None

    deltas: list[OrderBookDelta] = []
    for path in files:
        table = pq.read_table(path)
        actions = table["action"].to_numpy()
        sides = table["side"].to_numpy()
        raw_p = np.frombuffer(b"".join(table["price"].to_pylist()), dtype="<i8")
        raw_s = np.frombuffer(b"".join(table["size"].to_pylist()), dtype="<i8")
        order_ids = table["order_id"].to_numpy()
        flags = table["flags"].to_numpy()
        seqs = table["sequence"].to_numpy()
        ts_events = table["ts_event"].to_numpy()
        ts_inits = table["ts_init"].to_numpy()

        prices = raw_p / 1e9
        sizes = raw_s / 1e9

        for i in range(len(table)):
            ts = ts_events[i]
            if start_nanos and ts < start_nanos:
                continue
            if end_nanos and ts > end_nanos:
                continue

            p = Price(prices[i], precision=p_prec)
            s = Quantity(sizes[i], precision=s_prec)
            order = BookOrder(_SIDE_MAP[sides[i]], p, s, int(order_ids[i]))
            deltas.append(OrderBookDelta(
                inst_id,
                _ACTION_MAP[actions[i]],
                order,
                int(flags[i]),
                int(seqs[i]),
                int(ts),
                int(ts_inits[i]),
            ))

        del table, actions, sides, raw_p, raw_s, order_ids, flags, seqs
        del ts_events, ts_inits, prices, sizes

    logger.info("Loaded %d OrderBookDelta objects for %s", len(deltas), inst_str)
    return deltas


def load_trade_ticks(
    instrument: Instrument,
    catalog_dir: str = "data",
    start: str | None = None,
    end: str | None = None,
    max_files: int | None = None,
) -> list[TradeTick]:
    """Load and parse TradeTick records from Parquet files."""
    inst_str = instrument.id.value
    trade_dir = Path(catalog_dir) / "trades" / inst_str
    if not trade_dir.exists():
        logger.warning("No trades found for %s at %s", inst_str, trade_dir)
        return []

    files = sorted(glob.glob(str(trade_dir / "*.parquet")))
    files = _filter_files_by_time(files, start, end)
    if max_files:
        files = files[:max_files]

    if not files:
        return []

    logger.info("Loading %d trade tick files for %s...", len(files), inst_str)

    inst_id = instrument.id
    p_prec = instrument.price_precision
    s_prec = instrument.size_precision

    start_nanos = pd.Timestamp(start, tz="UTC").value if start else None
    end_nanos = pd.Timestamp(end, tz="UTC").value if end else None

    trades: list[TradeTick] = []
    for path in files:
        table = pq.read_table(path)
        raw_p = np.frombuffer(b"".join(table["price"].to_pylist()), dtype="<i8")
        raw_s = np.frombuffer(b"".join(table["size"].to_pylist()), dtype="<i8")
        aggressor_sides = table["aggressor_side"].to_numpy()
        trade_ids = table["trade_id"].to_pylist()
        ts_events = table["ts_event"].to_numpy()
        ts_inits = table["ts_init"].to_numpy()

        prices = raw_p / 1e9
        sizes = raw_s / 1e9

        for i in range(len(table)):
            ts = ts_events[i]
            if start_nanos and ts < start_nanos:
                continue
            if end_nanos and ts > end_nanos:
                continue

            p = Price(prices[i], precision=p_prec)
            s = Quantity(sizes[i], precision=s_prec)
            agg_side = _AGGRESSOR_SIDE_MAP.get(aggressor_sides[i], AggressorSide.NO_AGGRESSOR)
            trades.append(TradeTick(
                instrument_id=inst_id,
                price=p,
                size=s,
                aggressor_side=agg_side,
                trade_id=TradeId(str(trade_ids[i])),
                ts_event=int(ts),
                ts_init=int(ts_inits[i]),
            ))

        del table, raw_p, raw_s, aggressor_sides, trade_ids
        del ts_events, ts_inits, prices, sizes

    logger.info("Loaded %d TradeTick objects for %s", len(trades), inst_str)
    return trades
