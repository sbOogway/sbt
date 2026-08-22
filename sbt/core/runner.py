"""Reusable backtest runner extracted from the monolithic __main__ block.

Usage (standalone)::

    from sbt.core.config import RunConfig
    from sbt.core.runner import BacktestRunner

    cfg = RunConfig.from_toml("config.toml", "overnight_drift")
    runner = BacktestRunner(cfg)
    result = runner.run()
    # runner.engine / runner.venue available for report generation
"""

import glob
from decimal import Decimal
import dataclasses
import re
import time
from pathlib import Path

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import USDC, USDT, Currency
from nautilus_trader.model.data import Bar, BarType, FundingRateUpdate
from nautilus_trader.model.enums import AccountType, BookType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Price, Quantity

from ..stats import AnnualizedReturn, CalmarRatio, RunConfigStatistic, system_quality_number
from ..utils import get_strategy_class, make_perpetual, parse_interval
from ..plugins import RunnerPlugin, Window, get_runner_plugin_class
from .config import RunConfig
from .job import BacktestResult, JobStatus
from .l2 import list_l2_instruments, load_l2_instrument, load_order_book_deltas, load_trade_ticks

_CURRENCY_MAP = {
    "USDT": USDT,
    "USDC": USDC,
}

# Positions/fills reports above this row count are spilled to parquet under
# reports/artifacts/{job_id}/ instead of being embedded in the result (and
# then in every DB row / ZMQ payload).
INLINE_ROW_BUDGET = 200


def _jsonable_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of JSON-serializable dicts.

    Results travel over ZMQ and into SQLite as JSON; engine report frames
    carry pandas.Timestamp / numpy scalars that json.dumps chokes on.
    """
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[col]):
            d[col] = d[col].apply(
                lambda v: v.isoformat() if pd.notna(v) else None
            )
    out = []
    for rec in d.to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp) or isinstance(
                v, __import__("datetime").datetime
            ):
                v = v.isoformat()
            elif hasattr(v, "item"):  # numpy scalars
                try:
                    v = v.item()
                except (ValueError, AttributeError):
                    pass
            elif isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            clean[k] = v
        out.append(clean)
    return out


def _spill_artifacts(
    job_id: str, positions_df: pd.DataFrame, fills_df: pd.DataFrame
) -> dict:
    """Package positions/fills for BacktestResult, spilling big ones to disk."""
    out: dict = {
        "positions": [],
        "fills": [],
        "positions_path": None,
        "fills_path": None,
        "positions_count": int(len(positions_df)),
        "fills_count": int(len(fills_df)),
    }
    needs_spill = max(len(positions_df), len(fills_df)) > INLINE_ROW_BUDGET
    if not needs_spill:
        out["positions"] = (
            _jsonable_records(positions_df) if len(positions_df) else []
        )
        out["fills"] = _jsonable_records(fills_df) if len(fills_df) else []
        return out

    artifact_dir = Path("reports") / "artifacts" / job_id
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if len(positions_df):
            p = artifact_dir / "positions.parquet"
            positions_df.to_parquet(p)
            out["positions_path"] = str(p.resolve())
        if len(fills_df):
            f = artifact_dir / "fills.parquet"
            fills_df.to_parquet(f)
            out["fills_path"] = str(f.resolve())
        print(
            f"Artifacts spilled to {artifact_dir} "
            f"(positions={out['positions_count']}, fills={out['fills_count']})"
        )
    except Exception as e:
        # Parquet write failed — keep inline rather than lose the data.
        print(f"WARNING: artifact spill failed ({e}); embedding rows inline")
        out["positions_path"] = None
        out["fills_path"] = None
        out["positions"] = _jsonable_records(positions_df) if len(positions_df) else []
        out["fills"] = _jsonable_records(fills_df) if len(fills_df) else []
    return out


# ------------------------------------------------------------------
# Data helpers (moved from __main__)
# ------------------------------------------------------------------


def _to_utc_ts(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalize a date string or Timestamp to a tz-aware UTC Timestamp."""
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
    return pd.Timestamp(value, tz="UTC")


def _fmt_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:+,.2f}"
    return f"{value:,}"


def load_bars(
    df: pd.DataFrame, bar_type: BarType, instrument=None
) -> list[Bar]:
    """Convert an OHLCV DataFrame into a list of Nautilus Bar objects.

    Iterates plain numpy arrays rather than DataFrame rows (itertuples
    boxes each row into a namedtuple); nautilus' ``BarDataWrangler`` was
    tried here but raises "buffer source array is read-only" on every
    input shape under nautilus 1.230.0.
    """
    if instrument is not None:
        price_precision = instrument.price_precision
        size_precision = instrument.size_precision
    else:
        price_precision, size_precision = 1, 3

    # Normalize to int64 nanos regardless of source resolution (feathers
    # may carry datetime64[ms]/[us]); naive stamps are assumed UTC.
    stamp = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("ns")
    ts_nanos = stamp.astype("int64").to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    volumes = df["volume"].to_numpy()

    return [
        Bar(
            bar_type=bar_type,
            open=Price(opens[i], precision=price_precision),
            high=Price(highs[i], precision=price_precision),
            low=Price(lows[i], precision=price_precision),
            close=Price(closes[i], precision=price_precision),
            volume=Quantity(volumes[i], precision=size_precision),
            ts_event=ts_nanos[i],
            ts_init=ts_nanos[i],
        )
        for i in range(len(df))
    ]


def load_funding_rates(df: pd.DataFrame, instrument_id) -> list[FundingRateUpdate]:
    """Convert a funding-rate DataFrame into Nautilus FundingRateUpdate objects."""
    updates = []
    for row in df.itertuples(index=False):
        ts_nanos = dt_to_unix_nanos(row.timestamp)
        updates.append(
            FundingRateUpdate(
                instrument_id=instrument_id,
                rate=float(row.funding_rate),
                ts_event=ts_nanos,
                ts_init=ts_nanos,
            )
        )
    return updates


def _feather_range(path: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Parse the _YYYYMMDD_YYYYMMDD suffix convention into (start, end)."""
    m = re.search(r"_(\d{8})_(\d{8})\.feather$", path)
    if not m:
        return None
    s, e = m.group(1), m.group(2)
    return (
        pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:]}", tz="UTC"),
        pd.Timestamp(f"{e[:4]}-{e[4:6]}-{e[6:]} 23:59:59", tz="UTC"),
    )


def find_feather(
    exchange: str,
    symbol: str,
    interval: str,
    search_dirs: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> str | None:
    """Discover a feather data file by convention.

    Searches *search_dirs* (defaulting to ``["data", "."]``) for files
    matching ``{exchange}_{symbol}_{interval}_*.feather``. Unprefixed
    ``{symbol}_{interval}_*.feather`` files are considered only when they
    are the unique match in a directory (never preferred over prefixed
    ones). When several files match, the one best covering [start, end]
    wins (full coverage first, then max overlap, then newest range); the
    chosen file is printed.
    """
    if search_dirs is None:
        search_dirs = ["data", "."]

    raw_symbol = symbol.replace("/", "")
    req_start = _to_utc_ts(start) if start else None
    req_end = _to_utc_ts(end) if end else None

    for d in search_dirs:
        prefixed = sorted(
            glob.glob(f"{d}/{exchange.lower()}_{raw_symbol}_{interval}_*.feather")
        )
        bare = [
            f
            for f in sorted(glob.glob(f"{d}/{raw_symbol}_{interval}_*.feather"))
            if f not in prefixed
        ]
        candidates = prefixed or (bare if len(bare) == 1 else [])
        if not candidates:
            continue

        def rank(path: str):
            rng = _feather_range(path)
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
            print(
                f"find_feather: {len(candidates)} matches; chose {choice}"
            )
        return choice
    return None


# ------------------------------------------------------------------
# Runner-plugin resolution
# ------------------------------------------------------------------


def resolve_runner_plugin(cfg: RunConfig) -> RunnerPlugin | None:
    """Map flat RunConfig fields onto registered runner plugins."""
    if cfg.train_val_split is not None:
        return get_runner_plugin_class("train_val_split")(cfg.train_val_split)
    return None


# ------------------------------------------------------------------
# BacktestRunner
# ------------------------------------------------------------------


class BacktestRunner:
    """Configures and executes a single backtest.

    After :meth:`run` completes, the Nautilus ``engine`` and ``venue``
    are available as instance attributes for downstream report
    generation.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.engine: BacktestEngine | None = None
        self.venue: Venue | None = None
        self.strategy = None
        # Per-window engines when a runner plugin split the job
        # (e.g. {"in_sample": engine, "out_of_sample": engine}).
        self.window_engines: dict[str, BacktestEngine] = {}

    def run(self, job_id: str = "standalone") -> BacktestResult:
        """Execute the backtest and return a structured result.

        A configured runner plugin (flat ``train_val_split`` field today)
        expands the job into windows; each window runs through the normal
        execution path and the plugin merges them into one result.
        """
        plugin = resolve_runner_plugin(self.config)
        if plugin is not None:
            return self._run_windows(job_id, plugin)
        return self._run_window(job_id, start=self.config.start, end=self.config.end)

    # ------------------------------------------------------------------
    # Windowed execution (runner plugins)
    # ------------------------------------------------------------------

    def _run_windows(self, job_id: str, plugin: RunnerPlugin) -> BacktestResult:
        cfg = self.config

        df: pd.DataFrame | None = None
        if cfg.data_type != "l2":
            feather_path = cfg.feather_path or find_feather(
                cfg.exchange,
                cfg.symbol,
                cfg.interval,
                search_dirs=[cfg.data_dir, "."],
                start=cfg.start,
                end=cfg.end,
            )
            if not feather_path:
                return BacktestResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error=f"No feather data found for {cfg.symbol} ({cfg.interval})",
                )
            print(f"Loading data from {feather_path}...")
            try:
                df = pd.read_feather(feather_path)[
                    ["timestamp", "open", "high", "low", "close", "volume"]
                ]
            except FileNotFoundError:
                return BacktestResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error=f"Feather file '{feather_path}' not found.",
                )

            df = df[df["timestamp"] >= _to_utc_ts(cfg.start)].reset_index(drop=True)
            if cfg.end:
                df = df[df["timestamp"] <= _to_utc_ts(cfg.end)].reset_index(drop=True)

        try:
            windows: dict[str, Window] = plugin.expand(cfg, df)
        except ValueError as e:
            return BacktestResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                error=str(e),
            )

        self.window_engines = {}
        results: dict[str, BacktestResult] = {}
        for key, win in windows.items():
            print(f"\n--- {win.label} window: {win.start} -> {win.end} ---")
            res = self._run_window(
                f"{job_id}:{key}",
                start=win.start,
                end=win.end,
                df=win.df,
                pre_sliced=True,
            )
            results[key] = res
            if res.status != JobStatus.DONE:
                return res
            self.window_engines[key] = self.engine

        combined = plugin.combine(job_id, results, windows)
        plugin.summarize(results)
        return combined

    # ------------------------------------------------------------------
    # Single-window execution
    # ------------------------------------------------------------------

    def _run_window(
        self,
        job_id: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None,
        df: pd.DataFrame | None = None,
        pre_sliced: bool = False,
    ) -> BacktestResult:
        """Execute one backtest over the [start, end] window.

        *df* may carry a preloaded OHLCV frame (split mode); with
        ``pre_sliced=True`` the frame is trusted to include any warm-up
        bars ahead of *start* (trading gates via strategy ``active_from``),
        so only the upper bound is applied here.
        """
        t0 = time.monotonic()
        cfg = self.config

        # --------------------------------------------------------------
        # Layer 2 Execution Mode
        # --------------------------------------------------------------
        if cfg.data_type == "l2":
            avail = list_l2_instruments(cfg.data_dir)
            inst_id_str = cfg.symbol
            if inst_id_str not in avail:
                matched = [
                    a
                    for a in avail
                    if cfg.symbol.replace("/", "").replace(":", "").lower()
                    in a.replace("-", "").replace(".", "").lower()
                ]
                if matched:
                    inst_id_str = matched[0]
                elif avail:
                    inst_id_str = avail[0]
                else:
                    return BacktestResult(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        error=f"No L2 instruments found in catalog '{cfg.data_dir}'",
                    )

            print(f"Loading L2 instrument '{inst_id_str}' from {cfg.data_dir}...")
            try:
                instrument = load_l2_instrument(inst_id_str, catalog_dir=cfg.data_dir)
            except Exception as e:
                return BacktestResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error=f"Failed loading L2 instrument {inst_id_str}: {e}",
                )

            venue = instrument.id.venue
            self.venue = venue
            engine = BacktestEngine(config=BacktestEngineConfig())
            self.engine = engine

            settle_currency = instrument.settlement_currency or _CURRENCY_MAP.get(
                cfg.settle_currency, Currency(cfg.settle_currency, 2, 0, cfg.settle_currency, 0)
            )

            engine.add_venue(
                venue=venue,
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN,
                base_currency=settle_currency,
                starting_balances=[Money(cfg.capital, settle_currency)],
                default_leverage=Decimal(str(cfg.leverage)),
                book_type=BookType.L2_MBP,
            )
            engine.add_instrument(instrument)

            # Load L2 deltas and trades (loaders expect plain date strings)
            start_str = str(_to_utc_ts(start))
            end_str = str(_to_utc_ts(end)) if end is not None else None
            deltas = load_order_book_deltas(
                instrument,
                catalog_dir=cfg.data_dir,
                start=start_str,
                end=end_str,
                max_files=cfg.l2_max_files,
            )
            if deltas:
                engine.add_data(deltas)

            trades = load_trade_ticks(
                instrument,
                catalog_dir=cfg.data_dir,
                start=start_str,
                end=end_str,
                max_files=cfg.l2_max_files,
            )
            if trades:
                engine.add_data(trades)

            # Strategy setup
            StrategyClass, ConfigClass = get_strategy_class(cfg.strategy_name)
            strategy_kwargs = {
                "instrument_id": instrument.id,
                "capital": cfg.capital,
                "leverage": cfg.leverage,
                "backtest_start_date": _to_utc_ts(start).strftime("%Y-%m-%d"),
                "active_from": _to_utc_ts(start).isoformat(),
                **cfg.strategy_params,
            }

            annotations = getattr(ConfigClass, "__annotations__", {})
            if "bar_type" in annotations or hasattr(ConfigClass, "bar_type"):
                try:
                    interval_nt = parse_interval(cfg.interval)
                except ValueError as e:
                    raise ValueError(
                        f"Cannot build bar_type for L2 strategy "
                        f"'{cfg.strategy_name}': {e}"
                    ) from e
                bar_type = BarType.from_str(
                    f"{instrument.id.value}-{interval_nt}-LAST-EXTERNAL"
                )
                strategy_kwargs["bar_type"] = bar_type

            strategy_config = ConfigClass(**strategy_kwargs)

        # --------------------------------------------------------------
        # Bar (OHLCV) Execution Mode
        # --------------------------------------------------------------
        else:
            if df is None:
                feather_path = cfg.feather_path or find_feather(
                    cfg.exchange,
                    cfg.symbol,
                    cfg.interval,
                    search_dirs=[cfg.data_dir, "."],
                    start=cfg.start,
                    end=cfg.end,
                )
                if not feather_path:
                    return BacktestResult(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        error=f"No feather data found for {cfg.symbol} ({cfg.interval})",
                    )

                print(f"Loading data from {feather_path}...")
                try:
                    df = pd.read_feather(feather_path)
                except FileNotFoundError:
                    return BacktestResult(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        error=f"Feather file '{feather_path}' not found.",
                    )
            else:
                print("Using preloaded data frame for this window.")

            try:
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            except KeyError as e:
                return BacktestResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error=f"Feather file missing expected columns: {e}",
                )

            window_start = _to_utc_ts(start)
            window_end = _to_utc_ts(end) if end is not None else None
            if pre_sliced:
                # Caller sliced the frame (incl. warm-up bars before start);
                # only trim the upper bound.
                if window_end is not None:
                    df = df[df["timestamp"] <= window_end].reset_index(drop=True)
            else:
                df = df[df["timestamp"] >= window_start].reset_index(drop=True)
                if window_end is not None:
                    df = df[df["timestamp"] <= window_end].reset_index(drop=True)

            if len(df) < 2:
                return BacktestResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error=(
                        f"Not enough bars in [{window_start}, {end}] "
                        f"({len(df)} rows)."
                    ),
                )

            ref_price = float(df["close"].iloc[0])
            slippage_bps = cfg.slippage_ticks * cfg.tick_size / ref_price * 10000
            taker_fee = cfg.taker_fee + Decimal(str(slippage_bps)) / Decimal(10000)

            settle_currency = _CURRENCY_MAP.get(cfg.settle_currency)
            if settle_currency is None:
                settle_currency = Currency(
                    cfg.settle_currency, 2, 0, cfg.settle_currency, 0
                )

            interval_nt = parse_interval(cfg.interval)
            leverage_dec = Decimal(str(cfg.leverage))
            venue = Venue(cfg.exchange)
            self.venue = venue

            engine = BacktestEngine(config=BacktestEngineConfig())
            self.engine = engine

            engine.add_venue(
                venue=venue,
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN,
                base_currency=settle_currency,
                starting_balances=[Money(cfg.capital, settle_currency)],
                default_leverage=leverage_dec,
            )

            base_code = cfg.symbol.split("/")[0]
            base_currency = _CURRENCY_MAP.get(
                base_code,
                Currency(base_code, 2, 0, base_code, 0),
            )
            instrument = make_perpetual(
                cfg.exchange,
                cfg.symbol,
                cfg.maker_fee,
                taker_fee,
                base_currency=base_currency,
                settlement_currency=settle_currency,
                quote_currency=settle_currency,
            )
            engine.add_instrument(instrument)

            bar_type = BarType.from_str(
                f"{instrument.id.value}-{interval_nt}-LAST-EXTERNAL"
            )

            StrategyClass, ConfigClass = get_strategy_class(cfg.strategy_name)
            strategy_config = ConfigClass(
                instrument_id=instrument.id,
                bar_type=bar_type,
                capital=cfg.capital,
                leverage=cfg.leverage,
                backtest_start_date=window_start.strftime("%Y-%m-%d"),
                active_from=window_start.isoformat(),
                **cfg.strategy_params,
            )

            print(f"Loaded {len(df)} {cfg.interval} bars (ref_price={ref_price}).")
            bars = load_bars(df, bar_type, instrument)
            engine.add_data(bars)

            funding_path = find_feather(
                cfg.exchange,
                cfg.symbol,
                "funding",
                search_dirs=[cfg.data_dir, "."],
            )
            if funding_path:
                print(f"Loading funding data from {funding_path}...")
                df_funding = pd.read_feather(funding_path)
                df_funding = df_funding[
                    df_funding["timestamp"] >= window_start
                ].reset_index(drop=True)
                if window_end is not None:
                    df_funding = df_funding[
                        df_funding["timestamp"] <= window_end
                    ].reset_index(drop=True)
                funding_updates = load_funding_rates(df_funding, instrument.id)
                engine.add_data(funding_updates)
                print(f"Loaded {len(funding_updates)} funding rate updates.")
            else:
                print(
                    "No funding rate data found (file pattern: *funding*). Running without funding."
                )

        # -- Register stats --------------------------------------------
        engine.portfolio.analyzer.register_statistic(CalmarRatio())
        engine.portfolio.analyzer.register_statistic(AnnualizedReturn())
        run_params = {
            "pair": cfg.symbol,
            "exchange": cfg.exchange,
            "interval": cfg.interval,
            "capital": f"${cfg.capital}",
            "leverage": f"{cfg.leverage}x",
            "maker_fee": f"{float(instrument.maker_fee):.7f}%",
            "taker_fee": f"{float(instrument.taker_fee):.7f}%",
            "slippage_ticks": f"{cfg.slippage_ticks}",
            "strategy": cfg.strategy_name,
        }
        engine.portfolio.analyzer.register_statistic(RunConfigStatistic(**run_params))

        # -- Run -------------------------------------------------------
        strategy = StrategyClass(config=strategy_config)
        self.strategy = strategy
        engine.add_strategy(strategy)

        print("Running backtest...")
        engine.run()

        # -- Collect results -------------------------------------------
        stats_pnls = engine.portfolio.analyzer.get_performance_stats_pnls()
        stats_returns = engine.portfolio.analyzer.get_performance_stats_returns()
        stats_general = engine.portfolio.analyzer.get_performance_stats_general()
        all_stats = {**stats_pnls, **stats_returns, **stats_general}

        positions_df = engine.trader.generate_positions_report()
        fills_df = engine.trader.generate_fills_report()

        # Van Tharp SQN over per-trade returns of closed positions
        sqn = None
        if len(positions_df) and "realized_return" in positions_df.columns:
            closed = positions_df
            if "ts_closed" in closed.columns:
                closed = closed[closed["ts_closed"].notna()]
            rets = pd.to_numeric(closed["realized_return"], errors="coerce").dropna()
            sqn = system_quality_number(rets.tolist())

        # Extract optimisation objectives
        pnl = stats_pnls.get("PnL (total)")
        sharpe = stats_returns.get("Sharpe Ratio (252 days)")
        num_trades = len(positions_df)

        # Funding side-channel (typed tracker on SBTStrategy subclasses)
        funding_tracker = getattr(strategy, "funding", None)
        funding_pnl = (
            float(funding_tracker.total_paid) if funding_tracker is not None else 0.0
        )

        elapsed = time.monotonic() - t0

        return BacktestResult(
            job_id=job_id,
            status=JobStatus.DONE,
            sharpe_ratio=float(sharpe) if sharpe is not None else None,
            num_trades=num_trades,
            pnl=float(pnl) if pnl is not None else None,
            sqn=sqn,
            stats=all_stats,
            funding_pnl=funding_pnl,
            duration_seconds=elapsed,
            **_spill_artifacts(job_id, positions_df, fills_df),
        )
