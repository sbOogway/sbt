"""Reusable backtest runner extracted from the monolithic __main__ block.

Usage (standalone)::

    from sbt.core.config import RunConfig
    from sbt.core.runner import BacktestRunner

    cfg = RunConfig.from_toml("config.toml", "overnight_drift")
    runner = BacktestRunner(cfg)
    result = runner.run()
    # runner.engine / runner.venue available for report generation
"""

from decimal import Decimal
import dataclasses
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
from .feather import find_feather, to_utc_ts
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


def _fmt_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:+,.2f}"
    return f"{value:,}"


# ------------------------------------------------------------------
# Shared assembly helpers (used by both bar and L2 execution modes)
# ------------------------------------------------------------------

_BARS_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _fail(job_id: str, error: str) -> BacktestResult:
    return BacktestResult(job_id=job_id, status=JobStatus.FAILED, error=error)


def _resolve_currency(code: str):
    """Map a currency code onto a Nautilus Currency, synthesizing when unknown."""
    cur = _CURRENCY_MAP.get(code)
    return cur if cur is not None else Currency(code, 2, 0, code, 0)


def _select_bars_columns(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Restrict a frame to the six OHLCV columns; report missing ones."""
    try:
        return df[_BARS_COLUMNS], None
    except KeyError as e:
        return None, f"Bars frame missing expected columns: {e}"


def _discover_bars(
    cfg: RunConfig, start_ts: pd.Timestamp, end_ts: pd.Timestamp | None
) -> tuple[pd.DataFrame | None, str | None]:
    """Resolve, load, column-select and date-filter the feather OHLCV frame."""
    feather_path = cfg.feather_path or find_feather(
        cfg.exchange,
        cfg.symbol,
        cfg.interval,
        search_dirs=[cfg.data_dir, "."],
        start=cfg.start,
        end=cfg.end,
    )
    if not feather_path:
        return None, f"No feather data found for {cfg.symbol} ({cfg.interval})"

    print(f"Loading data from {feather_path}...")
    try:
        df = pd.read_feather(feather_path)
    except FileNotFoundError:
        return None, f"Feather file '{feather_path}' not found."

    df, err = _select_bars_columns(df)
    if err:
        return None, err

    df = df[df["timestamp"] >= start_ts].reset_index(drop=True)
    if end_ts is not None:
        df = df[df["timestamp"] <= end_ts].reset_index(drop=True)
    return df, None


def _slice_frame(
    df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp | None
) -> pd.DataFrame:
    """Trim a timestamped frame to [start_ts, end_ts] (inclusive)."""
    out = df[df["timestamp"] >= start_ts].reset_index(drop=True)
    if end_ts is not None:
        out = out[out["timestamp"] <= end_ts].reset_index(drop=True)
    return out


def _add_venue(
    engine: BacktestEngine,
    venue: Venue,
    *,
    settle_currency,
    capital: Decimal,
    leverage: float,
    book_type=None,
) -> None:
    kwargs = dict(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=settle_currency,
        starting_balances=[Money(capital, settle_currency)],
        default_leverage=Decimal(str(leverage)),
    )
    if book_type is not None:
        kwargs["book_type"] = book_type
    engine.add_venue(**kwargs)


def _base_strategy_kwargs(cfg: RunConfig, instrument_id, start_ts) -> dict:
    """Strategy-config fields every execution mode must supply."""
    start_ts = to_utc_ts(start_ts)
    return {
        "instrument_id": instrument_id,
        "capital": cfg.capital,
        "leverage": cfg.leverage,
        "backtest_start_date": start_ts.strftime("%Y-%m-%d"),
        "active_from": start_ts.isoformat(),
        **cfg.strategy_params,
    }


def _build_strategy_config(ConfigClass, **kwargs):
    """Construct a strategy config, rejecting unknown keys loudly.

    msgspec Structs reject unknown kwargs anyway, but the error names no
    remedy; optimizer/server typos deserve a message listing the valid
    fields. Silent defaults are disabled to keep optimizer trials honest.
    """
    known = tuple(ConfigClass.__struct_fields__)
    unknown = [key for key in kwargs if key not in known]
    if unknown:
        raise ValueError(
            f"Unknown parameters {sorted(unknown)} for "
            f"{ConfigClass.__name__}. Valid fields: {sorted(known)}."
        )
    return ConfigClass(**kwargs)


def _register_stats(engine: BacktestEngine, cfg: RunConfig, instrument) -> None:
    analyzer = engine.portfolio.analyzer
    analyzer.register_statistic(CalmarRatio())
    analyzer.register_statistic(AnnualizedReturn())
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
    analyzer.register_statistic(RunConfigStatistic(**run_params))


def _collect_result(
    engine: BacktestEngine, strategy, job_id: str, t0: float
) -> BacktestResult:
    """Extract stats, reports and objectives from a finished engine."""
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

    return BacktestResult(
        job_id=job_id,
        status=JobStatus.DONE,
        sharpe_ratio=float(sharpe) if sharpe is not None else None,
        num_trades=num_trades,
        pnl=float(pnl) if pnl is not None else None,
        sqn=sqn,
        stats=all_stats,
        funding_pnl=funding_pnl,
        duration_seconds=time.monotonic() - t0,
        **_spill_artifacts(job_id, positions_df, fills_df),
    )


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

    def run(
        self,
        job_id: str = "standalone",
        bars: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Execute the backtest and return a structured result.

        Data enters through an explicit-frame seam so the runner can run
        headless (tests, notebooks) without touching the filesystem:

        * ``bars=None`` (default) resolves OHLCV data via the feather
          naming convention, exactly as before.
        * An explicit ``bars`` frame is used **as-is** — the caller owns
          content, slicing and warm-up; no file is read.
        * An explicit ``funding`` frame is sliced to each window's bounds
          and injected into the engine. Explicit bars without a funding
          frame run without funding instead of searching the disk.

        A configured runner plugin (flat ``train_val_split`` field today)
        expands the job into windows; each window runs through the normal
        execution path and the plugin merges them into one result.
        """
        plugin = resolve_runner_plugin(self.config)
        if plugin is not None:
            return self._run_windows(job_id, plugin, bars=bars, funding=funding)
        return self._run_window(
            job_id, self.config.start, self.config.end, bars=bars, funding=funding
        )

    # ------------------------------------------------------------------
    # Windowed execution (runner plugins)
    # ------------------------------------------------------------------

    def _run_windows(
        self,
        job_id: str,
        plugin: RunnerPlugin,
        bars: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> BacktestResult:
        cfg = self.config

        df = bars
        if df is None and cfg.data_type != "l2":
            df, err = _discover_bars(
                cfg, to_utc_ts(cfg.start), to_utc_ts(cfg.end)
            )
            if err:
                return _fail(job_id, err)

        try:
            windows: dict[str, Window] = plugin.expand(cfg, df)
        except ValueError as e:
            return _fail(job_id, str(e))

        self.window_engines = {}
        results: dict[str, BacktestResult] = {}
        for key, win in windows.items():
            print(f"\n--- {win.label} window: {win.start} -> {win.end} ---")
            res = self._run_window(
                f"{job_id}:{key}",
                win.start,
                win.end,
                bars=win.df,
                funding=funding,
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
        bars: pd.DataFrame | None = None,
        funding: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Execute one backtest over the [start, end] window.

        ``bars`` carries an OHLCV frame; when supplied it is trusted
        as-is (runner-plugin windows arrive pre-sliced with any warm-up
        bars ahead of *start*, trading gated via strategy ``active_from``).
        When ``None``, the feather convention resolves the frame.
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
                    return _fail(
                        job_id,
                        f"No L2 instruments found in catalog '{cfg.data_dir}'",
                    )

            print(f"Loading L2 instrument '{inst_id_str}' from {cfg.data_dir}...")
            try:
                instrument = load_l2_instrument(inst_id_str, catalog_dir=cfg.data_dir)
            except Exception as e:
                return _fail(
                    job_id, f"Failed loading L2 instrument {inst_id_str}: {e}"
                )

            venue = instrument.id.venue
            self.venue = venue
            engine = BacktestEngine(config=BacktestEngineConfig())
            self.engine = engine

            settle_currency = (
                instrument.settlement_currency
                or _resolve_currency(cfg.settle_currency)
            )
            _add_venue(
                engine,
                venue,
                settle_currency=settle_currency,
                capital=cfg.capital,
                leverage=cfg.leverage,
                book_type=BookType.L2_MBP,
            )
            engine.add_instrument(instrument)

            # Load L2 deltas and trades (loaders expect plain date strings)
            start_str = str(to_utc_ts(start))
            end_str = str(to_utc_ts(end)) if end is not None else None
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

            StrategyClass, ConfigClass = get_strategy_class(cfg.strategy_name)
            strategy_kwargs = _base_strategy_kwargs(cfg, instrument.id, start)
            # L2 configs subclass SBTStrategyConfig directly: no bar_type.
            try:
                strategy_config = _build_strategy_config(
                    ConfigClass, **strategy_kwargs
                )
            except (TypeError, ValueError) as e:
                # TypeError: missing required injected field (wrong tier);
                # ValueError: unknown strategy_params key.
                return _fail(job_id, str(e))

        # --------------------------------------------------------------
        # Bar (OHLCV) Execution Mode
        # --------------------------------------------------------------
        else:
            window_start = to_utc_ts(start)
            window_end = to_utc_ts(end) if end is not None else None

            if bars is None:
                df, err = _discover_bars(cfg, window_start, window_end)
                if err:
                    return _fail(job_id, err)
            else:
                print("Using caller-supplied data frame for this window.")
                df, err = _select_bars_columns(bars)
                if err:
                    return _fail(job_id, err)

            if len(df) < 2:
                return _fail(
                    job_id,
                    f"Not enough bars in [{window_start}, {end}] ({len(df)} rows).",
                )

            ref_price = float(df["close"].iloc[0])
            slippage_bps = cfg.slippage_ticks * cfg.tick_size / ref_price * 10000
            taker_fee = cfg.taker_fee + Decimal(str(slippage_bps)) / Decimal(10000)

            settle_currency = _resolve_currency(cfg.settle_currency)
            interval_nt = parse_interval(cfg.interval)
            venue = Venue(cfg.exchange)
            self.venue = venue

            engine = BacktestEngine(config=BacktestEngineConfig())
            self.engine = engine

            _add_venue(
                engine,
                venue,
                settle_currency=settle_currency,
                capital=cfg.capital,
                leverage=cfg.leverage,
            )

            base_code = cfg.symbol.split("/")[0]
            instrument = make_perpetual(
                cfg.exchange,
                cfg.symbol,
                cfg.maker_fee,
                taker_fee,
                base_currency=_resolve_currency(base_code),
                settlement_currency=settle_currency,
                quote_currency=settle_currency,
            )
            engine.add_instrument(instrument)

            bar_type = BarType.from_str(
                f"{instrument.id.value}-{interval_nt}-LAST-EXTERNAL"
            )

            StrategyClass, ConfigClass = get_strategy_class(cfg.strategy_name)
            strategy_kwargs = _base_strategy_kwargs(cfg, instrument.id, window_start)
            # Bar-mode configs subclass SBTBarStrategyConfig: bar_type required.
            strategy_kwargs["bar_type"] = bar_type
            try:
                strategy_config = _build_strategy_config(
                    ConfigClass, **strategy_kwargs
                )
            except (TypeError, ValueError) as e:
                # TypeError: missing required injected field (wrong tier);
                # ValueError: unknown strategy_params key.
                return _fail(job_id, str(e))

            print(f"Loaded {len(df)} {cfg.interval} bars (ref_price={ref_price}).")
            bar_updates = load_bars(df, bar_type, instrument)
            engine.add_data(bar_updates)

            # Funding side-channel: injected frame > feather discovery > none.
            funding_df = None
            if funding is not None:
                funding_df = _slice_frame(funding, window_start, window_end)
            elif bars is None:
                funding_path = find_feather(
                    cfg.exchange,
                    cfg.symbol,
                    "funding",
                    search_dirs=[cfg.data_dir, "."],
                )
                if funding_path:
                    print(f"Loading funding data from {funding_path}...")
                    funding_df = pd.read_feather(funding_path)
            else:
                print("Explicit bars without funding frame; running without funding.")
            if funding_df is not None:
                funding_updates = load_funding_rates(funding_df, instrument.id)
                if funding_updates:
                    engine.add_data(funding_updates)
                    print(f"Loaded {len(funding_updates)} funding rate updates.")

        # -- Shared tail: register stats, run, collect -------------------
        _register_stats(engine, cfg, instrument)
        strategy = StrategyClass(config=strategy_config)
        self.strategy = strategy
        engine.add_strategy(strategy)

        print("Running backtest...")
        engine.run()

        return _collect_result(engine, strategy, job_id, t0)
