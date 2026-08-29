"""Fully resolved backtest configuration."""

import argparse
import dataclasses
import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints


@dataclass
class RunConfig:
    """All parameters needed to execute a single backtest run.

    Constructed from a TOML config file + CLI overrides, or directly
    by the server/optimizer when dispatching jobs.

    Serialization is fields-derived: :meth:`to_dict` / :meth:`from_dict`
    walk ``dataclasses.fields`` and coerce values to the annotated types,
    so adding a field never requires touching the codec.
    """

    exchange: str
    # Single-instrument mode (``symbol`` set, ``symbols`` empty) and
    # multi-instrument portfolio mode (``symbols`` set, ``symbol``
    # empty) are both valid. At least one must be set; ``from_toml``
    # and ``parse_cli`` enforce this. ``symbol`` is kept on the
    # dataclass for backward compat with the runner's internal use.
    symbol: str = ""
    # Multi-instrument (portfolio) mode. When non-empty and len > 1 the run
    # executes one engine across all symbols on a shared margin account;
    # the strategy must be a portfolio strategy. Empty => single-instrument
    # mode keyed on ``symbol``.
    symbols: list[str] = field(default_factory=list)
    interval: str = "5m"
    strategy_name: str = "bitcoin_intraday_momentum"
    strategy_params: dict = field(default_factory=dict)
    capital: Decimal = Decimal("1000")
    leverage: float = 1.0
    start: str = "2020-01-01"
    end: str | None = None
    maker_fee: Decimal = Decimal("0.0")
    taker_fee: Decimal = Decimal("0.0")
    settle_currency: str = "USDT"
    slippage_ticks: int = 0
    tick_size: float = 0.1
    feather_path: str | None = None
    data_dir: str = "data"
    data_type: str = "bar"  # 'bar' or 'l2'
    l2_max_files: int | None = None
    # Holdout split: fraction of the data span used for in-sample; the rest
    # is out-of-sample (e.g. 0.7 -> 70% train / 30% validation).
    train_val_split: float | None = None
    # Bars loaded before each window's trading start so indicators/plugins
    # warm up without polluting window metrics (orders are gated off).
    warmup_bars: int | None = None
    # Open the generated tearsheet in a browser after the run.
    open_report: bool = True

    @property
    def all_symbols(self) -> list[str]:
        """The effective tradeable list: ``symbols`` when non-empty else [symbol]."""
        if self.symbols:
            return list(self.symbols)
        if self.symbol:
            return [self.symbol]
        return []

    def __post_init__(self) -> None:
        # ``exchange``, ``symbol``/``symbols``, ``interval`` are now
        # CLI-only (ticket #58 Part 2). They may be empty when
        # constructed by the test suite or the legacy DB loader, but
        # the CLI parser enforces them. Warn when both ``symbol`` and
        # ``symbols`` are empty (would be a no-op run).
        if not self.symbol and not self.symbols:
            import warnings
            warnings.warn(
                "RunConfig has neither symbol nor symbols set; the run "
                "will be a no-op.",
                stacklevel=2,
            )

    def with_overrides(self, params: dict) -> "RunConfig":
        """Return a copy with strategy_params updated from *params*."""
        return dataclasses.replace(
            self, strategy_params={**self.strategy_params, **params}
        )

    def to_dict(self) -> dict:
        """Convert RunConfig to a JSON-serializable dictionary."""
        hints = get_type_hints(type(self))
        return {
            f.name: (
                str(getattr(self, f.name))
                if hints[f.name] is Decimal
                else getattr(self, f.name)
            )
            for f in dataclasses.fields(self)
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Reconstruct RunConfig from a dictionary (unknown keys ignored).

        Values arriving over JSON / CLI strings are coerced to the annotated
        field types so downstream math never sees e.g. a string leverage.
        """
        known = {f.name: f for f in dataclasses.fields(cls)}
        hints = get_type_hints(cls)
        kwargs = {}
        for key, value in d.items():
            if key not in known or value is None:
                continue
            kwargs[key] = _coerce(value, hints[key])
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(
        cls,
        toml_path: str | Path,
        strategy_name: str,
        cli_overrides: dict | None = None,
    ) -> "RunConfig":
        """Build a RunConfig from a TOML file + optional CLI overrides.

        The ``[run]`` table maps 1:1 onto field names; CLI overrides win.

        ``exchange``, ``symbol``, and ``interval`` are NOT loaded from
        TOML — they must come from the CLI (or be inferred from
        ``--feather``). Old config.toml files that still carry them
        are silently dropped here.
        """
        path = Path(toml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("rb") as f:
            cfg = tomllib.load(f)

        data = dict(cfg.get("run", {}))
        # Drop fields that are now CLI-only. ``from_dict`` also ignores
        # unknown keys, but we strip them here so the dataclass
        # constructor never sees them.
        for k in ("exchange", "symbol", "interval"):
            data.pop(k, None)
        for k, v in (cli_overrides or {}).items():
            if v is not None:
                data["feather_path" if k == "feather" else k] = v

        # Infer L2 mode from instrument format or explicit request.
        if "-LINEAR." in str(data.get("symbol", "")) or data.get("data_type") == "l2":
            data["data_type"] = "l2"

        data["strategy_name"] = strategy_name
        return cls.from_dict(data)

    @classmethod
    def parse_cli(cls) -> "RunConfig":
        """Parse CLI arguments and build a RunConfig.

        ``exchange``, ``symbol`` (or ``--symbols`` for multi-instrument),
        and ``interval`` are required CLI args. If ``--feather PATH`` is
        given, they are inferred from the filename instead
        (``data/{exchange}_{symbol}_{tag}_{start}_{end}.feather``).
        """
        parser = argparse.ArgumentParser(description="Run a strategy backtest")
        parser.add_argument(
            "--config",
            default="config.toml",
            help="Path to TOML config file (default: config.toml)",
        )
        parser.add_argument(
            "--feather",
            help=(
                "Path to a single feather file. When given, exchange/symbol/"
                "interval are inferred from the filename and the other "
                "three become optional."
            ),
        )
        parser.add_argument(
            "--exchange",
            help="Exchange/venue name (e.g. 'bybit'). Required unless --feather is given.",
        )
        parser.add_argument(
            "--symbol",
            help="Single trading pair (e.g. 'BTC/USDT:USDT'). Required unless --feather is given.",
        )
        parser.add_argument(
            "--symbols",
            action="append",
            metavar="SYMBOL[,SYMBOL...]",
            help=(
                "Multi-instrument symbols (comma-separated or repeatable); "
                "enables portfolio mode when more than one is given. "
                "Required (instead of --symbol) for portfolio strategies."
            ),
        )
        parser.add_argument(
            "--interval",
            help="Candle interval (e.g. '1d', '1h'). Required unless --feather is given.",
        )
        parser.add_argument("--leverage", help="Override leverage from config")
        parser.add_argument("--start", help="Override backtest start date from config")
        parser.add_argument("--end", help="Override backtest end date from config")
        parser.add_argument(
            "--warmup-bars",
            type=int,
            help=(
                "Bars loaded before each window's trading start for indicator "
                "warm-up (used with --train-val-split)"
            ),
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
            "--train-val-split",
            type=float,
            metavar="FRACTION",
            help=(
                "Holdout split: fraction of data span for in-sample training; "
                "remainder runs as out-of-sample validation (e.g. 0.7)"
            ),
        )
        parser.add_argument(
            "--no-open",
            action="store_true",
            help="Do not open the tearsheet in a browser after the run",
        )
        parser.add_argument(
            "--param",
            action="append",
            metavar="NAME=VALUE",
            help=(
                "Override a single strategy parameter, e.g. "
                "--param entry_threshold=0.6 (repeatable)"
            ),
        )
        parser.add_argument(
            "--strategy",
            default="bitcoin_intraday_momentum",
            help="Strategy section name in config (default: bitcoin_intraday_momentum)",
        )
        args = parser.parse_args()

        # Resolve exchange/symbol/interval: CLI > --feather inference.
        # ``--symbols`` (multi-instrument) overrides ``--symbol`` when both given.
        feather_inferred = None
        if args.feather:
            from .feather import infer_instrument_from_path
            inferred = infer_instrument_from_path(args.feather)
            if inferred is None:
                raise ValueError(
                    f"Could not infer exchange/symbol/interval from "
                    f"--feather path {args.feather!r}. Expected a name "
                    f"like 'data/bybit_BTCUSDT:USDT_1d_20230101_20260827."
                    f"feather'. Pass --exchange/--symbol/--interval "
                    f"explicitly."
                )
            feather_inferred = inferred
        # CLI wins over inference; inference fills in missing CLI args.
        exchange = (args.exchange or (feather_inferred[0] if feather_inferred else None))
        interval = (args.interval or (feather_inferred[2] if feather_inferred else None))
        if args.symbols:
            symbol = None  # multi-instrument mode
            symbols = _flatten_symbols(args.symbols)
        elif args.symbol:
            symbol = args.symbol
            symbols = None
        else:
            symbol = feather_inferred[1] if feather_inferred else None
            symbols = None

        missing = []
        if not exchange:
            missing.append("--exchange")
        if not symbol and not symbols:
            missing.append("--symbol or --symbols")
        if not interval:
            missing.append("--interval")
        if missing:
            raise SystemExit(
                f"Missing required CLI args: {', '.join(missing)}. "
                f"Either pass them explicitly or supply --feather PATH "
                f"to infer them from the filename."
            )

        cli_overrides = {
            "exchange": exchange,
            "symbol": symbol,
            "symbols": symbols,
            "interval": interval,
            "leverage": args.leverage,
            "start": args.start,
            "end": args.end,
            "feather": args.feather,
            "warmup_bars": args.warmup_bars,
            "data_type": args.data_type,
            "l2_max_files": args.l2_max_files,
            "train_val_split": args.train_val_split,
        }
        if args.no_open:
            cli_overrides["open_report"] = False
        cfg = cls.from_toml(
            toml_path=args.config,
            strategy_name=args.strategy,
            cli_overrides=cli_overrides,
        )
        if args.param:
            overrides = {}
            for spec in args.param:
                name, _, raw = spec.partition("=")
                if not name or not raw:
                    raise ValueError(f"Invalid --param '{spec}': expected NAME=VALUE")
                overrides[name.strip()] = _parse_scalar(raw)
            cfg = cfg.with_overrides(overrides)
        return cfg


def _flatten_symbols(values) -> list[str] | None:
    """Flatten appended ``--symbols`` specs into a single list of symbols."""
    if not values:
        return None
    out = []
    for spec in values:
        for part in spec.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _parse_scalar(raw: str):
    """Infer the type of a CLI scalar: int -> float -> bool -> str."""
    text = raw.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    return text


def _coerce(value, tp):
    """Coerce *value* to the annotated type *tp* (JSON/CLI friendly)."""
    origin = get_origin(tp)
    if origin is Union:  # Optional[X] -> X
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) != 1:
            return value
        tp = args[0]
    if tp is Decimal:
        return Decimal(str(value))
    if tp in (int, float, str):
        return tp(value)
    if tp is bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes"}
        return bool(value)
    return value
