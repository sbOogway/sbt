"""Fully resolved backtest configuration."""

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
    # empty) are both valid. At least one must be set; :func:`from_cli_args`
    # (the CLI assembly path) enforces this. ``symbol`` is kept on the
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
    def from_cli_args(cls, args) -> "RunConfig":
        """Build a RunConfig from a parsed argparse Namespace.

        Single entry point for the ``sbt backtest`` and ``sbt optimize
        --walk-forward`` subcommands. Resolves ``--feather`` inference,
        validates the required exchange/symbol/interval triple, builds
        the cli_overrides dict, then applies any ``--param`` overrides.

        Raises ``ValueError`` on missing required args or unparseable
        ``--feather``; CLI subcommands translate that to a user-facing
        error and ``sys.exit(1)``.
        """
        cli_overrides = cli_overrides_from_args(args)
        cfg = cls.from_toml(
            toml_path=args.config,
            strategy_name=args.strategy,
            cli_overrides=cli_overrides,
        )
        overrides = param_overrides_from_args(args)
        if overrides:
            cfg = cfg.with_overrides(overrides)
        return cfg


# ----------------------------------------------------------------------
# CLI assembly helpers (public so subcommand modules can reuse them)
# ----------------------------------------------------------------------


def cli_overrides_from_args(args) -> dict:
    """Return the ``cli_overrides`` dict for *args*.

    Handles the full CLI -> RunConfig preamble in one place:

    - ``--feather PATH`` inference: when given, exchange/symbol/interval
      are filled in from the filename unless the user supplied them
      explicitly. CLI values always win over inference.
    - Multi-instrument mode: ``--symbols`` (with one or more values)
      wins over ``--symbol`` and enables portfolio mode.
    - Required-arg validation: exchange, (symbol or symbols), and
      interval must all be set after inference; raises ``ValueError``
      listing the missing ones otherwise.
    - ``--no-open`` flips ``open_report`` to False.

    ``optimize`` hands the returned dict straight to ``run_optuna_study``;
    ``backtest`` and ``--walk-forward`` go through :meth:`from_cli_args`.
    """
    feather_inferred = None
    if args.feather:
        from .feather import infer_instrument_from_path

        inferred = infer_instrument_from_path(args.feather)
        if inferred is None:
            raise ValueError(
                f"Could not infer exchange/symbol/interval from "
                f"--feather path {args.feather!r}. Expected a name like "
                f"'data/bybit_BTCUSDT:USDT_1d_20230101_20260827.feather'. "
                f"Pass --exchange/--symbol/--interval explicitly."
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
        raise ValueError(
            f"Missing required CLI args: {', '.join(missing)}. "
            f"Either pass them explicitly or supply --feather PATH "
            f"to infer them from the filename."
        )

    overrides = {
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
    if getattr(args, "no_open", False):
        overrides["open_report"] = False
    return overrides


def param_overrides_from_args(args) -> dict:
    """Parse ``--param NAME=VALUE`` specs from *args* into a dict.

    Each spec is a string like ``entry_threshold=0.6``; values are
    type-coerced via :func:`_parse_scalar` (int -> float -> bool -> str).
    Raises ``ValueError`` on a malformed spec (no ``=``).
    """
    if not getattr(args, "param", None):
        return {}
    overrides = {}
    for spec in args.param:
        name, _, raw = spec.partition("=")
        if not name or not raw:
            raise ValueError(f"Invalid --param '{spec}': expected NAME=VALUE")
        overrides[name.strip()] = _parse_scalar(raw)
    return overrides


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
