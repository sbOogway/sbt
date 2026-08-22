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
    symbol: str
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
                str(getattr(self, f.name)) if hints[f.name] is Decimal else getattr(self, f.name)
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
        """
        path = Path(toml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("rb") as f:
            cfg = tomllib.load(f)

        data = dict(cfg.get("run", {}))
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
        """Parse CLI arguments and build a RunConfig."""
        parser = argparse.ArgumentParser(description="Run a strategy backtest")
        parser.add_argument(
            "--config",
            default="config.toml",
            help="Path to TOML config file (default: config.toml)",
        )
        parser.add_argument(
            "--feather", help="Path to feather file (auto-detect if omitted)"
        )
        parser.add_argument("--exchange", help="Override exchange from config")
        parser.add_argument("--symbol", help="Override trading pair from config")
        parser.add_argument("--interval", help="Override candle interval from config")
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
            "--strategy",
            default="bitcoin_intraday_momentum",
            help="Strategy section name in config (default: bitcoin_intraday_momentum)",
        )
        args = parser.parse_args()

        return cls.from_toml(
            toml_path=args.config,
            strategy_name=args.strategy,
            cli_overrides={
                "exchange": args.exchange,
                "symbol": args.symbol,
                "interval": args.interval,
                "leverage": args.leverage,
                "start": args.start,
                "end": args.end,
                "feather": args.feather,
                "warmup_bars": args.warmup_bars,
                "data_type": args.data_type,
                "l2_max_files": args.l2_max_files,
                "train_val_split": args.train_val_split,
                "open_report": not args.no_open,
            },
        )


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
