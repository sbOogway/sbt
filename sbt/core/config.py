"""Fully resolved backtest configuration."""

import argparse
import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path


@dataclass
class RunConfig:
    """All parameters needed to execute a single backtest run.

    Constructed from a TOML config file + CLI overrides, or directly
    by the server/optimizer when dispatching jobs.
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

    def with_overrides(self, params: dict) -> "RunConfig":
        """Return a copy with strategy_params updated from *params*."""
        merged = {**self.strategy_params, **params}
        return RunConfig(
            exchange=self.exchange,
            symbol=self.symbol,
            interval=self.interval,
            strategy_name=self.strategy_name,
            strategy_params=merged,
            capital=self.capital,
            leverage=self.leverage,
            start=self.start,
            end=self.end,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            settle_currency=self.settle_currency,
            slippage_ticks=self.slippage_ticks,
            tick_size=self.tick_size,
            feather_path=self.feather_path,
            data_dir=self.data_dir,
            data_type=self.data_type,
            l2_max_files=self.l2_max_files,
        )

    def to_dict(self) -> dict:
        """Convert RunConfig to a JSON-serializable dictionary."""
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "interval": self.interval,
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
            "capital": str(self.capital),
            "leverage": self.leverage,
            "start": self.start,
            "end": self.end,
            "maker_fee": str(self.maker_fee),
            "taker_fee": str(self.taker_fee),
            "settle_currency": self.settle_currency,
            "slippage_ticks": self.slippage_ticks,
            "tick_size": self.tick_size,
            "feather_path": self.feather_path,
            "data_dir": self.data_dir,
            "data_type": self.data_type,
            "l2_max_files": self.l2_max_files,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Reconstruct RunConfig from a dictionary."""
        return cls(
            exchange=d["exchange"],
            symbol=d["symbol"],
            interval=d.get("interval", "5m"),
            strategy_name=d["strategy_name"],
            strategy_params=d.get("strategy_params", {}),
            capital=Decimal(str(d.get("capital", "1000"))),
            leverage=float(d.get("leverage", 1.0)),
            start=d.get("start", "2020-01-01"),
            end=d.get("end"),
            maker_fee=Decimal(str(d.get("maker_fee", "0.0"))),
            taker_fee=Decimal(str(d.get("taker_fee", "0.0"))),
            settle_currency=d.get("settle_currency", "USDT"),
            slippage_ticks=int(d.get("slippage_ticks", 0)),
            tick_size=float(d.get("tick_size", 0.1)),
            feather_path=d.get("feather_path"),
            data_dir=d.get("data_dir", "data"),
            data_type=d.get("data_type", "bar"),
            l2_max_files=d.get("l2_max_files"),
        )

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
        """Build a RunConfig from a TOML file + optional CLI overrides."""
        path = Path(toml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("rb") as f:
            cfg = tomllib.load(f)

        run = cfg.get("run", {})

        overrides = {k: v for k, v in (cli_overrides or {}).items() if v is not None}

        # Infer l2 data_type if instrument format matches or explicitly set
        data_type = overrides.get("data_type", run.get("data_type", "bar"))
        symbol = overrides.get("symbol", run.get("symbol", "BTC/USDT"))
        if "-LINEAR." in symbol or data_type == "l2":
            data_type = "l2"

        return cls(
            exchange=overrides.get("exchange", run.get("exchange", "BINANCE")),
            symbol=symbol,
            interval=overrides.get("interval", run.get("interval", "5m")),
            strategy_name=strategy_name,
            strategy_params={},
            capital=Decimal(str(run.get("capital", "1000"))),
            leverage=float(overrides.get("leverage", run.get("leverage", 1.0))),
            start=overrides.get("start", run.get("start", "2020-01-01")),
            end=overrides.get("end", run.get("end")),
            maker_fee=Decimal(str(run.get("maker_fee", "0.0"))),
            taker_fee=Decimal(str(run.get("taker_fee", "0.0"))),
            settle_currency=run.get("settle_currency", "USDT"),
            slippage_ticks=int(run.get("slippage_ticks", 0)),
            tick_size=float(run.get("tick_size", 0.1)),
            feather_path=overrides.get("feather"),
            data_dir=run.get("data_dir", "data"),
            data_type=data_type,
            l2_max_files=overrides.get("l2_max_files", run.get("l2_max_files")),
        )

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
                "data_type": args.data_type,
                "l2_max_files": args.l2_max_files,
            },
        )
