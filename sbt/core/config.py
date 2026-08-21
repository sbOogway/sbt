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
    interval: str
    strategy_name: str
    strategy_params: dict = field(default_factory=dict)
    capital: Decimal = Decimal(1000)
    leverage: float = 1.0
    start: str = "2020-01-01"
    maker_fee: Decimal = Decimal("0.0")
    taker_fee: Decimal = Decimal("0.0")
    settle_currency: str = "USDT"
    slippage_ticks: int = 0
    tick_size: float = 0.1
    feather_path: str | None = None
    data_dir: str = "data"

    def with_overrides(self, params: dict) -> RunConfig:
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
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            settle_currency=self.settle_currency,
            slippage_ticks=self.slippage_ticks,
            tick_size=self.tick_size,
            feather_path=self.feather_path,
            data_dir=self.data_dir,
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
            "maker_fee": str(self.maker_fee),
            "taker_fee": str(self.taker_fee),
            "settle_currency": self.settle_currency,
            "slippage_ticks": self.slippage_ticks,
            "tick_size": self.tick_size,
            "feather_path": self.feather_path,
            "data_dir": self.data_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        """Reconstruct RunConfig from a dictionary."""
        return cls(
            exchange=d["exchange"],
            symbol=d["symbol"],
            interval=d["interval"],
            strategy_name=d["strategy_name"],
            strategy_params=d.get("strategy_params", {}),
            capital=Decimal(str(d.get("capital", "1000"))),
            leverage=float(d.get("leverage", 1.0)),
            start=d.get("start", "2020-01-01"),
            maker_fee=Decimal(str(d.get("maker_fee", "0.0"))),
            taker_fee=Decimal(str(d.get("taker_fee", "0.0"))),
            settle_currency=d.get("settle_currency", "USDT"),
            slippage_ticks=int(d.get("slippage_ticks", 0)),
            tick_size=float(d.get("tick_size", 0.1)),
            feather_path=d.get("feather_path"),
            data_dir=d.get("data_dir", "data"),
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
    ) -> RunConfig:
        """Build a RunConfig from a TOML file + optional CLI overrides.

        *cli_overrides* keys match CLI flag names (exchange, symbol,
        interval, leverage, start, feather).  ``None`` values are
        ignored so you can pass ``vars(args)`` directly.
        """
        path = Path(toml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("rb") as f:
            cfg = tomllib.load(f)

        run = cfg.get("run", {})
        all_strat = cfg.get("strategy", {})
        strat_params = dict(all_strat.get(strategy_name, {}))

        overrides = {k: v for k, v in (cli_overrides or {}).items() if v is not None}

        return cls(
            exchange=overrides.get("exchange", run.get("exchange", "BINANCE")),
            symbol=overrides.get("symbol", run.get("symbol", "BTC/USDT")),
            interval=overrides.get("interval", run.get("interval", "5m")),
            strategy_name=strategy_name,
            strategy_params=strat_params,
            capital=Decimal(str(run.get("capital", "1000"))),
            leverage=float(overrides.get("leverage", run.get("leverage", 1.0))),
            start=overrides.get("start", run.get("start", "2020-01-01")),
            maker_fee=Decimal(str(run.get("maker_fee", "0.0"))),
            taker_fee=Decimal(str(run.get("taker_fee", "0.0"))),
            settle_currency=run.get("settle_currency", "USDT"),
            slippage_ticks=int(run.get("slippage_ticks", 0)),
            tick_size=float(run.get("tick_size", 0.1)),
            feather_path=overrides.get("feather"),
            data_dir=run.get("data_dir", "data"),
        )

    @classmethod
    def parse_cli(cls) -> RunConfig:
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
                "feather": args.feather,
            },
        )
