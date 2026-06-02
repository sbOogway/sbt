import argparse
import glob
import tomllib
from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Price, Quantity

from .stats import AnnualizedReturn, CalmarRatio, RunConfig
from .utils import make_perpetual, parse_interval, get_strategy_class


def load_bars(df: pd.DataFrame, bar_type: BarType) -> list[Bar]:
    bars = []
    for row in df.itertuples(index=False):
        ts_nanos = dt_to_unix_nanos(row.timestamp)
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price(row.open, precision=1),
                high=Price(row.high, precision=1),
                low=Price(row.low, precision=1),
                close=Price(row.close, precision=1),
                volume=Quantity(row.volume, precision=3),
                ts_event=ts_nanos,
                ts_init=ts_nanos,
            )
        )
    return bars


def find_feather(exchange: str, symbol: str, interval: str) -> str | None:
    raw_symbol = symbol.replace("/", "")
    search_dirs = ["data", "."]
    for d in search_dirs:
        pattern = f"{d}/{exchange.lower()}_{raw_symbol}_{interval}_*.feather"
        files = sorted(glob.glob(pattern))
        if files:
            return files[-1]
        pattern = f"{d}/{raw_symbol}_{interval}_*.feather"
        files = sorted(glob.glob(pattern))
        if files:
            return files[-1]
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a strategy backtest")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to TOML config file (default: config.toml)",
    )
    parser.add_argument("--feather", help="Path to feather file (auto-detect if omitted)")
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

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config file not found: {cfg_path}")
        exit(1)

    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)

    run = cfg.get("run", {})
    all_strat_sections = cfg.get("strategy", {})
    strat_params = all_strat_sections.get(args.strategy, {})

    ex = args.exchange or run.get("exchange", "BINANCE")
    sym = args.symbol or run.get("symbol", "BTC/USDT")
    interval_ccxt = args.interval or run.get("interval", "5m")
    capital = Decimal(str(run.get("capital", "1000")))
    leverage_val = float(args.leverage or run.get("leverage", 1.0))
    start = args.start or run.get("start", "2020-01-01")
    maker_fee = Decimal(str(run.get("maker_fee", "0.0")))
    taker_fee = Decimal(str(run.get("taker_fee", "0.0")))

    interval_nt = parse_interval(interval_ccxt)
    leverage_dec = Decimal(str(leverage_val))

    venue = Venue(ex)
    engine = BacktestEngine(config=BacktestEngineConfig())

    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(capital, USDT)],
        default_leverage=leverage_dec,
    )

    instrument = make_perpetual(ex, sym, maker_fee, taker_fee)
    engine.add_instrument(instrument)

    bar_type = BarType.from_str(f"{instrument.id.value}-{interval_nt}-LAST-EXTERNAL")

    StrategyClass, ConfigClass = get_strategy_class(args.strategy)

    strategy_config = ConfigClass(
        instrument_id=instrument.id,
        bar_type=bar_type,
        capital=capital,
        leverage=leverage_val,
        backtest_start_date=start,
        **strat_params,
    )

    feather_path = args.feather or find_feather(ex, sym, interval_ccxt)
    if not feather_path:
        print(f"ERROR: No feather data found for {sym} ({interval_ccxt})")
        exit(1)

    print(f"Loading data from {feather_path}...")
    try:
        df = pd.read_feather(feather_path)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        start_ts = pd.Timestamp(strategy_config.backtest_start_date, tz="UTC")
        df = df[df["timestamp"] >= start_ts].reset_index(drop=True)
        bars = load_bars(df, bar_type)
        engine.add_data(bars)
        print(f"Loaded {len(bars)} {interval_ccxt} bars.")
    except FileNotFoundError:
        print(f"ERROR: Feather file '{feather_path}' not found.")
        exit(1)

    engine.portfolio.analyzer.register_statistic(CalmarRatio())
    engine.portfolio.analyzer.register_statistic(AnnualizedReturn())
    run_params = {
        "pair": sym,
        "exchange": ex,
        "interval": interval_ccxt,
        "capital": f"${capital}",
        "leverage": f"{leverage_val}x",
        "maker_fee": f"{float(instrument.maker_fee) :.4f}%",
        "taker_fee": f"{float(instrument.taker_fee) :.4f}%",
        "strategy": args.strategy,
    }
    engine.portfolio.analyzer.register_statistic(RunConfig(**run_params))

    strategy = StrategyClass(config=strategy_config)
    engine.add_strategy(strategy)

    print("Running backtest...")
    engine.run()

    from .report import print_report

    strat_label = args.strategy.replace("_", " ").title()
    print_report(
        engine,
        venue,
        title=f"{strat_label} — {ex} {sym} {interval_ccxt}",
        pair=sym,
    )
