import argparse
import glob
import tomllib
from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import Currency, USDC, USDT
from nautilus_trader.model.data import Bar, BarType, FundingRateUpdate
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Price, Quantity

from nautilus_trader.backtest.models import FillModel

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


def load_funding_rates(df: pd.DataFrame, instrument_id) -> list[FundingRateUpdate]:
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
    settle_code = run.get("settle_currency", "USDT")
    slippage_ticks = int(run.get("slippage_ticks", 0))

    _CURRENCY_MAP = {
        "USDT": USDT,
        "USDC": USDC,
    }
    settle_currency = _CURRENCY_MAP.get(settle_code)
    if settle_currency is None:
        settle_currency = Currency(settle_code, 2, 0, settle_code, 0)

    interval_nt = parse_interval(interval_ccxt)
    leverage_dec = Decimal(str(leverage_val))

    feather_path = args.feather or find_feather(ex, sym, interval_ccxt)
    if not feather_path:
        print(f"ERROR: No feather data found for {sym} ({interval_ccxt})")
        exit(1)

    print(f"Loading data from {feather_path}...")
    try:
        df = pd.read_feather(feather_path)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        start_ts = pd.Timestamp(start, tz="UTC")
        df = df[df["timestamp"] >= start_ts].reset_index(drop=True)
    except FileNotFoundError:
        print(f"ERROR: Feather file '{feather_path}' not found.")
        exit(1)

    ref_price = float(df["close"].iloc[0])
    tick_size = 0.1
    slippage_bps = slippage_ticks * tick_size / ref_price * 10000

    taker_fee += Decimal(str(slippage_bps)) / Decimal(10000)

    venue = Venue(ex)
    engine = BacktestEngine(config=BacktestEngineConfig())

    fill_model = FillModel(prob_slippage=1.0) if slippage_ticks > 0 else None

    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=settle_currency,
        starting_balances=[Money(capital, settle_currency)],
        default_leverage=leverage_dec,
        fill_model=fill_model,
    )

    base_code = sym.split("/")[0]
    base_currency = _CURRENCY_MAP.get(base_code, Currency(base_code, 2, 0, base_code, 0))
    instrument = make_perpetual(ex, sym, maker_fee, taker_fee,
                                base_currency=base_currency,
                                settlement_currency=settle_currency,
                                quote_currency=settle_currency)
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

    print(f"Loaded {len(df)} {interval_ccxt} bars (ref_price={ref_price}).")
    bars = load_bars(df, bar_type)
    engine.add_data(bars)

    funding_path = find_feather(ex, sym, "funding")
    if funding_path:
        print(f"Loading funding data from {funding_path}...")
        df_funding = pd.read_feather(funding_path)
        start_ts = pd.Timestamp(strategy_config.backtest_start_date, tz="UTC")
        df_funding = df_funding[df_funding["timestamp"] >= start_ts].reset_index(drop=True)
        funding_updates = load_funding_rates(df_funding, instrument.id)
        engine.add_data(funding_updates)
        print(f"Loaded {len(funding_updates)} funding rate updates.")
    else:
        print("No funding rate data found (file pattern: *funding*). Running without funding.")

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
        "slippage_ticks": f"{slippage_ticks}",
        "strategy": args.strategy,
    }
    engine.portfolio.analyzer.register_statistic(RunConfig(**run_params))

    strategy = StrategyClass(config=strategy_config)
    engine.add_strategy(strategy)

    print("Running backtest...")
    engine.run()

    total_funding_cost = Decimal("0")
    if hasattr(strategy, '_trade_funding_costs') and strategy._trade_funding_costs:
        total_funding_cost = sum(strategy._trade_funding_costs)
    if total_funding_cost != 0:
        print(f"\n--- Funding Summary ---")
        print(f"  Total funding PnL: {float(total_funding_cost):+.2f} USDC")
        print(f"  (Negative = strategy paid, Positive = strategy received)")



    from .report import print_report

    strat_label = args.strategy.replace("_", " ").title()
    print_report(
        engine,
        venue,
        title=f"{strat_label} — {ex} {sym} {interval_ccxt}",
        pair=sym,
    )
