import argparse
import glob
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd
from nautilus_trader.analysis import create_tearsheet
from nautilus_trader.analysis.statistic import PortfolioStatistic
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy


# ---------------------------------------------------------
# Strategy Configuration
# ---------------------------------------------------------
class BitcoinIntradayMomentumConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    capital: Decimal
    leverage: float = 1.0

    backtest_start_date: str = "2020-01-01"

    # Time constants formatted in EST (Eastern Standard Time)
    # The paper defines open as "volume spikes" (~8:30am EST when US econ news released),
    # so first half-hour ends at ~9:00am EST (30 min after volume spikes).
    onfh_close_time: str = "09:30"
    slh_open_time: str = "16:00"
    slh_close_time: str = "16:30"

    # Volatility scaling (Moreira & Muir 2017)
    vol_scaling: bool = True
    rv_lookback: int = 22
    max_leverage: float = 2.0


# ---------------------------------------------------------
# Strategy Logic
# ---------------------------------------------------------
class BitcoinIntradayMomentum(Strategy):
    def __init__(self, config: BitcoinIntradayMomentumConfig) -> None:
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        # State tracking for daily intervals
        self.prev_close: Optional[Decimal] = None
        self.onfh_close: Optional[Decimal] = None
        self.slh_open: Optional[Decimal] = None

        self.r_onfh: Optional[float] = None
        self.r_slh: Optional[float] = None

        self.current_position_side: Optional[OrderSide] = None
        self._open_qty: Optional[Quantity] = None

        # Volatility scaling state (Moreira & Muir)
        self._daily_returns: list[float] = []
        self._rv_history: list[float] = []
        self._current_weight: float = 1.0

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        dt_utc = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
        dt_est = dt_utc.tz_convert("US/Eastern")
        time_str = dt_est.strftime("%H:%M")

        if time_str == self.config.onfh_close_time:
            self.onfh_close = Decimal(bar.close.as_double())
            if self.prev_close:
                self.r_onfh = float(self.onfh_close / self.prev_close) - 1.0

        if time_str == self.config.slh_open_time:
            self.slh_open = Decimal(bar.close.as_double())

        if time_str == self.config.slh_close_time:
            if self.slh_open:
                slh_close = Decimal(bar.close.as_double())
                self.r_slh = float(slh_close / self.slh_open) - 1.0
            self.evaluate_signal_and_trade(Decimal(bar.close.as_double()))

        if time_str == "17:00":
            self.close_positions()
            close_val = Decimal(bar.close.as_double())
            if self.config.vol_scaling:
                if self.prev_close is not None:
                    daily_ret = float(close_val / self.prev_close) - 1.0
                    self._daily_returns.append(daily_ret)
            self.prev_close = close_val

            if self.config.vol_scaling:
                dt_today = dt_est.date()
                dt_tomorrow = dt_today + pd.Timedelta(days=1)
                if dt_tomorrow.month != dt_today.month and len(self._daily_returns) >= self.config.rv_lookback:
                    self._rebalance()

    def evaluate_signal_and_trade(self, price: Decimal) -> None:
        if self.r_onfh is None or self.r_slh is None:
            return  # Wait until both intervals are safely captured for the day

        if self.r_onfh <= 0 and self.r_slh >= 0:
            self._open_trade(OrderSide.SELL, price)
        elif self.r_onfh > 0 and self.r_slh < 0:
            self._open_trade(OrderSide.BUY, price)
        # else: no trade

    def close_positions(self) -> None:
        if self.current_position_side == OrderSide.BUY:
            self._close_trade(OrderSide.SELL)
        elif self.current_position_side == OrderSide.SELL:
            self._close_trade(OrderSide.BUY)

        self.current_position_side = None
        self._open_qty = None

    def _open_trade(self, order_side: OrderSide, price: Decimal) -> None:
        notional = self.config.capital * Decimal(self.config.leverage) * Decimal(self._current_weight)
        raw_size = notional / price
        self._open_qty = Quantity(round(float(raw_size), 3), precision=3)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=self._open_qty,
        )
        self.submit_order(order)
        self.current_position_side = order_side

    def _close_trade(self, order_side: OrderSide) -> None:
        if self._open_qty is None:
            return
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=self._open_qty,
        )
        self.submit_order(order)

    def _rebalance(self) -> None:
        rv = sum(r * r for r in self._daily_returns[-self.config.rv_lookback:])
        self._rv_history.append(rv)
        c = sum(self._rv_history) / len(self._rv_history)
        self._current_weight = min(self.config.max_leverage, c / rv) if rv > 0 else self.config.max_leverage


# ---------------------------------------------------------
# Custom Portfolio Statistic: Calmar Ratio
# ---------------------------------------------------------


class CalmarRatio(PortfolioStatistic):
    def calculate_from_returns(self, returns: pd.Series) -> float | None:
        if not self._check_valid_returns(returns):
            return None
        daily = self._downsample_to_daily_bins(returns)
        if len(daily) < 2:
            return None
        ann_return = daily.mean() * 252
        cum = (1 + daily).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = dd.min()
        if max_dd >= 0:
            return None
        return float(ann_return / abs(max_dd))


class RunConfig(PortfolioStatistic):
    def __init__(self, **kwargs: str) -> None:
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "Run Config"

    def calculate_from_positions(self, positions: list) -> str:
        return " | ".join(f"{k}={v}" for k, v in self._kwargs.items())


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def make_perpetual(
    venue_name: str,
    symbol_str: str,
    maker_fee: Decimal = Decimal("0.0"),
    taker_fee: Decimal = Decimal("0.0"),
) -> CryptoPerpetual:
    raw = symbol_str.replace("/", "")
    inst_id = InstrumentId(symbol=Symbol(f"{raw}-PERP"), venue=Venue(venue_name))
    return CryptoPerpetual(
        instrument_id=inst_id,
        raw_symbol=Symbol(raw),
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=1,
        price_increment=Price.from_str("0.1"),
        size_precision=3,
        size_increment=Quantity.from_str("0.001"),
        max_quantity=Quantity.from_str("1000.000"),
        min_quantity=Quantity.from_str("0.001"),
        max_notional=None,
        min_notional=Money(10.00, USDT),
        max_price=Price.from_str("999999.0"),
        min_price=Price.from_str("0.1"),
        margin_init=Decimal("0.0500"),
        margin_maint=Decimal("0.0250"),
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        ts_event=0,
        ts_init=0,
    )


_INTERVAL_MAP = {
    "1m": "1-MINUTE",
    "3m": "3-MINUTE",
    "5m": "5-MINUTE",
    "15m": "15-MINUTE",
    "30m": "30-MINUTE",
    "1h": "1-HOUR",
    "2h": "2-HOUR",
    "4h": "4-HOUR",
    "6h": "6-HOUR",
    "8h": "8-HOUR",
    "12h": "12-HOUR",
    "1d": "1-DAY",
    "1w": "1-WEEK",
}


def parse_interval(interval: str) -> str:
    result = _INTERVAL_MAP.get(interval)
    if result is None:
        raise ValueError(f"Unknown interval: {interval}")
    return result


# ---------------------------------------------------------
# Backtest Initialization Engine & Data Loading
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BTC intraday momentum backtest")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to TOML config file (default: config.toml)",
    )
    parser.add_argument(
        "--feather", help="Path to feather file (auto-detect if omitted)"
    )
    parser.add_argument(
        "--exchange", help="Venue/exchange name (default from config.toml)"
    )
    parser.add_argument(
        "--symbol", help="Trading pair (default from config.toml)"
    )
    parser.add_argument(
        "--interval", help="Candle interval (default from config.toml)"
    )
    parser.add_argument("--leverage", help="Leverage (default from config.toml)")
    parser.add_argument(
        "--start",
        help="Backtest start date (default from config.toml)",
    )
    args = parser.parse_args()

    # Load TOML config (CLI args override TOML values)
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            cfg = tomllib.load(f)
    else:
        cfg = {"run": {}, "strategy": {}}

    ex = args.exchange or cfg.get("run", {}).get("exchange", "BINANCE")
    sym = args.symbol or cfg.get("run", {}).get("symbol", "BTC/USDT")
    interval_ccxt = args.interval or cfg.get("run", {}).get("interval", "5m")
    capital = Decimal(str(cfg.get("run", {}).get("capital", "1000")))
    leverage_val = float(args.leverage or cfg.get("run", {}).get("leverage", 1.0))
    start = args.start or cfg.get("run", {}).get("start", "2020-01-01")
    maker_fee = Decimal(str(cfg.get("run", {}).get("maker_fee", "0.0")))
    taker_fee = Decimal(str(cfg.get("run", {}).get("taker_fee", "0.0")))
    strat_cfg = cfg.get("strategy", {})

    venue = Venue(ex)
    raw_symbol = sym.replace("/", "")
    interval_nt = parse_interval(interval_ccxt)
    leverage = Decimal(str(leverage_val))

    # 1. Initialize Engine
    engine = BacktestEngine(config=BacktestEngineConfig())

    # 2. Add Venue and Account Structure
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(capital, USDT)],
        default_leverage=leverage,
    )

    # 3. Setup Instrument
    instrument = make_perpetual(ex, sym, maker_fee, taker_fee)
    engine.add_instrument(instrument)

    # 4. Define the BarType
    bar_type = BarType.from_str(f"{instrument.id.value}-{interval_nt}-LAST-EXTERNAL")

    # 5. Create Strategy Configuration
    strategy_config = BitcoinIntradayMomentumConfig(
        instrument_id=instrument.id,
        bar_type=bar_type,
        capital=capital,
        leverage=leverage_val,
        backtest_start_date=start,
        onfh_close_time=strat_cfg.get("onfh_close_time", "09:30"),
        slh_open_time=strat_cfg.get("slh_open_time", "16:00"),
        slh_close_time=strat_cfg.get("slh_close_time", "16:30"),
        vol_scaling=strat_cfg.get("vol_scaling", True),
        rv_lookback=strat_cfg.get("rv_lookback", 22),
        max_leverage=strat_cfg.get("max_leverage", 2.0),
    )

    # 6. Data Ingestion Block
    feather_path = args.feather
    if not feather_path:
        search_dirs = ["data", "."]
        for d in search_dirs:
            pattern = (
                f"{d}/{ex.lower()}_{raw_symbol}_{interval_ccxt}_*.feather"
            )
            files = sorted(glob.glob(pattern))
            if files:
                feather_path = files[-1]
                break
            pattern = f"{d}/{raw_symbol}_{interval_ccxt}_*.feather"
            files = sorted(glob.glob(pattern))
            if files:
                feather_path = files[-1]
                break
        if not feather_path:
            print(f"ERROR: No feather files found matching download_data.py naming.")
            print(
                f"       Either pass --feather or ensure a file matching '{raw_symbol}_{interval_ccxt}_*.feather' exists in data/."
            )
            exit(1)

    print(f"Loading data from {feather_path}...")
    try:
        df = pd.read_feather(feather_path)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        start_ts = pd.Timestamp(strategy_config.backtest_start_date, tz="UTC")
        df = df[df["timestamp"] >= start_ts]
        df = df.reset_index(drop=True)

        bars_list = []
        for row in df.itertuples(index=False):
            ts_nanos = dt_to_unix_nanos(row.timestamp)
            bar = Bar(
                bar_type=bar_type,
                open=Price(row.open, precision=1),
                high=Price(row.high, precision=1),
                low=Price(row.low, precision=1),
                close=Price(row.close, precision=1),
                volume=Quantity(row.volume, precision=3),
                ts_event=ts_nanos,
                ts_init=ts_nanos,
            )
            bars_list.append(bar)

        engine.add_data(bars_list)
        print(
            f"Successfully loaded {len(bars_list)} {interval_ccxt} bars into the engine."
        )

    except FileNotFoundError:
        print(f"ERROR: Feather file '{feather_path}' not found.")
        exit(1)

    # 7. Register custom statistics
    engine.portfolio.analyzer.register_statistic(CalmarRatio())
    engine.portfolio.analyzer.register_statistic(
        RunConfig(
            pair=sym,
            exchange=ex,
            interval=interval_ccxt,
            capital=f"${capital}",
            leverage=f"{leverage_val}x",
            maker_fee=f"{float(instrument.maker_fee) * 100:.4f}%",
            taker_fee=f"{float(instrument.taker_fee) * 100:.4f}%",
            onfh_close_time=strategy_config.onfh_close_time,
            vol_scaling=str(strategy_config.vol_scaling),
            rv_lookback=str(strategy_config.rv_lookback),
            max_leverage=f"{strategy_config.max_leverage}x",
        )
    )

    # 8. Attach the Momentum Strategy
    strategy = BitcoinIntradayMomentum(config=strategy_config)
    engine.add_strategy(strategy)

    # 9. Run Execution
    print("Running backtest...")
    engine.run()

    # 9. Display the final report
    print("\n========== BACKTEST COMPLETE ==========")

    # ------------------------------------------------------------------
    # 9a. Portfolio Performance Statistics
    # ------------------------------------------------------------------
    stats_pnls = engine.portfolio.analyzer.get_performance_stats_pnls()
    stats_returns = engine.portfolio.analyzer.get_performance_stats_returns()
    stats_general = engine.portfolio.analyzer.get_performance_stats_general()

    print("\n--- Portfolio Performance ---")
    for k, v in {**stats_pnls, **stats_returns, **stats_general}.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 9b. Positions Report
    # ------------------------------------------------------------------
    positions_report = engine.trader.generate_positions_report()
    print(f"\n--- Positions Report ({len(positions_report)} rows) ---")
    print(positions_report.to_string(max_rows=20))

    # ------------------------------------------------------------------
    # 9c. Fills Report
    # ------------------------------------------------------------------
    fills_report = engine.trader.generate_fills_report()
    print(f"\n--- Fills Report ({len(fills_report)} rows) ---")
    print(fills_report.to_string(max_rows=20))

    # ------------------------------------------------------------------
    # 9d. Orders Report
    # ------------------------------------------------------------------
    orders_report = engine.trader.generate_orders_report()
    print(f"\n--- Orders Report ({len(orders_report)} rows) ---")
    print(orders_report.to_string(max_rows=20))

    # ------------------------------------------------------------------
    # 9e. Account Report
    # ------------------------------------------------------------------
    account_report = engine.trader.generate_account_report(venue)
    print(f"\n--- Account Report ({len(account_report)} rows) ---")
    print(account_report.to_string(max_rows=10))

    # ------------------------------------------------------------------
    # 9f. Interactive Tearsheet (HTML)
    # ------------------------------------------------------------------
    run_id = engine.run_id
    tearsheet_path = f"reports/tearsheet_{run_id}.html"
    print(f"\n--- Generating tearsheet ({run_id}) ---")
    create_tearsheet(
        engine,
        output_path=tearsheet_path,
        title=f"BTC Intraday Momentum — {ex} {sym} {interval_ccxt}",
    )
    print(f"Tearsheet saved to {tearsheet_path}")

    print("\n========== DONE ==========")
