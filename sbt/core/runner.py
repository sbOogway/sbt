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
import time
from decimal import Decimal

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import Currency, USDC, USDT
from nautilus_trader.model.data import Bar, BarType, FundingRateUpdate
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Price, Quantity

from ..stats import AnnualizedReturn, CalmarRatio, RunConfig as RunConfigStat
from ..utils import get_strategy_class, make_perpetual, parse_interval
from .config import RunConfig
from .job import BacktestResult, JobStatus


_CURRENCY_MAP = {
    "USDT": USDT,
    "USDC": USDC,
}


# ------------------------------------------------------------------
# Data helpers (moved from __main__)
# ------------------------------------------------------------------

def load_bars(df: pd.DataFrame, bar_type: BarType) -> list[Bar]:
    """Convert an OHLCV DataFrame into a list of Nautilus Bar objects."""
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


def find_feather(
    exchange: str,
    symbol: str,
    interval: str,
    search_dirs: list[str] | None = None,
) -> str | None:
    """Discover a feather data file by convention.

    Searches *search_dirs* (defaulting to ``["data", "."]``) for files
    matching ``{exchange}_{symbol}_{interval}_*.feather``.
    """
    if search_dirs is None:
        search_dirs = ["data", "."]

    raw_symbol = symbol.replace("/", "")
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

    def run(self, job_id: str = "standalone") -> BacktestResult:
        """Execute the backtest and return a structured result."""
        t0 = time.monotonic()
        cfg = self.config

        # -- Resolve data path -----------------------------------------
        feather_path = cfg.feather_path or find_feather(
            cfg.exchange, cfg.symbol, cfg.interval,
            search_dirs=[cfg.data_dir, "."],
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
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            start_ts = pd.Timestamp(cfg.start, tz="UTC")
            df = df[df["timestamp"] >= start_ts].reset_index(drop=True)
        except FileNotFoundError:
            return BacktestResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                error=f"Feather file '{feather_path}' not found.",
            )

        # -- Slippage as fee bps ---------------------------------------
        ref_price = float(df["close"].iloc[0])
        slippage_bps = cfg.slippage_ticks * cfg.tick_size / ref_price * 10000
        taker_fee = cfg.taker_fee + Decimal(str(slippage_bps)) / Decimal(10000)

        # -- Resolve currencies ----------------------------------------
        settle_currency = _CURRENCY_MAP.get(cfg.settle_currency)
        if settle_currency is None:
            settle_currency = Currency(cfg.settle_currency, 2, 0, cfg.settle_currency, 0)

        # -- Engine setup ----------------------------------------------
        interval_nt = parse_interval(cfg.interval)
        leverage_dec = Decimal(str(cfg.leverage))
        venue = Venue(cfg.exchange)
        self.venue = venue

        engine = BacktestEngine(config=BacktestEngineConfig())
        self.engine = engine

        fill_model = FillModel(prob_slippage=1.0) if cfg.slippage_ticks > 0 else None

        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=settle_currency,
            starting_balances=[Money(cfg.capital, settle_currency)],
            default_leverage=leverage_dec,
            fill_model=fill_model,
        )

        # -- Instrument ------------------------------------------------
        base_code = cfg.symbol.split("/")[0]
        base_currency = _CURRENCY_MAP.get(
            base_code, Currency(base_code, 2, 0, base_code, 0),
        )
        instrument = make_perpetual(
            cfg.exchange, cfg.symbol, cfg.maker_fee, taker_fee,
            base_currency=base_currency,
            settlement_currency=settle_currency,
            quote_currency=settle_currency,
        )
        engine.add_instrument(instrument)

        bar_type = BarType.from_str(f"{instrument.id.value}-{interval_nt}-LAST-EXTERNAL")

        # -- Strategy --------------------------------------------------
        StrategyClass, ConfigClass = get_strategy_class(cfg.strategy_name)
        strategy_config = ConfigClass(
            instrument_id=instrument.id,
            bar_type=bar_type,
            capital=cfg.capital,
            leverage=cfg.leverage,
            backtest_start_date=cfg.start,
            **cfg.strategy_params,
        )

        # -- Load bars -------------------------------------------------
        print(f"Loaded {len(df)} {cfg.interval} bars (ref_price={ref_price}).")
        bars = load_bars(df, bar_type)
        engine.add_data(bars)

        # -- Load funding rates ----------------------------------------
        funding_path = find_feather(
            cfg.exchange, cfg.symbol, "funding",
            search_dirs=[cfg.data_dir, "."],
        )
        if funding_path:
            print(f"Loading funding data from {funding_path}...")
            df_funding = pd.read_feather(funding_path)
            start_ts = pd.Timestamp(cfg.start, tz="UTC")
            df_funding = df_funding[df_funding["timestamp"] >= start_ts].reset_index(drop=True)
            funding_updates = load_funding_rates(df_funding, instrument.id)
            engine.add_data(funding_updates)
            print(f"Loaded {len(funding_updates)} funding rate updates.")
        else:
            print("No funding rate data found (file pattern: *funding*). Running without funding.")

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
        engine.portfolio.analyzer.register_statistic(RunConfigStat(**run_params))

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

        # Extract optimisation objectives
        pnl = stats_pnls.get("PnL (total)")
        sharpe = stats_returns.get("Sharpe Ratio (252 days)")
        num_trades = len(positions_df)

        # Funding side-channel
        funding_pnl = 0.0
        if hasattr(strategy, "_trade_funding_costs") and strategy._trade_funding_costs:
            funding_pnl = float(sum(strategy._trade_funding_costs))

        elapsed = time.monotonic() - t0

        return BacktestResult(
            job_id=job_id,
            status=JobStatus.DONE,
            sharpe_ratio=float(sharpe) if sharpe is not None else None,
            num_trades=num_trades,
            pnl=float(pnl) if pnl is not None else None,
            stats=all_stats,
            equity_curve=[],  # populated by report layer if needed
            positions=positions_df.to_dict("records") if len(positions_df) else [],
            fills=fills_df.to_dict("records") if len(fills_df) else [],
            funding_pnl=funding_pnl,
            duration_seconds=elapsed,
        )
