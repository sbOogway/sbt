import pandas as pd
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, OmsType, AccountType
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.currencies import USD, USDT
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.analysis import ReportProvider, create_tearsheet

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
    onfh_close_time: str = "08:30"
    slh_open_time: str = "16:00"
    slh_close_time: str = "16:30"

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

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        dt_utc = pd.Timestamp(bar.ts_event, unit='ns', tz='UTC')
        dt_est = dt_utc.tz_convert('US/Eastern')
        time_str = dt_est.strftime('%H:%M')
        
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
            self.prev_close = Decimal(bar.close.as_double())

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
        notional = self.config.capital * Decimal(self.config.leverage)
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

# ---------------------------------------------------------
# Backtest Initialization Engine & Data Loading
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Initialize Engine
    engine = BacktestEngine(config=BacktestEngineConfig())
    
    # 2. Add Venue and Account Structure
    venue = Venue("BINANCE")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(1_000, USDT)],
    )
    
    # 3. Setup Instrument (Mock representation of BTC/USD)
    BTCUSD = TestInstrumentProvider.btcusdt_perp_binance()
    engine.add_instrument(BTCUSD)
    
    # 4. Define the BarType (needed for both strategy config and data loading)
    bar_type = BarType.from_str(f"{BTCUSD.id.value}-5-MINUTE-LAST-EXTERNAL")
    
    # 5. Create Strategy Configuration
    strategy_config = BitcoinIntradayMomentumConfig(
        instrument_id=BTCUSD.id,
        bar_type=bar_type,
        capital=Decimal("1000"),
        leverage=1.0,
    )
    
    # 6. Data Ingestion Block
    # Loads a Binance 5-minute feather file created by download_binance.py
    import glob

    feather_files = sorted(glob.glob("BTCUSDT_5m_*.feather"))
    if not feather_files:
        print("ERROR: No BTCUSDT_5m_*.feather file found. Run download_binance.py first.")
        exit(1)

    feather_path = feather_files[-1]
    print(f"Loading data from {feather_path}...")
    try:
        df = pd.read_feather(feather_path)
        # Keep only columns needed by the backtest
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        
        # Filter data from the configured start date to exclude low-quality early data
        start_ts = pd.Timestamp(strategy_config.backtest_start_date, tz='UTC')
        df = df[df['timestamp'] >= start_ts]
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
            
        # Push the parsed list of Bars into the Backtest Engine
        engine.add_data(bars_list)
        print(f"Successfully loaded {len(bars_list)} 5-minute bars into the engine.")
        
    except FileNotFoundError:
        print(f"ERROR: Feather file '{feather_path}' not found.")
        exit(1)

    # 7. Attach the Momentum Strategy
    strategy = BitcoinIntradayMomentum(config=strategy_config)
    engine.add_strategy(strategy)
    
    # 8. Run Execution
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
    account_report = engine.trader.generate_account_report(Venue("BINANCE"))
    print(f"\n--- Account Report ({len(account_report)} rows) ---")
    print(account_report.to_string(max_rows=10))
    
    # ------------------------------------------------------------------
    # 9f. Interactive Tearsheet (HTML)
    # ------------------------------------------------------------------
    print("\n--- Generating tearsheet ---")
    create_tearsheet(engine, output_path="tearsheet.html")
    print("Tearsheet saved to tearsheet.html")
    
    print("\n========== DONE ==========")
