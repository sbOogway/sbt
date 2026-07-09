# SBT — Strategy Backtesting Tool

A modular backtesting framework built on [nautilus-trader](https://nautilustrader.io) and [ccxt](https://ccxt.readthedocs.io/), focused on cryptocurrency perpetual futures strategies.

## Quick Start

```bash
uv run python3 -m sbt --config config.toml --strategy overnight_drift
```

Fetch data:

```bash
uv run python3 -m sbt.data --exchange hyperliquid --symbol XYZ-SP500/USDC:USDC --interval 1h --start 2026-03-18 --type ohlcv
uv run python3 -m sbt.data --exchange hyperliquid --symbol XYZ-SP500/USDC:USDC --start 2026-03-18 --type funding
```

## Structure

```
sbt/
├── __main__.py          CLI entry point
├── data.py              Data downloader (ccxt → feather)
├── report.py            HTML tearsheet + TradingView chart
├── stats.py             Custom portfolio statistics
├── utils.py             Strategy loader, instrument factory
├── volatility.py        Moreira & Muir rolling vol scaling
└── strategies/
    ├── overnight_drift.py        Overnight hold (NYSE close → EU open)
    ├── bitcoin_intraday_momentum.py  Intraday momentum (BTC)
    ├── glucksmann.py             Trend/volatility (Glucksmann thesis)
    └── orb.py                    Opening Range Breakout
data/               .feather files (OHLCV + funding)
config.toml         Run + strategy parameters
reports/            Generated HTML tearsheets
```

## Key Conventions

- **Times are UTC** throughout config and strategy code
- **Bars are 1h** by default; bar timestamps at UTC hour boundaries
- **USDC settlement** for Hyperliquid perps
- **Perpetual instrument factory**: `make_perpetual()` in `sbt/utils.py` with configurable currencies, fees, and precision (`price_precision=1`, `size_precision=3`)
- **Slippage defined in ticks** (1 tick = $0.1 for SP500), converted to equivalent bps via `tick_size / ref_price` and added to `taker_fee`
- **Funding rates tracked as a side-channel** in strategy metadata; does not flow through engine PnL
- **Fees** are set in config.toml `[run]`: `taker_fee` + slippage equivalent

## Adding a New Strategy

1. Create `sbt/strategies/<name>.py` with two classes:
   - `<Name>Config(StrategyConfig, frozen=True)` — typed config fields
   - `<Name>(Strategy)` — strategy logic
2. Register in `sbt/utils.py` `get_strategy_class()` function
3. Add `[strategy.<name>]` section in `config.toml`
4. Run: `uv run python3 -m sbt --strategy <name>`

## Important Notes for AI Agents

- **Never commit unless the user explicitly asks you to**
- The `--strategy` flag selects the config section in `config.toml`
- `capital` config is initial; strategy uses `risk_percent` of current equity for sizing (compounding)
- `FillModel(prob_slippage=1.0)` always applies 1 tick of nautilus-level slippage when `slippage_ticks > 0`
- Volatility scaling is rolling (daily), not monthly: weight updates every time `add_return()` is called
- Taker fee displayed in RunConfig is the raw fraction (e.g., `0.0002%`), not multiplied by 100
- Backtest data is loaded from `.feather` files; auto-detected by exchange/symbol/interval pattern
- Tearsheets are saved to `reports/` and auto-opened in browser

## Dependencies

- Python >=3.14
- `nautilus-trader[visualization]` — backtest engine
- `ccxt` — data fetching
- Managed via `uv`
