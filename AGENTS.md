# SBT — Strategy Backtesting Tool

Backtesting framework on [nautilus-trader](https://nautilustrader.io) + [ccxt](https://ccxt.readthedocs.io/) for crypto perpetual futures.

## Commands

```bash
uv run python3 -m sbt --config config.toml --strategy overnight_drift

uv run python3 -m sbt.data --exchange hyperliquid --symbol XYZ-SP500/USDC:USDC --interval 1h --start 2026-03-18 --type ohlcv
uv run python3 -m sbt.data --exchange hyperliquid --symbol XYZ-SP500/USDC:USDC --start 2026-03-18 --type funding
```

- `--strategy` defaults to `bitcoin_intraday_momentum` if omitted.
- `--type funding` ignores `--interval` (no interval param).
- No tests, linting, typechecking, or CI exist. No verification step to run.

## Key Conventions

- **Times UTC** throughout; bars at UTC hour boundaries.
- **USDC settlement** for Hyperliquid perps.
- **Instrument factory**: `make_perpetual()` in `sbt/utils.py` (`price_precision=1`, `size_precision=3`).
- **Slippage in ticks**: `slippage_ticks * tick_size / ref_price * 10000` bps added to `taker_fee`.
- **Funding rates** tracked as metadata side-channel; does not flow through engine PnL.
- **Vol scaling** is rolling (daily), not monthly — `add_return()` updates weight each call.
- **Position sizing**: `risk_percent * current_equity * leverage * vol_weight` (compounding).
- **FillModel**: `FillModel(prob_slippage=1.0)` when `slippage_ticks > 0` (1 tick nautilus slippage).
- **Data files** (`.feather`) auto-detected by `{exchange}_{symbol}_{interval}_*.feather` pattern in `data/` or `./`. Funding files matched by `*funding*` in path.
- **Tearsheets** saved to `reports/` and auto-opened via `webbrowser`.
- `data/` and `reports/` are gitignored (empty on fresh checkout).

## Adding a Strategy

1. Create `sbt/strategies/<name>.py` with `<Name>Config(StrategyConfig, frozen=True)` and `<Name>(Strategy)`.
2. Register in `sbt/utils.py` `get_strategy_class()`.
3. Add `[strategy.<name>]` section in `config.toml`.
4. Run: `uv run python3 -m sbt --strategy <name>`.

## Structure

```
sbt/
├── __main__.py             CLI entry point
├── data.py                 Data downloader (ccxt → feather)
├── report.py               HTML tearsheet + TradingView chart
├── stats.py                Custom portfolio statistics
├── utils.py                Strategy loader, instrument factory
├── volatility.py           Moreira & Muir rolling vol scaling
└── strategies/
    ├── overnight_drift.py
    ├── bitcoin_intraday_momentum.py
    ├── glucksmann.py
    └── orb.py
papers/                     Reference PDFs (not code)
data/                       .feather files (gitignored)
reports/                    Generated HTML tearsheets (gitignored)
config.toml                 Run + strategy parameters
```

## Stale Artifacts

- `sbt/strategies/orb.py.bak` — backup file, ignore it.

## Dependencies

- Python >=3.14 (`.python-version`)
- `nautilus-trader[visualization]`, `ccxt` — managed via `uv`
