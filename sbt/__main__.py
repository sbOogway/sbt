"""CLI entry point — preserved for backward compatibility.

Usage::

    uv run python3 -m sbt --config config.toml --strategy overnight_drift
"""

from .core.config import RunConfig
from .core.runner import BacktestRunner

if __name__ == "__main__":
    cfg = RunConfig.parse_cli()
    runner = BacktestRunner(cfg)
    result = runner.run()

    if result.error:
        print(f"ERROR: {result.error}")
        exit(1)

    # Funding summary
    if result.funding_pnl != 0:
        print("\n--- Funding Summary ---")
        print(f"  Total funding PnL: {result.funding_pnl:+.2f} {cfg.settle_currency}")
        print("  (Negative = strategy paid, Positive = strategy received)")

    # Report generation (uses engine/venue retained by the runner).
    # With a train/val split, one tearsheet per window is generated.
    from .report import print_report

    strat_label = cfg.strategy_name.replace("_", " ").title()
    base_title = f"{strat_label} — {cfg.exchange} {cfg.symbol} {cfg.interval}"

    if result.splits and runner.window_engines:
        window_titles = {
            "in_sample": f"{base_title} [In-Sample]",
            "out_of_sample": f"{base_title} [Out-of-Sample]",
        }
        for key, engine in runner.window_engines.items():
            if key in result.splits:
                print_report(
                    engine,
                    runner.venue,
                    title=window_titles.get(key, f"{base_title} [{key}]"),
                    pair=cfg.symbol,
                )
    else:
        print_report(runner.engine, runner.venue, title=base_title, pair=cfg.symbol)
