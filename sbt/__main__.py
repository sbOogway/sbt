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
        print(f"\n--- Funding Summary ---")
        print(f"  Total funding PnL: {result.funding_pnl:+.2f} {cfg.settle_currency}")
        print(f"  (Negative = strategy paid, Positive = strategy received)")

    # Report generation (uses engine/venue retained by the runner)
    from .report import print_report

    strat_label = cfg.strategy_name.replace("_", " ").title()
    print_report(
        runner.engine,
        runner.venue,
        title=f"{strat_label} — {cfg.exchange} {cfg.symbol} {cfg.interval}",
        pair=cfg.symbol,
    )
