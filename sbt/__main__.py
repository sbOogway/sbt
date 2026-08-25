"""CLI entry point — preserved for backward compatibility.

Usage::

    uv run python3 -m sbt --config config.toml --strategy overnight_drift
"""

from .core.config import RunConfig
from .core.runner import BacktestRunner

if __name__ == "__main__":
    cfg = RunConfig.parse_cli()
    runner = BacktestRunner(cfg, db_path="sbt.db")
    result = runner.run()

    if result.error:
        print(f"ERROR: {result.error}")
        exit(1)

    # Funding summary
    if result.funding_pnl != 0:
        print("\n--- Funding Summary ---")
        print(f"  Total funding PnL: {result.funding_pnl:+.2f} {cfg.settle_currency}")
        print("  (Positive = strategy paid, Negative = strategy received)")

    # Report generation (uses engine/venue retained by the runner).
    # With a train/val split, one tearsheet per window is generated;
    # window titles come from the runner plugin constants.
    from .report import print_report

    strat_label = cfg.strategy_name.replace("_", " ").title()
    base_title = f"{strat_label} — {cfg.exchange} {cfg.symbol} {cfg.interval}"

    if runner.window_engines:
        from .plugins.train_val_split import _WINDOW_LABELS

        for key, engine in runner.window_engines.items():
            label = _WINDOW_LABELS.get(key, key)
            print_report(
                engine,
                runner.venue,
                title=f"{base_title} [{label}]",
                pair=cfg.symbol,
                exchange=cfg.exchange,
                interval=cfg.interval,
                open_browser=cfg.open_report,
            )
    else:
        print_report(
            runner.engine,
            runner.venue,
            title=base_title,
            pair=cfg.symbol,
            exchange=cfg.exchange,
            interval=cfg.interval,
            open_browser=cfg.open_report,
        )
