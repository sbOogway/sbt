"""`sbt backtest` subcommand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``backtest`` subcommand to *subparsers*."""
    from .args import add_backtest_args

    p = subparsers.add_parser("backtest", help="Run a strategy backtest")
    add_backtest_args(p)
    p.set_defaults(func=run)


def _check_feather(args: argparse.Namespace) -> None:
    """Translate an unparseable --feather path into a CLI-friendly exit.

    Called before either the walk-forward or the standard path so the
    user gets a consistent error message regardless of which sub-flow
    the inference failure would have hit.
    """
    if args.feather and infer_instrument_from_path(args.feather) is None:
        print(
            f"ERROR: Could not infer exchange/symbol/interval from "
            f"--feather path {args.feather!r}. Pass them explicitly."
        )
        sys.exit(1)


def run(args: argparse.Namespace) -> None:
    """Execute a backtest from parsed CLI args."""
    from ..core.config import RunConfig
    from ..core.runner import BacktestRunner
    from ..report import print_report

    # --- Walk-forward mode ---
    # The walk-forward helper builds its own RunConfig, so we don't
    # build one here — but the --feather pre-flight check is shared.
    if args.walk_forward:
        _check_feather(args)
        try:
            from ..optimize.walk_forward import run_walk_forward_from_args

            run_walk_forward_from_args(args)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        return

    # --- Standard backtest mode ---
    _check_feather(args)
    try:
        cfg = RunConfig.from_cli_args(args)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    runner = BacktestRunner(cfg, db_path="sbt.db")
    result = runner.run()

    if result.error:
        print(f"ERROR: {result.error}")
        sys.exit(1)

    if result.funding_pnl != 0:
        print("\n--- Funding Summary ---")
        print(f"  Total funding PnL: {result.funding_pnl:+.2f} {cfg.settle_currency}")
        print("  (Positive = strategy paid, Negative = strategy received)")

    strat_label = cfg.strategy_name.replace("_", " ").title()
    if len(cfg.all_symbols) > 1:
        base_title = (
            f"{strat_label} — {cfg.exchange} {cfg.interval} "
            f"[{len(cfg.all_symbols)}-symbol portfolio]"
        )
    else:
        base_title = f"{strat_label} — {cfg.exchange} {cfg.symbol} {cfg.interval}"

    if runner.window_engines:
        from ..plugins.train_val_split import _WINDOW_LABELS

        for key, engine in runner.window_engines.items():
            label = _WINDOW_LABELS.get(key, key)
            print_report(
                engine,
                runner.venue,
                title=f"{base_title} [{label}]",
                open_browser=cfg.open_report,
            )
    else:
        print_report(
            runner.engine,
            runner.venue,
            title=base_title,
            open_browser=cfg.open_report,
        )

    run_id = runner.engine.run_id
    reports = Path("reports")
    latest = reports / "latest.html"
    target = f"tearsheet_{run_id}.html"
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(target)
    except OSError:
        pass
