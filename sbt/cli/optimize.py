"""`sbt optimize` subcommand."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``optimize`` subcommand to *subparsers*."""
    from .args import add_backtest_args

    p = subparsers.add_parser(
        "optimize", help="Run Optuna multi-objective hyperparameter optimization"
    )
    add_backtest_args(p)
    p.add_argument(
        "--trials", type=int, default=30, help="Number of trials (default: 30)"
    )
    p.add_argument(
        "--objective",
        choices=["sharpe", "sqn"],
        default="sharpe",
        help="Optimization objective (default: sharpe)",
    )
    p.add_argument(
        "--db", default="sbt.db", help="SQLite database path (default: sbt.db)"
    )
    p.add_argument(
        "--report",
        default=None,
        help="Output HTML report path (default: auto)",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Run an Optuna optimization study from parsed CLI args."""
    from ..core.config import RunConfig
    from ..optimize.study import run_optuna_study

    cli_overrides = {
        "exchange": args.exchange,
        "symbol": args.symbol,
        "interval": args.interval,
        "leverage": args.leverage,
        "start": args.start,
        "end": args.end,
        "feather": args.feather,
        "warmup_bars": args.warmup_bars,
        "data_type": args.data_type,
        "l2_max_files": args.l2_max_files,
        "train_val_split": args.train_val_split,
    }

    if args.no_open:
        cli_overrides["open_report"] = False

    run_optuna_study(
        config_path=args.config,
        strategy_name=args.strategy,
        n_trials=args.trials,
        params=args.param or [],
        db_path=args.db,
        output_report=args.report,
        objective=args.objective,
        overrides=cli_overrides,
    )
