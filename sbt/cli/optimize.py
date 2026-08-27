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
        "walk_forward": args.walk_forward,
        "wf_is_months": args.wf_is_months,
        "wf_oos_months": args.wf_oos_months,
        "wf_step_months": args.wf_step_months,
        "wf_trials": args.wf_trials,
    }

    if args.no_open:
        cli_overrides["open_report"] = False

    # --- Walk-forward mode ---
    if args.walk_forward:
        from ..core.config import RunConfig
        from ..optimize.walk_forward import run_walk_forward

        cfg = RunConfig.from_toml(
            args.config, args.strategy, cli_overrides=cli_overrides
        )
        if args.param:
            from ..core.config import _parse_scalar

            overrides = {}
            for spec in args.param:
                name, _, raw = spec.partition("=")
                if not name or not raw:
                    raise ValueError(f"Invalid --param '{spec}': expected NAME=VALUE")
                overrides[name.strip()] = _parse_scalar(raw)
            cfg = cfg.with_overrides(overrides)

        wf_result = run_walk_forward(cfg)
        print(f"\n{wf_result.summary_line()}")
        return

    # --- Standard Optuna optimization ---
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
