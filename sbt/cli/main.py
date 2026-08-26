"""Top-level CLI entry point for SBT.

Usage::

    sbt backtest --config config.toml --strategy overnight_drift
    sbt data --exchange binance --symbol BTC/USDT --interval 5m --start 2024-01-01
    sbt web --port 8080
    sbt optimize --config config.toml --strategy overnight_drift --trials 50
"""

from __future__ import annotations

import argparse
import sys


def _cmd_backtest(args: argparse.Namespace) -> None:
    from ..core.config import RunConfig
    from ..core.runner import BacktestRunner
    from ..report import print_report
    from pathlib import Path

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
    strategy_params = {}
    if args.param:
        for p in args.param:
            k, v = p.split("=", 1)
            strategy_params[k] = v

    cfg = RunConfig.from_toml(
        args.config, args.strategy, cli_overrides=cli_overrides,
        strategy_params=strategy_params or None,
    )
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
    base_title = f"{strat_label} — {cfg.exchange} {cfg.symbol} {cfg.interval}"

    if runner.window_engines:
        from ..plugins.train_val_split import _WINDOW_LABELS

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

    run_id = runner.engine.run_id
    reports = Path("reports")
    latest = reports / "latest.html"
    target = f"tearsheet_{run_id}.html"
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(target)
    except OSError:
        pass


def _cmd_data(args: argparse.Namespace) -> None:
    from ..data import main as data_main

    sys.argv = ["sbt", "data"] + sys.argv[2:]
    data_main()


def _cmd_web(args: argparse.Namespace) -> None:
    import uvicorn

    from ..web.app import create_app

    app = create_app(db_path=args.db, reports_dir=args.reports)
    print(f"Starting SBT Results dashboard at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _cmd_optimize(args: argparse.Namespace) -> None:
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
    strategy_params = {}
    if args.param:
        for p in args.param:
            k, v = p.split("=", 1)
            strategy_params[k] = v

    run_optuna_study(
        config_path=args.config,
        strategy_name=args.strategy,
        n_trials=args.trials,
        params=[],
        db_path=args.db,
        output_report=args.report,
        objective=args.objective,
        overrides={**cli_overrides, **strategy_params} if strategy_params else cli_overrides,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbt",
        description="SBT — Strategy Backtesting Tool",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # backtest
    from .args import add_backtest_args, add_data_args, add_web_args

    p_backtest = subparsers.add_parser("backtest", help="Run a strategy backtest")
    add_backtest_args(p_backtest)
    p_backtest.set_defaults(func=_cmd_backtest)

    # data
    p_data = subparsers.add_parser("data", help="Download market data via ccxt")
    add_data_args(p_data)
    p_data.set_defaults(func=_cmd_data)

    # web
    p_web = subparsers.add_parser("web", help="Launch the results web dashboard")
    add_web_args(p_web)
    p_web.set_defaults(func=_cmd_web)

    # optimize
    p_opt = subparsers.add_parser(
        "optimize", help="Run Optuna multi-objective hyperparameter optimization"
    )
    add_backtest_args(p_opt)
    p_opt.add_argument(
        "--trials", type=int, default=30, help="Number of trials (default: 30)"
    )
    p_opt.add_argument(
        "--objective",
        choices=["sharpe", "sqn"],
        default="sharpe",
        help="Optimization objective (default: sharpe)",
    )
    p_opt.add_argument(
        "--db", default="sbt.db", help="SQLite database path (default: sbt.db)"
    )
    p_opt.add_argument(
        "--report",
        default=None,
        help="Output HTML report path (default: auto)",
    )
    p_opt.set_defaults(func=_cmd_optimize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
