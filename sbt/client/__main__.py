"""Client CLI entry point.

Usage::

    uv run python3 -m sbt.client submit --config config.toml --strategy overnight_drift
    uv run python3 -m sbt.client submit --config config.toml --all-strategies --wait
    uv run python3 -m sbt.client status
    uv run python3 -m sbt.client results --job <id>
    uv run python3 -m sbt.client compare --jobs <id1,id2,...>
    uv run python3 -m sbt.client optimize --config config.toml --strategy overnight_drift --trials 20 --param "rv_lookback=int(3,30)"
"""

import argparse

from .cli import cmd_compare, cmd_optimize, cmd_results, cmd_status, cmd_submit


def main():
    parser = argparse.ArgumentParser(description="SBT Client CLI")
    parser.add_argument(
        "--port", type=int, default=5555, help="Scheduler port (default: 5555)"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # submit
    sub_submit = subparsers.add_parser("submit", help="Submit backtest job(s)")
    sub_submit.add_argument(
        "--config", default="config.toml", help="Path to config.toml"
    )
    sub_submit.add_argument(
        "--strategy", default="bitcoin_intraday_momentum", help="Strategy to run"
    )
    sub_submit.add_argument(
        "--all-strategies", action="store_true", help="Submit all strategies in config"
    )
    sub_submit.add_argument(
        "--wait", action="store_true", help="Wait for job completion"
    )
    sub_submit.add_argument("--exchange", help="Override exchange")
    sub_submit.add_argument("--symbol", help="Override trading pair")
    sub_submit.add_argument("--interval", help="Override interval")
    sub_submit.add_argument("--leverage", help="Override leverage")
    sub_submit.add_argument("--start", help="Override start date")
    sub_submit.add_argument("--feather", help="Override feather path")
    sub_submit.add_argument(
        "--train-val-split",
        type=float,
        metavar="FRACTION",
        help="Holdout split: in-sample fraction (e.g. 0.7)",
    )
    sub_submit.set_defaults(func=cmd_submit)

    # status
    sub_status = subparsers.add_parser(
        "status", help="Check scheduler and workers status"
    )
    sub_status.add_argument("--limit", type=int, default=20, help="Max jobs to display")
    sub_status.set_defaults(func=cmd_status)

    # results
    sub_results = subparsers.add_parser("results", help="View backtest results")
    sub_results.add_argument("--job", help="Specific job ID to inspect")
    sub_results.add_argument("--study", help="Filter by Optuna study name")
    sub_results.set_defaults(func=cmd_results)

    # compare
    sub_compare = subparsers.add_parser(
        "compare", help="Compare multiple backtest runs"
    )
    sub_compare.add_argument("--jobs", help="Comma-separated job IDs to compare")
    sub_compare.add_argument(
        "--study", help="Study name to compare all completed jobs from"
    )
    sub_compare.add_argument(
        "--output", default="reports/compare.html", help="Output comparison HTML path"
    )
    sub_compare.set_defaults(func=cmd_compare)

    # optimize
    sub_opt = subparsers.add_parser(
        "optimize", help="Run Optuna multi-objective hyperparameter optimization"
    )
    sub_opt.add_argument("--config", default="config.toml", help="Path to config.toml")
    sub_opt.add_argument("--strategy", required=True, help="Strategy to optimize")
    sub_opt.add_argument(
        "--trials", type=int, default=30, help="Number of trials (default: 30)"
    )
    sub_opt.add_argument(
        "--param",
        action="append",
        help="Parameter range: 'name=int(min,max)' or 'name=float(min,max)' or 'name=cat(v1,v2)'",
    )
    sub_opt.add_argument("--db", default="sbt.db", help="SQLite database path")
    sub_opt.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated report in a browser",
    )
    sub_opt.add_argument(
        "--objective",
        choices=["sharpe", "sqn"],
        default="sharpe",
        help="Optimization objective: 'sharpe' = (Sharpe, Trades, PnL) Pareto front, 'sqn' = pure Van Tharp System Quality Number (default: sharpe)",
    )
    sub_opt.add_argument(
        "--report",
        default=None,
        help="Output HTML report path (default: reports/pareto_report.html or reports/sqn_report.html)",
    )
    sub_opt.add_argument("--exchange", help="Override exchange")
    sub_opt.add_argument("--symbol", help="Override trading pair")
    sub_opt.add_argument("--interval", help="Override interval")
    sub_opt.add_argument("--leverage", help="Override leverage")
    sub_opt.add_argument("--start", help="Override start date")
    sub_opt.add_argument("--end", help="Override end date")
    sub_opt.add_argument(
        "--local",
        action="store_true",
        help="Force in-process trials even if a scheduler is reachable",
    )
    sub_opt.set_defaults(func=cmd_optimize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
