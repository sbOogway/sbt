"""CLI commands for interacting with the SBT Scheduler and running optimization."""

import argparse
import sys
import time
from pathlib import Path

from ..core.config import RunConfig
from ..utils import get_strategy_names
from .client import SbtClient


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format tabular data into an aligned ASCII table."""
    if not rows:
        return "(no data)"
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    mid = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    header_line = (
        "│ " + " │ ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " │"
    )
    data_lines = [
        "│ "
        + " │ ".join(f"{val!s:<{col_widths[i]}}" for i, val in enumerate(row))
        + " │"
        for row in rows
    ]

    return "\n".join([sep, header_line, mid] + data_lines + [bot])


def cmd_submit(args: argparse.Namespace) -> None:
    client = SbtClient(endpoint=f"tcp://127.0.0.1:{args.port}")

    if not client.ping():
        print(f"ERROR: Cannot connect to SBT Scheduler at tcp://127.0.0.1:{args.port}.")
        print("Please start the server first: uv run python3 -m sbt.server")
        sys.exit(1)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: Config file not found: {cfg_path}")
        sys.exit(1)

    overrides = {
        "exchange": args.exchange,
        "symbol": args.symbol,
        "interval": args.interval,
        "leverage": args.leverage,
        "start": args.start,
        "feather": args.feather,
    }

    if args.all_strategies:
        strategies = get_strategy_names()
        if not strategies:
            print("No strategies registered in sbt.utils._STRATEGY_REGISTRY.")
            sys.exit(1)
        configs = [
            RunConfig.from_toml(args.config, strat, overrides) for strat in strategies
        ]
        job_ids = client.submit_batch(configs)
        print(f"Submitted {len(job_ids)} jobs:")
        for strat, jid in zip(strategies, job_ids):
            print(f"  - {strat:<25} -> Job ID: {jid}")

        if args.wait:
            print("\nWaiting for all jobs to complete...")
            _wait_for_jobs(client, job_ids)

    else:
        strat = args.strategy
        config = RunConfig.from_toml(args.config, strat, overrides)
        job_id = client.submit(config)
        print(f"Submitted job for '{strat}' -> Job ID: {job_id}")

        if args.wait:
            print(f"Waiting for job {job_id}...")
            _wait_for_jobs(client, [job_id])


def _wait_for_jobs(client: SbtClient, job_ids: list[str]) -> None:
    pending = set(job_ids)
    results = {}
    while pending:
        time.sleep(1.0)
        for jid in list(pending):
            resp = client.get_result(jid)
            if resp.get("status") == "ok":
                results[jid] = resp.get("result", {})
                pending.remove(jid)
            elif resp.get("status") == "not_found":
                pending.remove(jid)

    print("\n========== ALL JOBS COMPLETED ==========")
    headers = ["Job ID", "Status", "Sharpe", "Trades", "PnL", "Duration"]
    rows = []
    for jid in job_ids:
        res = results.get(jid, {})
        sharpe = (
            f"{res.get('sharpe_ratio'):.2f}"
            if res.get("sharpe_ratio") is not None
            else "N/A"
        )
        trades = str(res.get("num_trades", "N/A"))
        pnl = f"${res.get('pnl', 0):+,.2f}" if res.get("pnl") is not None else "N/A"
        dur = f"{res.get('duration_seconds', 0):.1f}s"
        status = res.get("status", "unknown")
        rows.append([jid, status, sharpe, trades, pnl, dur])

    print(_format_table(headers, rows))


def cmd_status(args: argparse.Namespace) -> None:
    client = SbtClient(endpoint=f"tcp://127.0.0.1:{args.port}")
    if not client.ping():
        print(f"ERROR: Cannot connect to SBT Scheduler at tcp://127.0.0.1:{args.port}.")
        sys.exit(1)

    status = client.get_status()
    print(
        f"Workers: {status.get('workers_idle')}/{status.get('workers_total')} Idle | Busy: {status.get('workers_busy')} | Queue: {status.get('queue_length')}"
    )
    print()

    jobs = status.get("jobs", [])
    if not jobs:
        print("No jobs found in result store.")
        return

    headers = ["Job ID", "Strategy", "Status", "Worker", "Submitted (UTC)"]
    rows = []
    for j in jobs[: args.limit]:
        rows.append(
            [
                j.get("id"),
                j.get("config", {}).get("strategy_name", "N/A"),
                j.get("status"),
                j.get("worker_id") or "—",
                j.get("submitted_at", "")[:19],
            ]
        )
    print(_format_table(headers, rows))


def cmd_results(args: argparse.Namespace) -> None:
    client = SbtClient(endpoint=f"tcp://127.0.0.1:{args.port}")
    if not client.ping():
        print(f"ERROR: Cannot connect to SBT Scheduler at tcp://127.0.0.1:{args.port}.")
        sys.exit(1)

    if args.job:
        resp = client.get_result(args.job)
        if resp.get("status") != "ok":
            print(f"Job {args.job} status: {resp.get('job_status', 'not found')}")
            return
        res = resp["result"]
        print(f"\n--- Result for Job {args.job} ---")
        print(f"Status:       {res.get('status')}")
        print(f"Sharpe Ratio: {res.get('sharpe_ratio')}")
        print(f"Total Trades: {res.get('num_trades')}")
        print(f"Net PnL:      ${res.get('pnl', 0):+,.2f}")
        print(f"Duration:     {res.get('duration_seconds', 0):.2f}s")
        if res.get("error"):
            print(f"Error:        {res.get('error')}")

        stats = res.get("stats", {})
        if stats:
            print("\n--- Full Statistics ---")
            for k, v in stats.items():
                print(f"  {k:<35}: {v}")
    else:
        results = client.list_results(study_name=args.study)
        if not results:
            print("No completed results found.")
            return
        headers = ["Job ID", "Status", "Sharpe", "Trades", "PnL", "Duration"]
        rows = []
        for res in results:
            sharpe = (
                f"{res.get('sharpe_ratio'):.2f}"
                if res.get("sharpe_ratio") is not None
                else "N/A"
            )
            trades = str(res.get("num_trades", "N/A"))
            pnl = f"${res.get('pnl', 0):+,.2f}" if res.get("pnl") is not None else "N/A"
            dur = f"{res.get('duration_seconds', 0):.1f}s"
            rows.append(
                [res.get("job_id"), res.get("status"), sharpe, trades, pnl, dur]
            )
        print(_format_table(headers, rows))


def cmd_compare(args: argparse.Namespace) -> None:
    from ..compare.dashboard import generate_comparison_dashboard

    client = SbtClient(endpoint=f"tcp://127.0.0.1:{args.port}")
    if not client.ping():
        print(f"ERROR: Cannot connect to SBT Scheduler at tcp://127.0.0.1:{args.port}.")
        sys.exit(1)

    if args.jobs:
        job_ids = [j.strip() for j in args.jobs.split(",")]
        results = []
        for jid in job_ids:
            resp = client.get_result(jid)
            if resp.get("status") == "ok":
                results.append(resp["result"])
            else:
                print(f"Warning: Result for job {jid} not found or incomplete.")
    else:
        results = client.list_results(study_name=args.study)

    if not results:
        print("No results to compare.")
        return

    output_path = generate_comparison_dashboard(results, output_path=args.output)
    print(f"Comparison dashboard generated: {output_path}")


def cmd_optimize(args: argparse.Namespace) -> None:
    from ..optimize.study import run_optuna_study

    run_optuna_study(
        config_path=args.config,
        strategy_name=args.strategy,
        n_trials=args.trials,
        params=args.param or [],
        db_path=args.db,
        port=args.port,
        output_report=args.report,
        objective=args.objective,
        overrides={
            "exchange": args.exchange,
            "symbol": args.symbol,
            "interval": args.interval,
            "leverage": args.leverage,
            "start": args.start,
        },
    )
