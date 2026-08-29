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


def run(args: argparse.Namespace) -> None:
    """Execute a backtest from parsed CLI args."""
    from ..core.config import RunConfig
    from ..core.feather import infer_instrument_from_path
    from ..core.runner import BacktestRunner
    from ..report import print_report

    # Resolve exchange/symbol/interval: CLI > --feather inference.
    exchange = args.exchange
    symbol = args.symbol
    interval = args.interval
    if args.feather:
        inferred = infer_instrument_from_path(args.feather)
        if inferred is None:
            print(
                f"ERROR: Could not infer exchange/symbol/interval from "
                f"--feather path {args.feather!r}. Pass them explicitly."
            )
            sys.exit(1)
        if not exchange:
            exchange = inferred[0]
        if not interval:
            interval = inferred[2]
        if not symbol and not _flatten_symbols(args.symbols):
            symbol = inferred[1]
    missing = []
    if not exchange:
        missing.append("--exchange")
    if not symbol and not _flatten_symbols(args.symbols):
        missing.append("--symbol or --symbols")
    if not interval:
        missing.append("--interval")
    if missing:
        print(
            f"ERROR: Missing required CLI args: {', '.join(missing)}. "
            f"Either pass them explicitly or supply --feather PATH "
            f"to infer them from the filename."
        )
        sys.exit(1)

    cli_overrides = {
        "exchange": exchange,
        "symbol": symbol,
        "symbols": _flatten_symbols(args.symbols),
        "interval": interval,
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

    cfg = RunConfig.from_toml(args.config, args.strategy, cli_overrides=cli_overrides)

    if args.param:
        overrides = {}
        for spec in args.param:
            name, _, raw = spec.partition("=")
            if not name or not raw:
                print(f"ERROR: Invalid --param '{spec}': expected NAME=VALUE")
                sys.exit(1)
            overrides[name.strip()] = _parse_scalar(raw)
        cfg = cfg.with_overrides(overrides)

    # --- Walk-forward mode ---
    if cfg.walk_forward:
        from ..optimize.walk_forward import run_walk_forward

        wf_result = run_walk_forward(cfg)
        print(f"\n{wf_result.summary_line()}")
        return

    # --- Standard backtest mode ---
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


def _flatten_symbols(values) -> list[str] | None:
    """Flatten appended ``--symbols`` specs into a single list of symbols."""
    if not values:
        return None
    out = []
    for spec in values:
        for part in spec.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _parse_scalar(raw: str):
    """Infer the type of a CLI scalar: int -> float -> bool -> str."""
    text = raw.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    return text
