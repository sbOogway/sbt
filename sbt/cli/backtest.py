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
    from ..core.runner import BacktestRunner
    from ..report import print_report

    cli_overrides = {
        "exchange": args.exchange,
        "symbol": args.symbol,
        "symbols": _flatten_symbols(args.symbols),
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
