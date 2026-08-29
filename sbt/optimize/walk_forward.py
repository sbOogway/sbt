"""Walk-forward validation: rolling IS/OOS windows with per-window optimization.

Design:
- Rolling in-sample (IS) and out-of-sample (OOS) windows
- Optuna TPE trials per IS window (Sharpe / Trades / PnL Pareto objective)
- Executed via LocalExecutor spawn-multiprocessing for memory isolation
- Aggregated metrics: mean OOS Sharpe, consistency (# profitable / total), worst OOS Sharpe
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd

from ..core.config import RunConfig
from ..core.job import BacktestResult, JobStatus
from ..core.runner import BacktestRunner, _slice_frame
from .param_parser import (
    DEFAULT_PARAM_SPACES,
    get_default_param_space,
    parse_param_spec,
    suggest_params,
)
from .study import LocalExecutor

# Backward-compatible alias
PARAM_SPACES = DEFAULT_PARAM_SPACES


# ------------------------------------------------------------------
# Window generation
# ------------------------------------------------------------------


def _generate_windows(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    is_months: int,
    oos_months: int,
    step_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yield (is_start, is_end, oos_start, oos_end) for each walk-forward window.

    IS window is ``is_months`` long, followed by OOS of ``oos_months``.
    Windows advance by ``step_months``. The last OOS window may be shorter
    if it hits the data boundary.
    """
    windows = []
    cursor = data_start

    while True:
        is_start = cursor
        is_end = is_start + pd.DateOffset(months=is_months) - pd.Timedelta(days=1)
        oos_start = is_end + pd.Timedelta(days=1)
        oos_end_candidate = oos_start + pd.DateOffset(months=oos_months) - pd.Timedelta(days=1)

        if oos_start > data_end:
            break

        oos_end = min(oos_end_candidate, data_end)
        windows.append((is_start, is_end, oos_start, oos_end))

        if oos_end_candidate >= data_end:
            break

        cursor += pd.DateOffset(months=step_months)

    return windows


# ------------------------------------------------------------------
# Window execution via LocalExecutor seam
# ------------------------------------------------------------------


def _optimize_is(
    cfg: RunConfig,
    param_space: list[str],
    bars_is: pd.DataFrame,
    n_trials: int,
    job_id: str,
) -> dict | None:
    """Run Optuna optimization on the IS slice; return best params dict or None on failure."""
    space = parse_param_spec(param_space)

    bars_start = str(bars_is["timestamp"].iloc[0])
    bars_end = str(bars_is["timestamp"].iloc[-1])
    cfg_is = dataclasses.replace(
        cfg,
        start=bars_start,
        end=bars_end,
        open_report=False,
    )

    executor = LocalExecutor(
        base_config=cfg_is,
        objective="sharpe",
        db_path=None,
        bars=bars_is,
    )

    study = optuna.create_study(
        study_name=f"wf_{job_id}",
        directions=["maximize", "maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(),
    )

    def _objective(trial: optuna.Trial):
        params = suggest_params(trial, space)
        trial_cfg = cfg_is.with_overrides(params)
        trial_job_id = f"{job_id}_t{trial.number}"
        # Trial #0 runs inline (primes caches); subsequent trials use spawn isolation
        metrics = executor.run_single_trial(trial_cfg, trial_job_id, forked=(trial.number > 0))

        if metrics.get("status") != "done":
            return (float("-inf"), 0.0, float("-inf"))

        sharpe = metrics.get("sharpe_ratio") or 0.0
        trades = float(metrics.get("trades") or 0)
        pnl = metrics.get("pnl") or 0.0
        if sharpe != sharpe:  # NaN check
            return (float("-inf"), 0.0, float("-inf"))
        return (sharpe, trades, pnl)

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return None

    best = max(completed, key=lambda t: t.values[0])
    return best.params


def _run_oos(
    cfg: RunConfig,
    params: dict,
    bars_oos: pd.DataFrame,
    job_id: str,
) -> BacktestResult:
    """Run one backtest on the OOS slice with *params*."""
    bars_start = str(bars_oos["timestamp"].iloc[0])
    bars_end = str(bars_oos["timestamp"].iloc[-1])
    cfg_oos = dataclasses.replace(
        cfg.with_overrides(params),
        start=bars_start,
        end=bars_end,
        open_report=False,
    )

    executor = LocalExecutor(
        base_config=cfg_oos,
        objective="sharpe",
        db_path=None,
        bars=bars_oos,
    )

    metrics = executor.run_single_trial(cfg_oos, job_id, forked=True)
    status = JobStatus.DONE if metrics.get("status") == "done" else JobStatus.FAILED
    return BacktestResult(
        job_id=job_id,
        status=status,
        sharpe_ratio=metrics.get("sharpe_ratio"),
        num_trades=metrics.get("trades", 0),
        pnl=metrics.get("pnl"),
        error=metrics.get("error"),
    )


# ------------------------------------------------------------------
# Walk-forward result aggregation
# ------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward output."""

    strategy: str
    windows: list[dict]
    # Aggregated OOS metrics
    mean_oos_sharpe: float | None = None
    mean_oos_return_pct: float | None = None
    consistency: float | None = None  # fraction of profitable windows
    worst_oos_sharpe: float | None = None
    total_oos_trades: int = 0
    total_windows: int = 0
    profitable_windows: int = 0

    def summary_line(self) -> str:
        parts = [
            f"Strategy: {self.strategy}",
            f"Windows: {self.total_windows}",
            f"Mean OOS Sharpe: {self.mean_oos_sharpe:+.3f}" if self.mean_oos_sharpe is not None else "Mean OOS Sharpe: N/A",
            f"Consistency: {self.consistency:.0%}" if self.consistency is not None else "Consistency: N/A",
            f"Worst OOS Sharpe: {self.worst_oos_sharpe:+.3f}" if self.worst_oos_sharpe is not None else "Worst OOS Sharpe: N/A",
            f"Mean OOS Return: {self.mean_oos_return_pct:+.1f}%" if self.mean_oos_return_pct is not None else "Mean OOS Return: N/A",
            f"Total OOS Trades: {self.total_oos_trades}",
        ]
        return " | ".join(parts)


# ------------------------------------------------------------------
# Main walk-forward loop
# ------------------------------------------------------------------


def run_walk_forward(
    cfg: RunConfig,
    is_months: int = 12,
    oos_months: int = 3,
    step_months: int = 3,
    trials: int = 30,
    param_space: list[str] | None = None,
    bars: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Execute walk-forward validation for one strategy.

    Returns a WalkForwardResult with per-window and aggregated metrics.
    """
    strategy = cfg.strategy_name

    if param_space is None:
        param_space = get_default_param_space(strategy)
        if param_space is None:
            raise ValueError(
                f"No param space registered for '{strategy}'. "
                f"Available: {sorted(DEFAULT_PARAM_SPACES)}. "
                f"Pass --param specs manually."
            )

    # --- Load full data range ---
    from ..core.feather import to_utc_ts
    from ..core.runner import _discover_bars

    if bars is not None:
        full_df = bars
    else:
        full_df, err = _discover_bars(cfg, to_utc_ts(cfg.start), to_utc_ts(cfg.end))
        if err:
            raise RuntimeError(err)

    data_start = full_df["timestamp"].iloc[0]
    data_end = full_df["timestamp"].iloc[-1]
    ts_col = full_df["timestamp"]

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD: {strategy}")
    print(f"Data: {data_start.date()} → {data_end.date()} ({len(full_df)} bars)")
    print(f"Windows: {is_months}mo IS / {oos_months}mo OOS / {step_months}mo step")
    print(f"Optuna trials per window: {trials}")
    print(f"Parameter space: {[s.split('=')[0] for s in param_space]}")
    print(f"{'='*60}", flush=True)

    # --- Generate windows ---
    wf_windows = _generate_windows(data_start, data_end, is_months, oos_months, step_months)
    if not wf_windows:
        raise ValueError(
            f"Data range {data_start.date()}→{data_end.date()} too short "
            f"for {is_months}mo IS + {oos_months}mo OOS windows."
        )

    print(f"Generated {len(wf_windows)} windows.", flush=True)

    # --- Walk-forward loop ---
    window_results = []

    for i, (is_start, is_end, oos_start, oos_end) in enumerate(wf_windows):
        label = f"W{i+1:02d}"
        print(f"\n--- {label}: IS {is_start.date()}→{is_end.date()} | "
              f"OOS {oos_start.date()}→{oos_end.date()} ---", flush=True)

        # Slice data
        bars_is = full_df[(ts_col >= is_start) & (ts_col <= is_end)].reset_index(drop=True)
        bars_oos = full_df[(ts_col >= oos_start) & (ts_col <= oos_end)].reset_index(drop=True)

        if len(bars_is) < 10 or len(bars_oos) < 2:
            print(f"  SKIP: not enough bars (IS={len(bars_is)}, OOS={len(bars_oos)})", flush=True)
            window_results.append({
                "label": label,
                "is_range": f"{is_start.date()}→{is_end.date()}",
                "oos_range": f"{oos_start.date()}→{oos_end.date()}",
                "status": "skipped",
                "reason": f"insufficient bars (IS={len(bars_is)}, OOS={len(bars_oos)})",
            })
            continue

        # --- Optimize on IS ---
        t_is = time.monotonic()
        best_params = _optimize_is(cfg, param_space, bars_is, trials, label)
        is_duration = time.monotonic() - t_is

        if best_params is None:
            print(f"  IS optimization failed (no completed trials) ({is_duration:.1f}s)", flush=True)
            window_results.append({
                "label": label,
                "is_range": f"{is_start.date()}→{is_end.date()}",
                "oos_range": f"{oos_start.date()}→{oos_end.date()}",
                "status": "is_failed",
                "is_duration_s": is_duration,
            })
            continue

        print(f"  Best IS params: {best_params} ({is_duration:.1f}s)", flush=True)

        # --- Run OOS with best params ---
        t_oos = time.monotonic()
        oos_result = _run_oos(cfg, best_params, bars_oos, f"{label}_oos")
        oos_duration = time.monotonic() - t_oos

        wr = {
            "label": label,
            "is_range": f"{is_start.date()}→{is_end.date()}",
            "oos_range": f"{oos_start.date()}→{oos_end.date()}",
            "status": oos_result.status.value,
            "best_params": best_params,
            "is_duration_s": is_duration,
            "oos_sharpe": oos_result.sharpe_ratio,
            "oos_pnl": oos_result.pnl,
            "oos_trades": oos_result.num_trades,
            "oos_return_pct": None,
            "oos_duration_s": oos_duration,
        }

        if oos_result.status == JobStatus.DONE and oos_result.pnl is not None:
            wr["oos_return_pct"] = oos_result.pnl / float(cfg.capital) * 100
            print(
                f"  OOS: Sharpe={oos_result.sharpe_ratio:+.3f} | "
                f"PnL=${oos_result.pnl:+,.2f} | "
                f"Trades={oos_result.num_trades} ({oos_duration:.1f}s)",
                flush=True,
            )
        else:
            print(f"  OOS: FAILED ({oos_result.error})", flush=True)

        window_results.append(wr)

    # --- Aggregate ---
    result = WalkForwardResult(strategy=strategy, windows=window_results)

    oos_sharpes = [w["oos_sharpe"] for w in window_results
                   if w.get("oos_sharpe") is not None and w["status"] == "done"]
    oos_returns = [w["oos_return_pct"] for w in window_results
                   if w.get("oos_return_pct") is not None and w["status"] == "done"]
    oos_trades = [w["oos_trades"] for w in window_results
                  if w.get("oos_trades") is not None and w["status"] == "done"]

    result.total_windows = len(window_results)
    result.total_oos_trades = sum(oos_trades)

    if oos_sharpes:
        result.mean_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes)
        result.worst_oos_sharpe = min(oos_sharpes)
    if oos_returns:
        result.mean_oos_return_pct = sum(oos_returns) / len(oos_returns)

    completed = [w for w in window_results if w["status"] == "done"]
    result.profitable_windows = sum(
        1 for w in completed if (w.get("oos_pnl") or 0) > 0
    )
    if completed:
        result.consistency = result.profitable_windows / len(completed)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD SUMMARY: {strategy}")
    print(f"{'='*60}")
    print(f"  Windows completed: {len(completed)}/{result.total_windows}")
    print(f"  Mean OOS Sharpe:   {result.mean_oos_sharpe:+.3f}" if result.mean_oos_sharpe is not None else "  Mean OOS Sharpe:   N/A")
    print(f"  Worst OOS Sharpe:  {result.worst_oos_sharpe:+.3f}" if result.worst_oos_sharpe is not None else "  Worst OOS Sharpe:  N/A")
    print(f"  Mean OOS Return:   {result.mean_oos_return_pct:+.1f}%" if result.mean_oos_return_pct is not None else "  Mean OOS Return:   N/A")
    print(f"  Consistency:       {result.consistency:.0%}" if result.consistency is not None else "  Consistency:       N/A")
    print(f"  Total OOS Trades:  {result.total_oos_trades}")
    print(f"{'='*60}", flush=True)

    return result


# ------------------------------------------------------------------
# CLI glue
# ------------------------------------------------------------------


def run_walk_forward_from_args(args) -> "WalkForwardResult":
    """Build a RunConfig from *args* and run walk-forward validation.

    The single shared entry point for the ``sbt backtest --walk-forward``
    and ``sbt optimize --walk-forward`` subcommands. Expects *args* to
    carry the four ``--wf-*`` knobs (``wf_is_months``, ``wf_oos_months``,
    ``wf_step_months``, ``wf_trials``) and a ``param`` list — the same
    shape produced by :func:`sbt.cli.args.add_backtest_args`.

    Raises ``ValueError`` from the underlying :meth:`RunConfig.from_cli_args`
    on missing required args (CLI modules translate to a user-facing
    error and ``sys.exit(1)``). Prints the one-line ``summary_line()``
    after the multi-line block already printed by :func:`run_walk_forward`.
    """
    from ..core.config import RunConfig

    cfg = RunConfig.from_cli_args(args)
    result = run_walk_forward(
        cfg,
        is_months=args.wf_is_months,
        oos_months=args.wf_oos_months,
        step_months=args.wf_step_months,
        trials=args.wf_trials,
        param_space=args.param,
    )
    print(f"\n{result.summary_line()}")
    return result
