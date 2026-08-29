"""Orchestrates Optuna optimization studies.

Two objective modes:
- 'sharpe': 3-objective Pareto front (Sharpe, Trades, PnL) — the default.
- 'sqn': single-objective maximization of Van Tharp's System Quality Number.

Execution runs inline via LocalExecutor.
"""

import datetime
import gc
import multiprocessing
from pathlib import Path

import optuna

import pandas as pd
from ..core.config import RunConfig
from ..core.job import BacktestResult
from ..core.runner import BacktestRunner
from .param_parser import (
    get_default_param_space,
    parse_param_spec,
    suggest_params,
)
from .report import generate_pareto_report, generate_sqn_report


def _spawn_child_target(
    config: RunConfig,
    job_id: str,
    objective: str,
    db_path: str | None,
    queue,
    bars: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
):
    """Top-level target for spawn-multiprocessing backtest trials."""
    try:
        runner = BacktestRunner(config, db_path=db_path)
        try:
            result = runner.run(job_id=job_id, bars=bars)
        finally:
            del runner
            gc.collect()
        queue.put(
            {
                "status": result.status.value,
                "pnl": result.pnl,
                "trades": result.num_trades,
                "sqn": result.sqn if objective == "sqn" else None,
                "sharpe_ratio": (
                    result.sharpe_ratio if objective == "sharpe" else None
                ),
                "error": result.error,
            }
        )
    except BaseException as e:  # noqa: BLE001
        queue.put({"status": "failed", "error": f"{type(e).__name__}: {e}"})


class LocalExecutor:
    """Run trials with process isolation to avoid Nautilus engine memory accumulation.

    Trial #0 runs inline (priming caches); subsequent trials execute in a spawned child process.
    """

    name = "local"

    def __init__(
        self,
        base_config: RunConfig,
        objective: str = "sharpe",
        primary_label: str = "Sharpe Ratio",
        db_path: str | None = "sbt.db",
        bars: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    ):
        self.base_config = base_config
        self.objective = objective
        self.primary_label = primary_label
        self.db_path = db_path
        self.bars = bars
        self._mp_ctx = multiprocessing.get_context("spawn")

    @staticmethod
    def _metrics(result: BacktestResult, objective: str) -> dict:
        return {
            "status": result.status.value,
            "pnl": result.pnl,
            "trades": result.num_trades,
            "sqn": result.sqn if objective == "sqn" else None,
            "sharpe_ratio": (result.sharpe_ratio if objective == "sharpe" else None),
            "error": result.error,
        }

    def run_single_trial(
        self, config: RunConfig, job_id: str, forked: bool = True
    ) -> dict:
        """Execute a single trial either inline or in a spawned child process."""
        if not forked:
            return self._run_inline(config, job_id)
        return self._run_forked(config, job_id)

    def _run_inline(self, config: RunConfig, job_id: str) -> dict:
        runner = BacktestRunner(config, db_path=self.db_path)
        try:
            result = runner.run(job_id=job_id, bars=self.bars)
        finally:
            del runner
            gc.collect()
        return self._metrics(result, self.objective)

    def _run_forked(self, config: RunConfig, job_id: str) -> dict:
        queue = self._mp_ctx.Queue()
        proc = self._mp_ctx.Process(
            target=_spawn_child_target,
            args=(config, job_id, self.objective, self.db_path, queue, self.bars),
            daemon=True,
        )
        proc.start()
        proc.join()
        payload = None
        if proc.exitcode == 0:
            try:
                payload = queue.get(timeout=30)
            except Exception:
                payload = None
        if not isinstance(payload, dict):
            return {
                "status": "failed",
                "error": f"child process exited with code {proc.exitcode}",
            }
        return payload

    def _bad_result(self) -> tuple | float:
        """Return a sentinel that ranks worst on every objective."""
        if self.objective == "sqn":
            return float("-inf")
        return (float("-inf"), 0.0, float("-inf"))

    def run(self, study: optuna.Study, param_space: dict, n_trials: int) -> None:
        def objective_fn(trial: optuna.Trial):
            params = suggest_params(trial, param_space)
            config = self.base_config.with_overrides(params)
            job_id = f"trial_{trial.number}"
            try:
                metrics = (
                    self._run_inline(config, job_id)
                    if trial.number == 0  # primes the L2 data cache
                    else self._run_forked(config, job_id)
                )
            except Exception as e:
                print(
                    f"[Trial #{trial.number:03d}] FAILED: {e}",
                    flush=True,
                )
                # Return worst-possible values instead of TrialPruned.
                # TrialPruned leaves values=None which crashes the TPE
                # multi-objective sampler when building numpy arrays.
                return self._bad_result()

            primary = (
                metrics["sqn"] if self.objective == "sqn" else metrics["sharpe_ratio"]
            ) or 0.0
            if primary != primary:  # NaN (e.g. too few samples) -> reject trial
                print(
                    f"[Trial #{trial.number:03d}] REJECTED: NaN objective",
                    flush=True,
                )
                return self._bad_result()
            trades = float(metrics["trades"] or 0)
            pnl = metrics["pnl"] or 0.0

            print(
                f"[Trial #{trial.number:03d}] {self.primary_label}: {primary:+.2f} | "
                f"Trades: {int(trades):3d} | PnL: ${pnl:+,.2f} | Params: {params}",
                flush=True,
            )
            return primary if self.objective == "sqn" else (primary, trades, pnl)

        study.optimize(objective_fn, n_trials=n_trials)



def run_optuna_study(
    config_path: str,
    strategy_name: str,
    n_trials: int,
    params: list[str],
    db_path: str = "sbt.db",
    output_report: str | None = None,
    objective: str = "sharpe",
    overrides: dict | None = None,
) -> optuna.Study:
    """Run an Optuna optimization study for *strategy_name*.

    objective='sharpe' maximizes the (Sharpe, Trades, PnL) Pareto front;
    objective='sqn' purely maximizes the System Quality Number computed
    from per-trade returns.
    """
    if objective not in ("sharpe", "sqn"):
        raise ValueError(
            f"Unknown objective: {objective!r}. Expected 'sharpe' or 'sqn'."
        )

    base_config = RunConfig.from_toml(
        config_path, strategy_name, cli_overrides=overrides
    )

    if not params:
        params = get_default_param_space(strategy_name)
        if not params:
            raise ValueError(
                f"No --param specs provided for strategy '{strategy_name}'."
            )

    param_space = parse_param_spec(params)
    primary_label = "Sharpe Ratio" if objective == "sharpe" else "System Quality Number"

    # --- execution ----------------------------------------------------------
    executor = LocalExecutor(base_config, objective, primary_label, db_path=db_path)

    print(f"\n--- Starting Optuna Optimization for '{strategy_name}' ---")
    print(f"Objective mode: {objective} ({primary_label})")
    print(f"Total Trials: {n_trials}")
    print("Parameter Search Space:")
    for k, v in param_space.items():
        print(f"  - {k}: {v}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"opt_{strategy_name}_{objective}_{timestamp}"
    storage_url = f"sqlite:///{Path(db_path).resolve()}"

    directions = (
        ["maximize", "maximize", "maximize"] if objective == "sharpe" else ["maximize"]
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        directions=directions,
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
    )

    executor.run(study, param_space, n_trials)

    print("\n========== OPTIMIZATION COMPLETE ==========")
    failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"Completed {len(study.trials)} trials (failed={failed}, pruned={pruned}).")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No successful trials; skipping report generation.")
        return study
    if objective == "sqn":
        best = max(completed, key=lambda t: t.value)
        print(
            f"Best trial #{best.number}: SQN {best.value:+.4f} | Params: {best.params}"
        )
        report_path = generate_sqn_report(
            study=study,
            strategy_name=strategy_name,
            output_path=output_report or "reports/sqn_report.html",
            open_browser=base_config.open_report,
        )
    else:
        pareto_trials = _pareto_front(completed)
        print(f"Found {len(pareto_trials)} Pareto-optimal solutions.")
        report_path = generate_pareto_report(
            study=study,
            strategy_name=strategy_name,
            output_path=output_report or "reports/pareto_report.html",
            open_browser=base_config.open_report,
        )
    print(f"Report generated: {report_path}")
    return study


def _pareto_front(trials: list[optuna.trial.FrozenTrial]) -> list:
    """Maximize all three objectives; non-dominated filter."""
    pts = [t.values for t in trials]

    def dominates(i: int, j: int) -> bool:
        a, b = pts[i], pts[j]
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    front = []
    for i in range(len(trials)):
        if not any(dominates(j, i) for j in range(len(trials)) if j != i):
            front.append(trials[i])
    return front
