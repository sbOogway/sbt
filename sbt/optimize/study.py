"""Orchestrates Optuna optimization studies.

Two objective modes:
- 'sharpe': 3-objective Pareto front (Sharpe, Trades, PnL) — the default.
- 'sqn': single-objective maximization of Van Tharp's System Quality Number.

Execution modes:
- LocalExecutor: runs backtests inline in this process (original path).
- SchedulerExecutor: routes each trial through the scheduler daemon via an
  ask/tell loop, parallel across the worker pool. Selected automatically when
  a scheduler answers a ping on --port; pass --local to force inline runs.
"""

import datetime
import gc
import multiprocessing
import time
from pathlib import Path

import optuna

from ..client.client import SbtClient
from ..core.config import RunConfig
from ..core.job import JobStatus
from ..core.runner import BacktestRunner
from .param_parser import parse_param_spec, suggest_params
from .report import generate_pareto_report, generate_sqn_report

_RESULT_POLL_S = 1.0


def _spawn_child_target(config: RunConfig, job_id: str, objective: str, db_path: str, queue):
    """Top-level target for spawn-multiprocessing backtest trials."""
    try:
        runner = BacktestRunner(config, db_path=db_path)
        try:
            result = runner.run(job_id=job_id)
        finally:
            del runner
            gc.collect()
        queue.put(
            {
                "pnl": result.pnl,
                "trades": result.num_trades,
                "sqn": result.sqn if objective == "sqn" else None,
                "sharpe_ratio": (
                    result.sharpe_ratio if objective == "sharpe" else None
                ),
            }
        )
    except BaseException as e:  # noqa: BLE001
        queue.put({"error": f"{type(e).__name__}: {e}"})


class LocalExecutor:
    """Run every trial inline in the current process (original path).

    Trial #0 runs inline (priming the L2 loader cache); every later trial
    forks a child so the multi-GB engine state is reclaimed by the OS on
    exit instead of fragmenting this process.
    """

    name = "local"

    def __init__(self, base_config: RunConfig, objective: str, primary_label: str, db_path: str = "sbt.db"):
        self.base_config = base_config
        self.objective = objective
        self.primary_label = primary_label
        self.db_path = db_path
        self._mp_ctx = multiprocessing.get_context("spawn")

    @staticmethod
    def _metrics(result, objective: str) -> dict:
        return {
            "pnl": result.pnl,
            "trades": result.num_trades,
            "sqn": result.sqn if objective == "sqn" else None,
            "sharpe_ratio": (
                result.sharpe_ratio if objective == "sharpe" else None
            ),
        }

    def _run_inline(self, config: RunConfig, job_id: str) -> dict:
        runner = BacktestRunner(config, db_path=self.db_path)
        try:
            result = runner.run(job_id=job_id)
        finally:
            del runner
            gc.collect()
        return self._metrics(result, self.objective)

    def _run_forked(self, config: RunConfig, job_id: str) -> dict:
        queue = self._mp_ctx.Queue()
        proc = self._mp_ctx.Process(
            target=_spawn_child_target,
            args=(config, job_id, self.objective, self.db_path, queue),
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
        if not isinstance(payload, dict) or "error" in payload:
            reason = (
                payload.get("error")
                if isinstance(payload, dict)
                else f"child exited with code {proc.exitcode}"
            )
            raise RuntimeError(reason)
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


class SchedulerExecutor:
    """Route trials through the scheduler daemon (parallel across workers)."""

    name = "scheduler"

    def __init__(
        self,
        client: SbtClient,
        base_config: RunConfig,
        objective: str,
        primary_label: str,
        port: int,
    ):
        self.client = client
        self.base_config = base_config
        self.objective = objective
        self.primary_label = primary_label
        self.port = port

    def _max_inflight(self) -> int:
        try:
            status = self.client.get_status()
            return max(1, int(status.get("workers_total", 1)))
        except Exception:
            return 1

    def _result_values(self, result_dict: dict):
        primary = (
            result_dict.get("sqn" if self.objective == "sqn" else "sharpe_ratio")
        ) or 0.0
        if primary != primary:  # NaN
            primary = 0.0
        trades = float(result_dict.get("num_trades") or 0)
        pnl = float(result_dict.get("pnl") or 0.0)
        return primary if self.objective == "sqn" else (primary, trades, pnl)

    def run(self, study: optuna.Study, param_space: dict, n_trials: int) -> None:
        max_inflight = self._max_inflight()
        print(f"In-flight cap from scheduler pool: {max_inflight} worker(s)")

        inflight: dict[str, tuple[int, dict]] = {}  # job_id -> (trial_number, params)
        submitted = 0

        while submitted < n_trials or inflight:
            # Refill while the pool has capacity.
            while len(inflight) < max_inflight and submitted < n_trials:
                trial = study.ask()
                params = suggest_params(trial, param_space)
                config = self.base_config.with_overrides(params)
                try:
                    job_id = self.client.submit(
                        config, study_name=study.study_name
                    )
                except Exception as e:
                    print(f"[Trial #{trial.number:03d}] submit failed: {e}")
                    study.tell(
                        trial,
                        values=self._bad_result(),
                        state=optuna.trial.TrialState.FAIL,
                    )
                    continue
                inflight[job_id] = (trial.number, params)
                print(
                    f"[Trial #{trial.number:03d}] queued as job {job_id} | Params: {params}",
                    flush=True,
                )
                submitted += 1

            if not inflight:
                break

            time.sleep(_RESULT_POLL_S)

            for job_id, (tno, params) in list(inflight.items()):
                try:
                    resp = self.client.get_result(job_id)
                except Exception:
                    continue  # transient; retry next poll
                if resp.get("status") == "pending":
                    continue
                if resp.get("status") != "ok":
                    print(
                        f"[Trial #{tno:03d}] job {job_id} lost ({resp.get('status')}); failing trial"
                    )
                    study.tell(
                        tno,
                        values=self._bad_result(),
                        state=optuna.trial.TrialState.FAIL,
                    )
                    inflight.pop(job_id)
                    continue

                result = resp["result"]
                if result.get("status") != JobStatus.DONE.value:
                    print(
                        f"[Trial #{tno:03d}] FAILED on scheduler: {result.get('error')}"
                    )
                    study.tell(
                        tno,
                        values=self._bad_result(),
                        state=optuna.trial.TrialState.FAIL,
                    )
                    inflight.pop(job_id)
                    continue

                values = self._result_values(result)
                primary = values if self.objective == "sqn" else values[0]
                trades = int(float(result.get("num_trades") or 0))
                pnl = float(result.get("pnl") or 0.0)
                print(
                    f"[Trial #{tno:03d}] {self.primary_label}: {primary:+.2f} | "
                    f"Trades: {trades:3d} | PnL: ${pnl:+,.2f} | Params: {params}",
                    flush=True,
                )
                study.tell(tno, values)
                inflight.pop(job_id)


def run_optuna_study(
    config_path: str,
    strategy_name: str,
    n_trials: int,
    params: list[str],
    db_path: str = "sbt.db",
    port: int = 5555,
    output_report: str | None = None,
    objective: str = "sharpe",
    overrides: dict | None = None,
    local: bool = False,
) -> optuna.Study:
    """Run an Optuna optimization study for *strategy_name*.

    objective='sharpe' maximizes the (Sharpe, Trades, PnL) Pareto front;
    objective='sqn' purely maximizes the System Quality Number computed
    from per-trade returns.

    Execution is routed through the scheduler when one is reachable at
    ``tcp://127.0.0.1:{port}``; pass ``local=True`` to force in-process runs.
    """
    if objective not in ("sharpe", "sqn"):
        raise ValueError(
            f"Unknown objective: {objective!r}. Expected 'sharpe' or 'sqn'."
        )

    base_config = RunConfig.from_toml(
        config_path, strategy_name, cli_overrides=overrides
    )

    if not params:
        # Default parameter search spaces if none supplied on CLI
        if strategy_name == "overnight_drift":
            params = [
                "rv_lookback=int(3,30)",
                "vol_max_scale=float(1.0,4.0)",
                "entry_time=cat(18:00,19:00,20:00,21:00)",
                "exit_time=cat(04:00,06:00,08:00,14:00)",
            ]
        elif strategy_name == "orb":
            params = [
                "orb_period=int(1,6)",
                "atr_period=int(7,28)",
                "stop_multiple=float(1.0,3.5)",
                "rv_lookback=int(5,30)",
                "vol_max_scale=float(1.0,4.0)",
            ]
        elif strategy_name == "glucksmann":
            params = [
                "bb_length=int(10,30)",
                "bb_std=float(1.5,2.5)",
                "sma_fast=int(10,30)",
                "sma_slow=int(40,70)",
            ]
        else:
            raise ValueError(
                f"No --param specs provided for strategy '{strategy_name}'."
            )

    param_space = parse_param_spec(params)
    primary_label = (
        "Sharpe Ratio" if objective == "sharpe" else "System Quality Number"
    )

    # --- execution routing -------------------------------------------------
    endpoint = f"tcp://127.0.0.1:{port}"
    client = SbtClient(endpoint=endpoint, timeout_ms=2000)
    executor: LocalExecutor | SchedulerExecutor
    if local:
        executor = LocalExecutor(base_config, objective, primary_label, db_path=db_path)
        print("Execution mode: local (--local)")
    elif client.ping():
        executor = SchedulerExecutor(
            client, base_config, objective, primary_label, port
        )
        print(f"Execution mode: scheduler at {endpoint}")
    else:
        executor = LocalExecutor(base_config, objective, primary_label, db_path=db_path)
        print(f"No scheduler reachable at {endpoint}; falling back to local runs")

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

    try:
        executor.run(study, param_space, n_trials)
    finally:
        if isinstance(executor, SchedulerExecutor):
            client.close()

    print("\n========== OPTIMIZATION COMPLETE ==========")
    failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(
        f"Completed {len(study.trials)} trials "
        f"(failed={failed}, pruned={pruned})."
    )
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
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
        return all(x >= y for x, y in zip(a, b)) and any(
            x > y for x, y in zip(a, b)
        )

    front = []
    for i in range(len(trials)):
        if not any(dominates(j, i) for j in range(len(trials)) if j != i):
            front.append(trials[i])
    return front
