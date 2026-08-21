"""Orchestrates Optuna optimization studies.

Two objective modes:
- 'sharpe': 3-objective Pareto front (Sharpe, Trades, PnL) — the default.
- 'sqn': single-objective maximization of Van Tharp's System Quality Number.
"""

import datetime
from pathlib import Path

import optuna

from ..core.config import RunConfig
from ..core.runner import BacktestRunner
from .param_parser import parse_param_spec, suggest_params
from .report import generate_pareto_report, generate_sqn_report


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

    def evaluate(trial: optuna.Trial) -> tuple[float, float, float]:
        trial_params = suggest_params(trial, param_space)
        config = base_config.with_overrides(trial_params)

        runner = BacktestRunner(config)
        result = runner.run(job_id=f"trial_{trial.number}")

        primary = (
            result.sqn if objective == "sqn" else result.sharpe_ratio
        ) or 0.0
        trades = float(result.num_trades or 0)
        pnl = result.pnl or 0.0

        print(
            f"[Trial #{trial.number:03d}] {primary_label}: {primary:+.2f} | Trades: {int(trades):3d} | PnL: ${pnl:+,.2f} | Params: {trial_params}"
        )
        return (primary, trades, pnl)

    def objective_fn(trial: optuna.Trial):
        values = evaluate(trial)
        return values[0] if objective == "sqn" else values

    study.optimize(objective_fn, n_trials=n_trials)

    print("\n========== OPTIMIZATION COMPLETE ==========")
    print(f"Completed {len(study.trials)} trials.")
    if objective == "sqn":
        best = study.best_trial
        print(
            f"Best trial #{best.number}: SQN {best.value:+.4f} | Params: {best.params}"
        )
        report_path = generate_sqn_report(
            study=study,
            strategy_name=strategy_name,
            output_path=output_report or "reports/sqn_report.html",
        )
    else:
        print(f"Found {len(study.best_trials)} Pareto-optimal solutions.")
        report_path = generate_pareto_report(
            study=study,
            strategy_name=strategy_name,
            output_path=output_report or "reports/pareto_report.html",
        )
    print(f"Report generated: {report_path}")
    return study
