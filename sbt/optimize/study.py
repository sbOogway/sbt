"""Orchestrates Optuna multi-objective optimization studies."""

import datetime
from pathlib import Path
import tomllib
import optuna

from ..core.config import RunConfig
from ..core.runner import BacktestRunner
from .param_parser import parse_param_spec, suggest_params
from .report import generate_pareto_report


def run_optuna_study(
    config_path: str,
    strategy_name: str,
    n_trials: int,
    params: list[str],
    db_path: str = "sbt.db",
    port: int = 5555,
    output_report: str = "reports/pareto_report.html",
) -> optuna.Study:
    """Run a multi-objective hyperparameter optimization study."""
    base_config = RunConfig.from_toml(config_path, strategy_name)

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
            raise ValueError(f"No --param specs provided for strategy '{strategy_name}'.")

    param_space = parse_param_spec(params)
    print(f"\n--- Starting Optuna Optimization for '{strategy_name}' ---")
    print(f"Total Trials: {n_trials}")
    print("Parameter Search Space:")
    for k, v in param_space.items():
        print(f"  - {k}: {v}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"opt_{strategy_name}_{timestamp}"
    storage_url = f"sqlite:///{Path(db_path).resolve()}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        directions=["maximize", "maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        trial_params = suggest_params(trial, param_space)
        config = base_config.with_overrides(trial_params)

        runner = BacktestRunner(config)
        result = runner.run(job_id=f"trial_{trial.number}")

        sharpe = result.sharpe_ratio or 0.0
        trades = float(result.num_trades or 0)
        pnl = result.pnl or 0.0

        print(
            f"[Trial #{trial.number:03d}] Sharpe: {sharpe:+.2f} | Trades: {int(trades):3d} | PnL: ${pnl:+,.2f} | Params: {trial_params}"
        )
        return (sharpe, trades, pnl)

    study.optimize(objective, n_trials=n_trials)

    print(f"\n========== OPTIMIZATION COMPLETE ==========")
    print(f"Completed {len(study.trials)} trials.")
    print(f"Found {len(study.best_trials)} Pareto-optimal solutions.")

    report_path = generate_pareto_report(
        study=study,
        strategy_name=strategy_name,
        output_path=output_report,
    )
    print(f"Pareto frontier report generated: {report_path}")
    return study
