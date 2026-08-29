"""Parameter space definition and parsing for Optuna optimization."""

import re
from typing import Any

import optuna


def parse_param_spec(param_strs: list[str]) -> dict[str, tuple]:
    """Parse string parameter specifications into structured definitions.

    Formats supported:
    - name=int(min,max)
    - name=float(min,max)
    - name=cat(val1,val2,val3)

    Example:
        ['rv_lookback=int(3,30)', 'vol_max_scale=float(1.0,5.0)', 'entry_time=cat(18:00,19:00,20:00)']
    """
    space = {}
    pattern = re.compile(r"^([a-zA-Z0-9_]+)\s*=\s*(int|float|cat)\((.+)\)$")

    for p in param_strs:
        match = pattern.match(p.strip())
        if not match:
            raise ValueError(
                f"Invalid parameter specification: '{p}'. "
                "Expected format: name=int(min,max) or name=float(min,max) or name=cat(val1,val2)"
            )
        name, kind, raw_args = match.groups()
        args = [a.strip() for a in raw_args.split(",")]

        if kind == "int":
            if len(args) != 2:
                raise ValueError(f"int() requires min,max; got: {raw_args}")
            space[name] = ("int", int(args[0]), int(args[1]))
        elif kind == "float":
            if len(args) != 2:
                raise ValueError(f"float() requires min,max; got: {raw_args}")
            space[name] = ("float", float(args[0]), float(args[1]))
        elif kind == "cat":
            # preserve types if int/float or string
            cat_vals = []
            for a in args:
                if a.lower() == "true":
                    cat_vals.append(True)
                elif a.lower() == "false":
                    cat_vals.append(False)
                else:
                    try:
                        cat_vals.append(int(a))
                    except ValueError:
                        try:
                            cat_vals.append(float(a))
                        except ValueError:
                            cat_vals.append(a)
            space[name] = ("cat", cat_vals)

    return space


def suggest_params(
    trial: optuna.Trial, param_space: dict[str, tuple]
) -> dict[str, Any]:
    """Sample parameter values from Optuna trial according to param_space."""
    params = {}
    for name, spec in param_space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "cat":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


DEFAULT_PARAM_SPACES: dict[str, list[str]] = {
    "overnight_drift": [
        "rv_lookback=int(3,30)",
        "vol_max_scale=float(1.0,4.0)",
        "entry_time=cat(18:00,19:00,20:00,21:00)",
        "exit_time=cat(04:00,06:00,08:00,14:00)",
    ],
    "orb": [
        "orb_period=int(1,6)",
        "atr_period=int(7,28)",
        "stop_multiple=float(1.0,3.5)",
        "rv_lookback=int(5,30)",
        "vol_max_scale=float(1.0,4.0)",
    ],
    "glucksmann": [
        "bb_length=int(10,30)",
        "bb_std=float(1.5,2.5)",
        "sma_fast=int(10,30)",
        "sma_slow=int(40,70)",
    ],
    "trix": [
        "period=int(5,30)",
        "signal_period=int(3,15)",
    ],
    "keltner_channel": [
        "ema_period=int(10,40)",
        "atr_period=int(5,20)",
        "atr_mult=float(1.0,3.5)",
    ],
    "negative_volume_index": [
        "ema_period=int(100,400)",
    ],
    "envelope": [
        "period=int(10,40)",
        "pct=float(0.5,5.0)",
    ],
    "adx_trend": [
        "adx_period=int(7,30)",
        "adx_threshold=float(15.0,35.0)",
        "ema_fast=int(5,20)",
        "ema_slow=int(15,50)",
    ],
    "donchian_adx": [
        "channel_period=int(10,40)",
        "adx_period=int(7,30)",
        "adx_threshold=float(10.0,30.0)",
    ],
    "triple_ema_crossover": [
        "fast_period=int(3,15)",
        "mid_period=int(10,35)",
        "slow_period=int(30,100)",
    ],
    "trend_filter": [
        "fast_ma=int(5,20)",
        "slow_ma=int(100,300)",
        "filter_period=int(20,100)",
    ],
    "zigzag_momentum": [
        "swing_pct=float(1.0,6.0)",
        "holding_bars=int(3,20)",
    ],
}


def get_default_param_space(strategy_name: str) -> list[str] | None:
    """Return default parameter space specs for *strategy_name* if registered."""
    return DEFAULT_PARAM_SPACES.get(strategy_name)
