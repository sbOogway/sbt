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


def suggest_params(trial: optuna.Trial, param_space: dict[str, tuple]) -> dict[str, Any]:
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
