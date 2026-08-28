"""Cointegration helpers for the Leung & Nguyen style stat-arb spread.

Returns a normalized linear weight vector ``w`` over the leg price series such
that ``spread = w @ log(P)`` is (approximately) stationary and mean-reverting.
The primary leg (index 0) is normalized to weight 1; a stationary combination
then has at least one positive and one negative weight, giving a long/short
leg basket with near-zero market beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen

_METHODS = ("johansen", "engle_granger")


def _as_matrix(logp: pd.DataFrame) -> np.ndarray:
    return logp.astype(float).values


def engle_granger_weights(logp: pd.DataFrame) -> np.ndarray:
    """Regress the primary (col 0) log-price on the rest; weights w0=1, wj=-beta."""
    Xm = _as_matrix(logp)
    y = Xm[:, 0]
    design = np.column_stack([np.ones(len(y)), Xm[:, 1:]])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    w = np.zeros(len(logp.columns))
    w[0] = 1.0
    w[1:] = -beta[1:]
    return w


def johansen_weights(logp: pd.DataFrame, k_ar_diff: int = 1) -> np.ndarray:
    """Johansen maximum-eigenvalue cointegrating vector, normalized on leg 0."""
    Xm = _as_matrix(logp)
    if np.any(~np.isfinite(Xm)):
        raise ValueError("non-finite values in log-price window")
    joh = coint_johansen(Xm, det_order=0, k_ar_diff=k_ar_diff)
    evec = np.asarray(joh.evec)[:, 0].astype(float)
    if evec[0] == 0:
        raise ValueError("primary-leg cointegrating weight is zero; cannot normalize")
    return evec / evec[0]


def fit_weights(logp: pd.DataFrame, method: str) -> np.ndarray:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    return (
        johansen_weights(logp) if method == "johansen" else engle_granger_weights(logp)
    )
