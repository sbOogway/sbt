import math
from collections.abc import Sequence

import pandas as pd
from nautilus_trader.analysis.statistic import PortfolioStatistic


def system_quality_number(trade_returns: Sequence[float]) -> float | None:
    """Van Tharp's System Quality Number over per-trade returns.

    SQN = sqrt(N) * mean(R) / stddev(R) with sample stddev (ddof=1).
    Returns None when there are fewer than 2 trades or zero dispersion.
    """
    r = [float(x) for x in trade_returns if x is not None]
    n = len(r)
    if n < 2:
        return None
    mean = sum(r) / n
    var = sum((x - mean) ** 2 for x in r) / (n - 1)
    std = math.sqrt(var)
    if std < 1e-12:
        return None
    return math.sqrt(n) * mean / std


class CalmarRatio(PortfolioStatistic):
    def calculate_from_returns(self, returns: pd.Series) -> float | None:
        if not self._check_valid_returns(returns):
            return None
        daily = self._downsample_to_daily_bins(returns)
        if len(daily) < 2:
            return None
        ann_return = daily.mean() * 252
        cum = (1 + daily).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = dd.min()
        if max_dd >= 0:
            return None
        return float(ann_return / abs(max_dd))


class AnnualizedReturn(PortfolioStatistic):
    def calculate_from_returns(self, returns: pd.Series) -> float | None:
        if not self._check_valid_returns(returns):
            return None
        daily = self._downsample_to_daily_bins(returns)
        if len(daily) < 2:
            return None
        ann_return = float(daily.mean() * 252)
        return ann_return


class RunConfig(PortfolioStatistic):
    def __init__(self, **kwargs: str) -> None:
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "Run Config"

    def calculate_from_positions(self, positions: list) -> str:
        return " | ".join(f"{k}={v}" for k, v in self._kwargs.items())
