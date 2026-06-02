from typing import Optional


class VolatilityScaler:
    """Moreira & Muir volatility scaling.

    Collects daily returns, computes realized variance (RV) over a lookback
    window at each rebalance, and produces a scaling weight = min(max_leverage,
    C / RV) where C is the average of all historical RVs.
    """

    def __init__(self, rv_lookback: int = 22, max_leverage: float = 2.0) -> None:
        self.rv_lookback = rv_lookback
        self.max_leverage = max_leverage
        self.daily_returns: list[float] = []
        self._rv_history: list[float] = []
        self._weight: float = 1.0
        self._last_rebalance_month: Optional[int] = None

    def add_return(self, daily_return: float) -> None:
        self.daily_returns.append(daily_return)

    def rebalance(self, current_month: int) -> float:
        """Rebalance if the month has changed since the last rebalance.

        Returns the current scaling weight.
        """
        if len(self.daily_returns) < self.rv_lookback:
            return self._weight

        if self._last_rebalance_month is not None and current_month == self._last_rebalance_month:
            return self._weight

        rv = sum(r * r for r in self.daily_returns[-self.rv_lookback:])
        self._rv_history.append(rv)
        c = sum(self._rv_history) / len(self._rv_history)
        self._weight = min(self.max_leverage, c / rv) if rv > 0 else self.max_leverage
        self._last_rebalance_month = current_month
        return self._weight

    @property
    def weight(self) -> float:
        return self._weight
