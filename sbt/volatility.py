class VolatilityScaler:
    """Rolling-window volatility scaling (Moreira & Muir).

    Each day, computes realized variance (RV) over the trailing rv_lookback
    days, compares it against the average of all trailing-window RVs seen so
    far, and sets weight = min(vol_max_scale, C / RV).
    """

    def __init__(self, rv_lookback: int = 22, vol_max_scale: float = 2.0) -> None:
        self.rv_lookback = rv_lookback
        self.vol_max_scale = vol_max_scale
        self.daily_returns: list[float] = []
        self._rv_history: list[float] = []
        self._weight: float = 1.0

    def add_return(self, daily_return: float) -> None:
        self.daily_returns.append(daily_return)
        if len(self.daily_returns) < self.rv_lookback:
            return

        rv = sum(r * r for r in self.daily_returns[-self.rv_lookback :])
        self._rv_history.append(rv)
        c = sum(self._rv_history) / len(self._rv_history)
        self._weight = min(self.vol_max_scale, c / rv) if rv > 0 else self.vol_max_scale

    @property
    def weight(self) -> float:
        return self._weight
