import math

from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class HurstExponentConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    window: int = 100
    threshold_trend: float = 0.6
    threshold_mean_rev: float = 0.4
    signal_period: int = 20

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class HurstExponent(SBTStrategy):
    def __init__(self, config: HurstExponentConfig) -> None:
        super().__init__(config)
        self._closes: list[float] = []

    def _hurst(self, data: list[float]) -> float | None:
        n = len(data)
        if n < 20:
            return None
        max_k = min(n // 2, 50)
        rs_list = []
        for k in [int(n * f) for f in [0.25, 0.5, 0.75]]:
            if k < 10:
                continue
            sub = data[:k]
            mean = sum(sub) / k
            deviations = [sum(sub[: i + 1]) - mean * (i + 1) for i in range(k)]
            r = max(deviations) - min(deviations)
            s = (sum((x - mean) ** 2 for x in sub) / k) ** 0.5
            if s > 0:
                rs_list.append((math.log(k), math.log(r / s)))
        if len(rs_list) < 2:
            return None
        n_pts = len(rs_list)
        mean_x = sum(x for x, _ in rs_list) / n_pts
        mean_y = sum(y for _, y in rs_list) / n_pts
        cov = sum((x - mean_x) * (y - mean_y) for x, y in rs_list)
        var = sum((x - mean_x) ** 2 for x, _ in rs_list)
        if var <= 0:
            return None
        return cov / var

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()
        self._closes.append(close)

        if len(self._closes) < self.config.window:
            return

        h = self._hurst(self._closes[-self.config.window :])
        if h is None:
            return

        lookback = self.config.signal_period
        if len(self._closes) < lookback + 1:
            return

        momentum = close / self._closes[-lookback - 1] - 1.0

        if not self.in_position:
            if h > self.config.threshold_trend:
                if momentum > 0:
                    self.open_position(OrderSide.BUY, close)
                else:
                    self.open_position(OrderSide.SELL, close)
            elif h < self.config.threshold_mean_rev:
                if momentum > 0:
                    self.open_position(OrderSide.SELL, close)
                else:
                    self.open_position(OrderSide.BUY, close)
        else:
            if h > self.config.threshold_trend:
                if self.position_side == OrderSide.BUY and momentum < 0:
                    self.exit_market()
                elif self.position_side == OrderSide.SELL and momentum > 0:
                    self.exit_market()
            else:
                if self.position_side == OrderSide.BUY and momentum > 0:
                    self.exit_market()
                elif self.position_side == OrderSide.SELL and momentum < 0:
                    self.exit_market()
