from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class GarchVolForecastConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    lookback: int = 100
    forecast_horizon: int = 5
    alpha: float = 0.1
    beta: float = 0.85

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class GarchVolForecast(SBTStrategy):
    def __init__(self, config: GarchVolForecastConfig) -> None:
        super().__init__(config)
        self._returns: list[float] = []
        self._prev_close: float | None = None
        self._current_var: float = 0.0
        self._bars_since_entry: int = 0

    def on_trading_bar(self, bar) -> None:
        close = bar.close.as_double()

        if self._prev_close is not None:
            ret = close / self._prev_close - 1.0
            self._returns.append(ret)
        self._prev_close = close

        lb = self.config.lookback
        if len(self._returns) < lb:
            return

        hist_var = sum(r**2 for r in self._returns[-lb:]) / lb
        if self._current_var == 0:
            self._current_var = hist_var

        if len(self._returns) > 0:
            r = self._returns[-1]
            self._current_var = (
                self.config.alpha * r**2
                + self.config.beta * self._current_var
                + (1 - self.config.alpha - self.config.beta) * hist_var
            )

        forecast_var = self._current_var * (1 + self.config.alpha) ** self.config.forecast_horizon
        current_vol = self._current_var**0.5
        forecast_vol = forecast_var**0.5

        vol_change = (forecast_vol - current_vol) / current_vol if current_vol > 0 else 0

        if len(self._returns) < 5:
            return
        momentum = sum(self._returns[-5:]) / 5

        if not self.in_position:
            if vol_change > 0.1:
                if momentum > 0:
                    self.open_position(OrderSide.BUY, close)
                else:
                    self.open_position(OrderSide.SELL, close)
            elif vol_change < -0.1:
                self.open_position(OrderSide.BUY if momentum < 0 else OrderSide.SELL, close)
        else:
            self._bars_since_entry += 1
            if self._bars_since_entry > 20:
                self.exit_market()
                self._bars_since_entry = 0
