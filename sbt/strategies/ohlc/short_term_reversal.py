from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class ShortTermReversalConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    # Short-term reversal / overreaction fade (Kosc et al. 2019; Zaremba et
    # al. 2021; Caporale & Plastun 2019a): a daily move whose magnitude
    # exceeds threshold_mult x mean(|r|) over the trailing lookback days is
    # faded for holding_days.
    lookback: int = 30
    threshold_mult: float = 3.0
    holding_days: int = 1

    subscribe_funding: bool = True
    plugins: tuple[str, ...] = ("vol_scaling",)
    # Returns are fed manually per completed daily bar.
    vol_track_daily: bool = False
    rv_lookback: int = 30
    vol_max_scale: float = 3.0


class ShortTermReversal(SBTStrategy):
    def __init__(self, config: ShortTermReversalConfig) -> None:
        super().__init__(config)
        self._returns: list[float] = []
        self._prev_close: float | None = None
        self._bars_held: int = 0

    def _close_position(self) -> None:
        if self.exit_market():
            self._bars_held = 0

    def on_trading_bar(self, bar) -> None:
        close_price = bar.close.as_double()

        if self._prev_close is not None:
            daily_ret = close_price / self._prev_close - 1.0
            self._returns.append(daily_ret)
            scaler = self.plugins.get("vol_scaling")
            if scaler is not None:
                scaler.add_return(daily_ret)
        self._prev_close = close_price

        if self.in_position:
            self._bars_held += 1
            if self._bars_held >= self.config.holding_days:
                self._close_position()
            return

        warmup = max(self.config.lookback, self.config.rv_lookback)
        if len(self._returns) < warmup + 1:
            return

        window = self._returns[-self.config.lookback :]
        threshold = (
            self.config.threshold_mult
            * sum(abs(r) for r in window)
            / len(window)
        )
        r_last = self._returns[-1]
        if abs(r_last) <= threshold or threshold <= 0:
            return

        side = OrderSide.SELL if r_last > 0 else OrderSide.BUY
        if self.open_position(side, close_price):
            self._bars_held = 0
