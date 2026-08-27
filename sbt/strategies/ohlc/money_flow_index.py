from nautilus_trader.model.enums import OrderSide

from ...plugins import SBTBarStrategyConfig
from ..base import SBTStrategy


class MoneyFlowIndexConfig(SBTBarStrategyConfig, kw_only=True, frozen=True):
    period: int = 14
    overbought: float = 80.0
    oversold: float = 20.0

    risk_percent: float = 0.1
    plugins: tuple[str, ...] = ("vol_scaling",)
    rv_lookback: int = 22
    vol_max_scale: float = 2.0


class MoneyFlowIndex(SBTStrategy):
    def __init__(self, config: MoneyFlowIndexConfig) -> None:
        super().__init__(config)
        self._tp_vol: list[float] = []
        self._neg_tp_vol: list[float] = []

    def on_trading_bar(self, bar) -> None:
        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        volume = bar.volume.as_double() if hasattr(bar, "volume") else 1.0

        tp = (high + low + close) / 3.0
        tp_vol = tp * volume

        self._tp_vol.append(tp_vol)
        self._neg_tp_vol.append(tp_vol if close > (high + low) / 2 else 0.0)

        if len(self._tp_vol) < self.config.period + 1:
            return

        pos_flow = sum(self._neg_tp_vol[-self.config.period :])
        total_flow = sum(self._tp_vol[-self.config.period :])

        if total_flow == 0:
            return

        mfr = pos_flow / total_flow
        mfi = (1.0 - 1.0 / (1.0 + mfr)) * 100 if mfr > 0 else 0.0

        if not self.in_position:
            if mfi < self.config.oversold:
                self.open_position(OrderSide.BUY, close)
            elif mfi > self.config.overbought:
                self.open_position(OrderSide.SELL, close)
        else:
            if self.position_side == OrderSide.BUY and mfi > 50:
                self.exit_market()
            elif self.position_side == OrderSide.SELL and mfi < 50:
                self.exit_market()
