"""Signed trade flow with liquidity gating.

Bieganowski & Slepaczuk (2026), "Explainable Patterns in
Cryptocurrency Microstructure", arXiv:2602.00776. SHAP analysis of
their feature library shows order flow imbalance drives short-horizon
returns (monotone, concave at extremes) while wide spreads destroy
predictability. The strategy uses the signed trade-flow ratio (order
flow ratio, OFR): an EWMA of trade sizes signed by aggressor side,
normalized by the EWMA of absolute sizes, with a spread gate on new
entries.
"""

from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide

from ...plugins import SBTStrategyConfig
from .base import L2EventStrategy, clamped_dt_s, ewma_alpha


class L2SignedTradeFlowConfig(SBTStrategyConfig, kw_only=True, frozen=True):
    """Configuration for the signed trade flow strategy."""

    signal_interval_ms: int = 500
    entry_threshold: float = 0.3
    exit_threshold: float = 0.05
    max_hold_seconds: int = 120
    max_spread_bps: float = 5.0
    capital_fraction: float = 0.10
    trade_half_life_ms: int = 10000


class L2SignedTradeFlow(L2EventStrategy):
    """Trades the smoothed ratio of buyer-initiated to total flow."""

    needs_trade_ticks: bool = True

    def __init__(self, config: L2SignedTradeFlowConfig) -> None:
        super().__init__(config)
        self._signed_ewma: float = 0.0
        self._abs_ewma: float = 0.0
        self._signed_window: float = 0.0
        self._abs_window: float = 0.0
        self._tf_last_ts: int | None = None

    def on_trade_tick(self, tick: TradeTick) -> None:
        if tick.aggressor_side == AggressorSide.NO_AGGRESSOR:
            return
        signed = float(tick.size) * (
            1.0 if tick.aggressor_side == AggressorSide.BUYER else -1.0
        )
        self._signed_window += signed
        self._abs_window += abs(signed)

    def _compute_signal(self, ts_event: int) -> float | None:
        if self._tf_last_ts is None:
            self._tf_last_ts = ts_event
            self._signed_window = 0.0
            self._abs_window = 0.0
            return 0.0
        dt_s = clamped_dt_s(ts_event, self._tf_last_ts)
        alpha = ewma_alpha(dt_s, self.config.trade_half_life_ms / 1000.0)
        self._signed_ewma += alpha * (self._signed_window - self._signed_ewma)
        self._abs_ewma += alpha * (self._abs_window - self._abs_ewma)
        self._tf_last_ts = ts_event
        self._signed_window = 0.0
        self._abs_window = 0.0
        if self._abs_ewma <= 0:
            return None
        return self._signed_ewma / self._abs_ewma
