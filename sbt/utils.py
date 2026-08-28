import importlib
from datetime import timedelta
from decimal import Decimal

from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity

_STRATEGY_REGISTRY = {
    # OHLC / bar-driven strategies (sbt/strategies/ohlc/)
    "bitcoin_intraday_momentum": (
        "strategies.ohlc.bitcoin_intraday_momentum",
        "BitcoinIntradayMomentum",
        "BitcoinIntradayMomentumConfig",
    ),
    "glucksmann": (
        "strategies.ohlc.glucksmann",
        "GlucksmannStrategy",
        "GlucksmannConfig",
    ),
    "key_breakout": (
        "strategies.ohlc.key_breakout",
        "KeyBreakout",
        "KeyBreakoutConfig",
    ),
    "orb": (
        "strategies.ohlc.orb",
        "ORBStrategy",
        "ORBConfig",
    ),
    "overnight_drift": (
        "strategies.ohlc.overnight_drift",
        "OvernightDrift",
        "OvernightDriftConfig",
    ),
    "short_term_reversal": (
        "strategies.ohlc.short_term_reversal",
        "ShortTermReversal",
        "ShortTermReversalConfig",
    ),
    "weekday_seasonality": (
        "strategies.ohlc.weekday_seasonality",
        "WeekdaySeasonality",
        "WeekdaySeasonalityConfig",
    ),
    "dual_sma_crossover": (
        "strategies.ohlc.dual_sma_crossover",
        "DualSmaCrossover",
        "DualSmaCrossoverConfig",
    ),
    "bollinger_squeeze": (
        "strategies.ohlc.bollinger_squeeze",
        "BollingerSqueeze",
        "BollingerSqueezeConfig",
    ),
    "rsi_trend": (
        "strategies.ohlc.rsi_trend",
        "RsiTrend",
        "RsiTrendConfig",
    ),
    "zscore_mean_reversion": (
        "strategies.ohlc.zscore_mean_reversion",
        "ZscoreMeanReversion",
        "ZscoreMeanReversionConfig",
    ),
    "bollinger_mean_reversion": (
        "strategies.ohlc.bollinger_mean_reversion",
        "BollingerMeanReversion",
        "BollingerMeanReversionConfig",
    ),
    "donchian_breakout": (
        "strategies.ohlc.donchian_breakout",
        "DonchianBreakout",
        "DonchianBreakoutConfig",
    ),
    "triple_ema_crossover": (
        "strategies.ohlc.triple_ema_crossover",
        "TripleEmaCrossover",
        "TripleEmaCrossoverConfig",
    ),
    "momentum_breakout": (
        "strategies.ohlc.momentum_breakout",
        "MomentumBreakout",
        "MomentumBreakoutConfig",
    ),
    "volatility_regime": (
        "strategies.ohlc.volatility_regime",
        "VolatilityRegime",
        "VolatilityRegimeConfig",
    ),
    "hurst_exponent": (
        "strategies.ohlc.hurst_exponent",
        "HurstExponent",
        "HurstExponentConfig",
    ),
    "kalman_trend": (
        "strategies.ohlc.kalman_trend",
        "KalmanTrend",
        "KalmanTrendConfig",
    ),
    "garch_vol_forecast": (
        "strategies.ohlc.garch_vol_forecast",
        "GarchVolForecast",
        "GarchVolForecastConfig",
    ),
    "chande_momentum": (
        "strategies.ohlc.chande_momentum",
        "ChandeMomentum",
        "ChandeMomentumConfig",
    ),
    "stochastic_rsi": (
        "strategies.ohlc.stochastic_rsi",
        "StochasticRsi",
        "StochasticRsiConfig",
    ),
    "williams_r": (
        "strategies.ohlc.williams_r",
        "WilliamsR",
        "WilliamsRConfig",
    ),
    "cci": (
        "strategies.ohlc.cci",
        "Cci",
        "CciConfig",
    ),
    "keltner_channel": (
        "strategies.ohlc.keltner_channel",
        "KeltnerChannel",
        "KeltnerChannelConfig",
    ),
    "supertrend": (
        "strategies.ohlc.supertrend",
        "Supertrend",
        "SupertrendConfig",
    ),
    "adx_trend": (
        "strategies.ohlc.adx_trend",
        "AdxTrend",
        "AdxTrendConfig",
    ),
    "range_breakout": (
        "strategies.ohlc.range_breakout",
        "RangeBreakout",
        "RangeBreakoutConfig",
    ),
    "donchian_adx": (
        "strategies.ohlc.donchian_adx",
        "DonchianAdx",
        "DonchianAdxConfig",
    ),
    "aroon": (
        "strategies.ohlc.aroon",
        "Aroon",
        "AroonConfig",
    ),
    "vortex_indicator": (
        "strategies.ohlc.vortex_indicator",
        "VortexIndicator",
        "VortexIndicatorConfig",
    ),
    "elder_ray": (
        "strategies.ohlc.elder_ray",
        "ElderRay",
        "ElderRayConfig",
    ),
    "chaikin_money_flow": (
        "strategies.ohlc.chaikin_money_flow",
        "ChaikinMoneyFlow",
        "ChaikinMoneyFlowConfig",
    ),
    "money_flow_index": (
        "strategies.ohlc.money_flow_index",
        "MoneyFlowIndex",
        "MoneyFlowIndexConfig",
    ),
    "trix": (
        "strategies.ohlc.trix",
        "TRIX",
        "TRIXConfig",
    ),
    "force_index": (
        "strategies.ohlc.force_index",
        "ForceIndex",
        "ForceIndexConfig",
    ),
    "ultimate_oscillator": (
        "strategies.ohlc.ultimate_oscillator",
        "UltimateOscillator",
        "UltimateOscillatorConfig",
    ),
    "coppock_curve": (
        "strategies.ohlc.coppock_curve",
        "CoppockCurve",
        "CoppockCurveConfig",
    ),
    "mass_index": (
        "strategies.ohlc.mass_index",
        "MassIndex",
        "MassIndexConfig",
    ),
    "ease_of_movement": (
        "strategies.ohlc.ease_of_movement",
        "EaseOfMovement",
        "EaseOfMovementConfig",
    ),
    "chaikin_oscillator": (
        "strategies.ohlc.chaikin_oscillator",
        "ChaikinOscillator",
        "ChaikinOscillatorConfig",
    ),
    "negative_volume_index": (
        "strategies.ohlc.negative_volume_index",
        "NegativeVolumeIndex",
        "NegativeVolumeIndexConfig",
    ),
    "price_channel": (
        "strategies.ohlc.price_channel",
        "PriceChannel",
        "PriceChannelConfig",
    ),
    "true_range_breakout": (
        "strategies.ohlc.true_range_breakout",
        "TrueRangeBreakout",
        "TrueRangeBreakoutConfig",
    ),
    "momentum_reversal": (
        "strategies.ohlc.momentum_reversal",
        "MomentumReversal",
        "MomentumReversalConfig",
    ),
    "month_end_momentum": (
        "strategies.ohlc.month_end_momentum",
        "MonthEndMomentum",
        "MonthEndMomentumConfig",
    ),
    "volatility_breakout": (
        "strategies.ohlc.volatility_breakout",
        "VolatilityBreakout",
        "VolatilityBreakoutConfig",
    ),
    "adaptive_ma": (
        "strategies.ohlc.adaptive_ma",
        "AdaptiveMa",
        "AdaptiveMaConfig",
    ),
    "triple_ma": (
        "strategies.ohlc.triple_ma",
        "TripleMa",
        "TripleMaConfig",
    ),
    "inside_bar_breakout": (
        "strategies.ohlc.inside_bar_breakout",
        "InsideBarBreakout",
        "InsideBarBreakoutConfig",
    ),
    "range_expansion": (
        "strategies.ohlc.range_expansion",
        "RangeExpansion",
        "RangeExpansionConfig",
    ),
    "trend_strength": (
        "strategies.ohlc.trend_strength",
        "TrendStrength",
        "TrendStrengthConfig",
    ),
    "momentum_crash": (
        "strategies.ohlc.momentum_crash",
        "MomentumCrash",
        "MomentumCrashConfig",
    ),
    "xsectional_momentum": (
        "strategies.ohlc.xsectional.xsectional_momentum",
        "XSectionalMomentum",
        "XSectionalMomentumConfig",
    ),
    "factor_long_short": (
        "strategies.ohlc.xsectional.factor_long_short",
        "FactorLongShort",
        "FactorLongShortConfig",
    ),
    "trend_filter": (
        "strategies.ohlc.trend_filter",
        "TrendFilter",
        "TrendFilterConfig",
    ),
    "dual_thrust": (
        "strategies.ohlc.dual_thrust",
        "DualThrust",
        "DualThrustConfig",
    ),
    "kama_cross": (
        "strategies.ohlc.kama_cross",
        "KamaCross",
        "KamaCrossConfig",
    ),
    "relative_vigor": (
        "strategies.ohlc.relative_vigor",
        "RelativeVigor",
        "RelativeVigorConfig",
    ),
    "zigzag_momentum": (
        "strategies.ohlc.zigzag_momentum",
        "ZigzagMomentum",
        "ZigzagMomentumConfig",
    ),
    "envelope": (
        "strategies.ohlc.envelope",
        "Envelope",
        "EnvelopeConfig",
    ),
    "psar": (
        "strategies.ohlc.psar",
        "PSAR",
        "PSARConfig",
    ),
    "smi": (
        "strategies.ohlc.smi",
        "SMI",
        "SmiConfig",
    ),
    "hilo": (
        "strategies.ohlc.hilo",
        "Hilo",
        "HiloConfig",
    ),
    "mean_deviation": (
        "strategies.ohlc.mean_deviation",
        "MeanDeviation",
        "MeanDeviationConfig",
    ),
    "t3": (
        "strategies.ohlc.t3",
        "T3",
        "T3Config",
    ),
    # L2 order book strategies (sbt/strategies/l2/)
    "l2_order_imbalance": (
        "strategies.l2.order_imbalance",
        "L2OrderImbalance",
        "L2OrderImbalanceConfig",
    ),
    "l2_queue_imbalance": (
        "strategies.l2.queue_imbalance",
        "L2QueueImbalance",
        "L2QueueImbalanceConfig",
    ),
    "l2_best_quote_ofi": (
        "strategies.l2.best_quote_ofi",
        "L2BestQuoteOFI",
        "L2BestQuoteOFIConfig",
    ),
    "l2_multilevel_ofi": (
        "strategies.l2.multilevel_ofi",
        "L2MultilevelOFI",
        "L2MultilevelOFIConfig",
    ),
    "l2_signed_trade_flow": (
        "strategies.l2.signed_trade_flow",
        "L2SignedTradeFlow",
        "L2SignedTradeFlowConfig",
    ),
    "l2_book_pressure": (
        "strategies.l2.book_pressure",
        "L2BookPressure",
        "L2BookPressureConfig",
    ),
    "l2_microprice": (
        "strategies.l2.microprice",
        "L2Microprice",
        "L2MicropriceConfig",
    ),
    "l2_vpin_toxicity": (
        "strategies.l2.vpin_toxicity",
        "L2VpinToxicity",
        "L2VpinToxicityConfig",
    ),
}


def get_strategy_names() -> list[str]:
    return list(_STRATEGY_REGISTRY)


def get_strategy_class(name: str) -> tuple[type, type]:
    entry = _STRATEGY_REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"Unknown strategy: {name}")

    module_name, strategy_cls_name, config_cls_name = entry
    module = importlib.import_module(f".{module_name}", package=__package__)
    return (
        getattr(module, strategy_cls_name),
        getattr(module, config_cls_name),
    )


def make_instrument_id(venue_name: str, symbol_str: str) -> InstrumentId:
    """Build the conventioned instrument id for a *symbol_str* on a venue.

    ``BTC/USDT:USDT`` (or ``BTCUSDT:USDT``) -> ``BTCUSDT:USDT-PERP`` on the
    given venue. Single source of truth for the id rule, shared by
    ``make_perpetual`` and the portfolio strategy base so their leg ids
    always match the runner's instruments.
    """
    raw = symbol_str.replace("/", "")
    return InstrumentId(symbol=Symbol(f"{raw}-PERP"), venue=Venue(venue_name))


def make_perpetual(
    venue_name: str,
    symbol_str: str,
    maker_fee: Decimal = Decimal("0.0"),
    taker_fee: Decimal = Decimal("0.0"),
    base_currency=BTC,
    quote_currency=USDT,
    settlement_currency=USDT,
) -> CryptoPerpetual:
    inst_id = make_instrument_id(venue_name, symbol_str)
    raw = symbol_str.replace("/", "")
    return CryptoPerpetual(
        instrument_id=inst_id,
        raw_symbol=Symbol(raw),
        base_currency=base_currency,
        quote_currency=quote_currency,
        settlement_currency=settlement_currency,
        is_inverse=False,
        price_precision=1,
        price_increment=Price.from_str("0.1"),
        size_precision=3,
        size_increment=Quantity.from_str("0.001"),
        max_quantity=Quantity.from_str("1000.000"),
        min_quantity=Quantity.from_str("0.001"),
        max_notional=None,
        min_notional=Money(10.00, settlement_currency),
        max_price=Price.from_str("999999.0"),
        min_price=Price.from_str("0.1"),
        margin_init=Decimal("0.0500"),
        margin_maint=Decimal("0.0250"),
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        ts_event=0,
        ts_init=0,
    )


_INTERVAL_MAP = {
    "1m": "1-MINUTE",
    "3m": "3-MINUTE",
    "5m": "5-MINUTE",
    "15m": "15-MINUTE",
    "30m": "30-MINUTE",
    "1h": "1-HOUR",
    "2h": "2-HOUR",
    "4h": "4-HOUR",
    "6h": "6-HOUR",
    "8h": "8-HOUR",
    "12h": "12-HOUR",
    "1d": "1-DAY",
    "1w": "1-WEEK",
}


def parse_interval(interval: str) -> str:
    result = _INTERVAL_MAP.get(interval)
    if result is None:
        raise ValueError(f"Unknown interval: {interval}")
    return result


_INTERVAL_DELTAS = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}


def interval_delta(interval: str) -> timedelta:
    """Wall-clock duration of a single bar for the given interval key."""
    delta = _INTERVAL_DELTAS.get(interval)
    if delta is None:
        raise ValueError(f"Unknown interval: {interval}")
    return delta
