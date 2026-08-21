import importlib
from decimal import Decimal

from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity

_STRATEGY_REGISTRY = {
    "bitcoin_intraday_momentum": (
        "strategies.bitcoin_intraday_momentum",
        "BitcoinIntradayMomentum",
        "BitcoinIntradayMomentumConfig",
    ),
    "glucksmann": (
        "strategies.glucksmann",
        "GlucksmannStrategy",
        "GlucksmannConfig",
    ),
    "orb": (
        "strategies.orb",
        "ORBStrategy",
        "ORBConfig",
    ),
    "overnight_drift": (
        "strategies.overnight_drift",
        "OvernightDrift",
        "OvernightDriftConfig",
    ),
    "l2_order_imbalance": (
        "strategies.l2_order_imbalance",
        "L2OrderImbalance",
        "L2OrderImbalanceConfig",
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


def make_perpetual(
    venue_name: str,
    symbol_str: str,
    maker_fee: Decimal = Decimal("0.0"),
    taker_fee: Decimal = Decimal("0.0"),
    base_currency=BTC,
    quote_currency=USDT,
    settlement_currency=USDT,
) -> CryptoPerpetual:
    raw = symbol_str.replace("/", "")
    inst_id = InstrumentId(symbol=Symbol(f"{raw}-PERP"), venue=Venue(venue_name))
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
