"""Shared fixtures: deterministic synthetic bars and run configs.

The synthetic frame rises within every day (strictly increasing hourly
prices), which makes ORB-style breakout entries deterministic once ATR
warm-up completes. No filesystem access is needed anywhere.
"""

from decimal import Decimal

import pandas as pd
import pytest

from sbt.core.config import RunConfig

BASE_TS = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def make_synthetic_bars(
    days: int = 20, hours_per_day: int = 24, base: float = 100.0, step: float = 0.5
) -> pd.DataFrame:
    """Hourly OHLCV rising steadily inside each day, resetting at midnight."""
    rows = []
    for d in range(days):
        for h in range(hours_per_day):
            ts = BASE_TS + pd.Timedelta(days=d, hours=h)
            p = base + h * step
            rows.append(
                {
                    "timestamp": ts,
                    "open": round(p - 0.05, 2),
                    "high": round(p + 0.10, 2),
                    "low": round(p - 0.10, 2),
                    "close": round(p + 0.05, 2),
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def make_funding_frame(bars: pd.DataFrame, rate: float = 0.0001) -> pd.DataFrame:
    """Hourly funding updates spanning the bars' timestamp range."""
    return pd.DataFrame({"timestamp": bars["timestamp"], "funding_rate": rate})


@pytest.fixture
def make_bars():
    return make_synthetic_bars


@pytest.fixture
def make_funding():
    return make_funding_frame


@pytest.fixture
def synthetic_bars() -> pd.DataFrame:
    return make_synthetic_bars()


@pytest.fixture
def orb_config() -> RunConfig:
    return RunConfig(
        exchange="TESTEX",
        symbol="BTC/USDT:USDT",
        interval="1h",
        strategy_name="orb",
        strategy_params={"orb_period": 4, "atr_period": 3},
        start="2024-01-01",
        end="2024-01-20",
        open_report=False,
    )
