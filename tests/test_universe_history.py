"""Unit tests for ``sbt.strategies.base.UniverseHistory``.

Exercises the bounded rolling history buffer in isolation from the
backtest engine: every public method on every leg, plus invariants
(prune correctness, empty-leg behavior, formation/dollar-volume parity
with the per-bar queries).
"""

from decimal import Decimal

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
import pytest

from sbt.strategies.base import UniverseHistory

_BT = BarType.from_str("BTCUSDT-PERP.TESTEX-1-HOUR-LAST-EXTERNAL")
_ETH_BT = BarType.from_str("ETHUSDT-PERP.TESTEX-1-HOUR-LAST-EXTERNAL")
_BTC = InstrumentId.from_str("BTCUSDT-PERP.TESTEX")
_ETH = InstrumentId.from_str("ETHUSDT-PERP.TESTEX")


def _bar(iid: InstrumentId, ts: int, close: float, high: float, volume: float) -> Bar:
    return Bar(
        bar_type=_BT if iid == _BTC else _ETH_BT,
        open=Price.from_str(str(close)),
        high=Price.from_str(str(high)),
        low=Price.from_str(str(close)),
        close=Price.from_str(str(close)),
        volume=Quantity.from_str(str(volume)),
        ts_event=ts,
        ts_init=ts,
    )


@pytest.fixture
def hist() -> UniverseHistory:
    return UniverseHistory([_BTC, _ETH])


def _fill(hist: UniverseHistory) -> None:
    """5 daily bars per leg, BTC trends up 100->104, ETH trends up 100->108."""
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000  # arbitrary ns epoch
    for d in range(5):
        hist.record(_BTC, _bar(_BTC, base + d * day, 100 + d, 102 + d, 50 + 10 * d))
        hist.record(_ETH, _bar(_ETH, base + d * day, 100 + 2 * d, 103 + 2 * d, 20 + 5 * d))


# ---------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------


def test_record_appends_tuples_in_order(hist: UniverseHistory) -> None:
    _fill(hist)
    assert len(hist._data[_BTC]) == 5
    assert len(hist._data[_ETH]) == 5
    # Strictly ascending timestamps within a leg.
    btc_ts = [row[0] for row in hist._data[_BTC]]
    assert btc_ts == sorted(btc_ts)
    assert btc_ts == list(dict.fromkeys(btc_ts))


def test_record_handles_missing_volume() -> None:
    """A bar with no volume attribute stores 0.0 instead of raising."""
    h = UniverseHistory([_BTC])

    class NoVolBar:
        bar_type = _BT
        ts_event = 1
        close = Price.from_str("100.0")
        high = Price.from_str("101.0")
        # No `volume` attribute on purpose.

    h.record(_BTC, NoVolBar())  # type: ignore[arg-type]
    assert h._data[_BTC][0][3] == 0.0


def test_instrument_ids_property(hist: UniverseHistory) -> None:
    assert set(hist.instrument_ids) == {_BTC, _ETH}


# ---------------------------------------------------------------------
# Single-leg queries
# ---------------------------------------------------------------------


def test_close_at_or_before_returns_last_match(hist: UniverseHistory) -> None:
    _fill(hist)
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    # 3rd bar's close (102) at the start of day 3.
    assert hist.close_at_or_before(_BTC, base + 2 * day + 1) == 102.0
    # Right at the 3rd bar's exact ts.
    assert hist.close_at_or_before(_BTC, base + 2 * day) == 102.0
    # Before the very first bar.
    assert hist.close_at_or_before(_BTC, base - 1) is None
    # Past the last bar: returns the last close.
    assert hist.close_at_or_before(_BTC, base + 999 * day) == 104.0
    # Unknown instrument.
    unknown = InstrumentId.from_str("SOLUSDT-PERP.TESTEX")
    assert hist.close_at_or_before(unknown, base + 999 * day) is None


def test_max_high_finds_peak_in_window(hist: UniverseHistory) -> None:
    _fill(hist)
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    # Window covers the last 2 days; BTC highs there are 105, 106 -> 106.
    assert hist.max_high(_BTC, base + 2 * day, base + 5 * day) == 106.0
    # Empty window past the data.
    assert hist.max_high(_BTC, base + 99 * day, base + 100 * day) is None
    # Window entirely before the data.
    assert hist.max_high(_BTC, base - 99 * day, base - 1) is None


def test_dollar_volume_sums_close_times_volume(hist: UniverseHistory) -> None:
    _fill(hist)
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    # BTC: closes 100..104, vols 50,60,70,80,90 -> sum(close*vol) = 100*50+101*60+102*70+103*80+104*90.
    expected_btc = (
        100 * 50 + 101 * 60 + 102 * 70 + 103 * 80 + 104 * 90
    )
    assert hist.dollar_volume(_BTC, base - 1, base + 999 * day) == expected_btc
    # Empty window.
    assert hist.dollar_volume(_BTC, base + 99 * day, base + 100 * day) == 0.0


def test_window_returns_tuples_in_window(hist: UniverseHistory) -> None:
    _fill(hist)
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    rows = hist.window(_BTC, base + day, base + 3 * day)
    # (lo, hi] is exclusive on lo, inclusive on hi -> days 2 and 3 (closes 102, 103).
    assert len(rows) == 2
    assert [r[1] for r in rows] == [102.0, 103.0]


# ---------------------------------------------------------------------
# Universe-wide queries
# ---------------------------------------------------------------------


def test_formation_returns_matches_per_leg_arithmetic(hist: UniverseHistory) -> None:
    _fill(hist)
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    rets = hist.formation_returns(end_ns=base + 4 * day, start_ns=base + 1 * day)
    # BTC: c_end=104, c_start=101 -> 104/101 - 1
    # ETH: c_end=108, c_start=102 -> 108/102 - 1
    assert rets[_BTC] == pytest.approx(104 / 101 - 1)
    assert rets[_ETH] == pytest.approx(108 / 102 - 1)


def test_formation_returns_skips_legs_missing_an_endpoint(hist: UniverseHistory) -> None:
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    # Only one bar exists; start_ns is before it, so close_at_or_before(start)=None.
    hist.record(_BTC, _bar(_BTC, base, 100.0, 101.0, 50.0))
    rets = hist.formation_returns(end_ns=base, start_ns=base - day)
    assert rets == {}


def test_formation_returns_yields_zero_when_start_and_end_collapse(hist: UniverseHistory) -> None:
    """When start_ns and end_ns fall on the same bar, the return is 0.0.

    Documents the *exact* behavior of close_at_or_before sampling at the
    same instant — not a bug, but a sharp edge that strategies must avoid
    by picking end_ns strictly greater than the latest bar.
    """
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    hist.record(_BTC, _bar(_BTC, base, 100.0, 101.0, 50.0))
    rets = hist.formation_returns(end_ns=base, start_ns=base)
    assert rets == {_BTC: 0.0}


def test_formation_returns_skips_zero_or_negative_start(hist: UniverseHistory) -> None:
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    hist.record(_BTC, _bar(_BTC, base, 0.0, 1.0, 1.0))
    hist.record(_BTC, _bar(_BTC, base + day, 100.0, 101.0, 1.0))
    rets = hist.formation_returns(end_ns=base + day, start_ns=base)
    assert rets == {}


def test_dollar_volumes_matches_per_leg_dollar_volume(hist: UniverseHistory) -> None:
    _fill(hist)
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    dvs = hist.dollar_volumes(lo_ns=base - 1, hi_ns=base + 999 * day)
    assert dvs == {
        iid: hist.dollar_volume(iid, base - 1, base + 999 * day)
        for iid in (_BTC, _ETH)
    }


# ---------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------


def test_prune_drops_old_rows_from_every_leg(hist: UniverseHistory) -> None:
    _fill(hist)
    day = 86_400_000_000_000
    base = 1_700_000_000 * 1_000_000_000
    # Prune everything older than or equal to day-2 -> keep last 2 bars.
    hist.prune(before_ns=base + 2 * day)
    assert len(hist._data[_BTC]) == 2
    assert len(hist._data[_ETH]) == 2
    # Kept rows are the last two (days 3 and 4).
    assert [r[1] for r in hist._data[_BTC]] == [103.0, 104.0]


def test_prune_with_empty_buffer_is_noop(hist: UniverseHistory) -> None:
    hist.prune(before_ns=0)
    assert hist._data[_BTC] == []
    assert hist._data[_ETH] == []


# ---------------------------------------------------------------------
# Initial-state guarantees
# ---------------------------------------------------------------------


def test_new_history_has_empty_lists_for_every_leg() -> None:
    h = UniverseHistory([_BTC, _ETH])
    assert h._data == {_BTC: [], _ETH: []}
