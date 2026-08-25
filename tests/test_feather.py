"""Tests for the feather filename contract owner (sbt.core.feather).

The downloader and the runner must never disagree about
``{exchange}_{symbol}_{tag}_{YYYYMMDD}[_{YYYYMMDD}].feather``; these pin
each half of the contract through the module's public interface.
"""

import pandas as pd

from sbt.core.feather import (
    actual_range_name,
    feather_path,
    find_feather,
    parse_range,
)


def test_parse_range_valid_and_invalid():
    assert parse_range("/x/hyperliquid_BTCUSDT_1h_20240101_20240301.feather") == (
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-03-01 23:59:59", tz="UTC"),
    )
    assert parse_range("/x/unprefixed.feather") is None


def test_feather_path_round_trips_through_parse_range():
    p = feather_path(
        "hyperliquid",
        "BTC/USDC:USDC",
        "1h",
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-03-01", tz="UTC"),
    )
    assert p == "data/hyperliquid_BTCUSDC:USDC_1h_20240101_20240301.feather"
    start, end = parse_range(p)
    assert start == pd.Timestamp("2024-01-01", tz="UTC")
    assert end == pd.Timestamp("2024-03-01 23:59:59", tz="UTC")


def _touch(dir, name):
    path = dir / name
    path.touch()
    return str(path)


def test_find_feather_prefers_coverage_then_overlap(tmp_path):
    partial = _touch(tmp_path, "ex_SYM_1h_20240101_20240201.feather")
    middle = _touch(tmp_path, "ex_SYM_1h_20240201_20240301.feather")
    covering = _touch(tmp_path, "ex_SYM_1h_20240101_20240401.feather")

    chosen = find_feather(
        "ex", "SYM", "1h", [str(tmp_path)], start="2024-02-15", end="2024-03-15"
    )

    assert chosen == covering
    assert partial not in (chosen,) and middle not in (chosen,)


def test_find_feather_prefixed_beats_bare_even_when_worse(tmp_path):
    prefixed = _touch(tmp_path, "ex_SYM_1h_20240101_20240201.feather")
    _touch(tmp_path, "SYM_1h_20240101_20240401.feather")

    chosen = find_feather(
        "ex", "SYM", "1h", [str(tmp_path)], start="2024-02-15", end="2024-03-15"
    )

    assert chosen == prefixed


def test_find_feather_unique_bare_fallback(tmp_path):
    bare = _touch(tmp_path, "SYM_1h_20240101_20240401.feather")

    assert find_feather("ex", "SYM", "1h", [str(tmp_path)]) == bare

    # a second bare match for the same symbol+tag is ambiguous -> no pick
    _touch(tmp_path, "SYM_1h_20240501_20240601.feather")
    assert find_feather("ex", "SYM", "1h", [str(tmp_path)]) is None


def test_find_feather_no_match_returns_none(tmp_path):
    assert find_feather("ex", "SYM", "1h", [str(tmp_path)]) is None


def test_find_feather_matches_funding_tag(tmp_path):
    funding = _touch(tmp_path, "ex_SYM_funding_20240101_20240601.feather")

    assert find_feather("ex", "SYM", "funding", [str(tmp_path)]) == funding


def test_actual_range_name_heals_stale_suffix():
    stale = "data/ex_SYM_1h_20240101_20240301.feather"
    healed = actual_range_name(
        stale,
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-06-01", tz="UTC"),
    )
    assert healed == "ex_SYM_1h_20240101_20240601.feather"


def test_actual_range_name_leaves_custom_names_alone():
    assert actual_range_name(
        "my_custom_name.feather",
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-06-01", tz="UTC"),
    ) is None


def test_actual_range_name_idempotent_when_matching():
    exact = "ex_SYM_1h_20240101_20240601.feather"
    assert (
        actual_range_name(
            f"data/{exact}",
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-06-01", tz="UTC"),
        )
        == exact
    )
