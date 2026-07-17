"""Unit tests for matched-pair detection and timestamp parsing."""

from datetime import datetime

import pandas as pd
import pytest

from emit_pics.id_matched_pairs import (
    PAIR_CSV_COLS, acq_datetime, acq_timestamp, find_matched_pairs,
)


def _scenes():
    return pd.DataFrame([
        {"scene_id": "A", "datetime": datetime(2023, 1, 1),
         "sza_mean": 30.0, "vza_mean": 5.0, "has_rfl": True},
        {"scene_id": "B", "datetime": datetime(2023, 6, 1),
         "sza_mean": 31.0, "vza_mean": 6.0, "has_rfl": True},
        {"scene_id": "C", "datetime": datetime(2023, 12, 1),
         "sza_mean": 40.0, "vza_mean": 20.0, "has_rfl": False},
    ])


def test_find_matched_pairs_flags_counts_and_deltas():
    out = find_matched_pairs(_scenes(), sza_tol=2.0, vza_tol=5.0)

    assert len(out) == 3                       # 3 unordered pairs of 3 scenes
    for c in PAIR_CSV_COLS:
        assert c in out.columns
    # pair order is (A,B), (A,C), (B,C)
    assert out["matched_pair"].tolist() == [True, False, False]

    ab = out.iloc[0]
    assert ab["delta_SZA"] == 1.0
    assert ab["delta_VZA"] == 1.0
    assert ab["delta_time"] == 151.0           # 2023-01-01 -> 2023-06-01


def test_find_matched_pairs_tolerance_is_inclusive():
    scenes = pd.DataFrame([
        {"scene_id": "A", "datetime": datetime(2023, 1, 1),
         "sza_mean": 30.0, "vza_mean": 5.0, "has_rfl": True},
        {"scene_id": "B", "datetime": datetime(2023, 2, 1),
         "sza_mean": 32.0, "vza_mean": 10.0, "has_rfl": True},
    ])
    # |dSZA| = 2.0 (== tol) and |dVZA| = 5.0 (== tol) -> matched (<=)
    out = find_matched_pairs(scenes, sza_tol=2.0, vza_tol=5.0)
    assert out["matched_pair"].tolist() == [True]


def test_find_matched_pairs_below_only_one_tolerance_not_matched():
    scenes = pd.DataFrame([
        {"scene_id": "A", "datetime": datetime(2023, 1, 1),
         "sza_mean": 30.0, "vza_mean": 5.0, "has_rfl": True},
        {"scene_id": "B", "datetime": datetime(2023, 2, 1),
         "sza_mean": 31.0, "vza_mean": 20.0, "has_rfl": True},  # VZA off by 15
    ])
    out = find_matched_pairs(scenes, sza_tol=2.0, vza_tol=5.0)
    assert out["matched_pair"].tolist() == [False]


def test_acq_timestamp_and_datetime():
    ur = "EMIT_L1B_RAD_001_20230101T100000_2300107_006"
    assert acq_timestamp(ur) == "20230101T100000"
    assert acq_datetime(ur) == datetime(2023, 1, 1, 10, 0, 0)


def test_acq_timestamp_missing_raises():
    with pytest.raises(ValueError):
        acq_timestamp("no_timestamp_here")
