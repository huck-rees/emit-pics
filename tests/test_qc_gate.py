"""Unit tests for the manual QC gate (opt-out scene filtering + review gate)."""

import pandas as pd
import pytest

from emit_pics.qc_gate import (
    QC_SHEET_COLS, QCReviewRequired, _keep, _truthy, apply_qc_filter,
)


def _write_sheet(path, rows, *, reviewed=True):
    """Write a minimal QC worksheet; `reviewed` sets only the top row's flag."""
    df = pd.DataFrame(rows)
    for c in QC_SHEET_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[QC_SHEET_COLS]
    df.loc[df.index[0], "reviewed"] = "TRUE" if reviewed else "FALSE"
    df.to_csv(path, index=False)


def _scenes():
    return pd.DataFrame({"timestamp": ["20230101T100000", "20230601T100000",
                                       "20231201T100000"]})


# ------------------------------------------------------------------ token parsing
def test_keep_is_opt_out():
    assert _keep("TRUE") and _keep("") and _keep("yes")          # kept
    assert not _keep("FALSE") and not _keep("X") and not _keep("no")


def test_truthy_tokens():
    assert _truthy("TRUE") and _truthy("1")
    assert not _truthy("false") and not _truthy("")


# ----------------------------------------------------------------- apply_qc_filter
def test_apply_qc_filter_drops_rejected_scene(tmp_path):
    qc = tmp_path / "qc.csv"
    _write_sheet(qc, [
        {"scene_ts": "20230101T100000", "keep": "TRUE", "reason": ""},
        {"scene_ts": "20230601T100000", "keep": "FALSE", "reason": "cloud"},
        {"scene_ts": "20231201T100000", "keep": "TRUE", "reason": ""},
    ], reviewed=True)

    filtered, report = apply_qc_filter(_scenes(), qc, require_review=True)

    assert report["n_before"] == 3
    assert report["n_after"] == 2
    assert report["n_excluded"] == 1
    assert report["reasons"] == {"cloud": 1}
    assert "20230601T100000" not in filtered["timestamp"].tolist()


def test_apply_qc_filter_unreviewed_sheet_raises(tmp_path):
    qc = tmp_path / "qc.csv"
    _write_sheet(qc, [{"scene_ts": "20230101T100000", "keep": "TRUE"}],
                 reviewed=False)
    with pytest.raises(QCReviewRequired):
        apply_qc_filter(_scenes(), qc, require_review=True)


def test_apply_qc_filter_missing_sheet_raises(tmp_path):
    with pytest.raises(QCReviewRequired):
        apply_qc_filter(_scenes(), tmp_path / "nope.csv", require_review=True)


def test_apply_qc_filter_bypass_when_review_not_required(tmp_path):
    filtered, report = apply_qc_filter(_scenes(), tmp_path / "nope.csv",
                                       require_review=False)
    assert report["n_excluded"] == 0
    assert len(filtered) == 3
