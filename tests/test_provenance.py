"""Tests for provenance sidecars."""

import json
from datetime import datetime, timezone

from emit_pics import provenance


def test_write_sidecar_names_and_embeds_params(tmp_path):
    out = tmp_path / "Libya_4_10_7_1_delta.csv"
    out.write_text("dummy", encoding="utf-8")

    sidecar = provenance.write_sidecar(out, {"site": "Libya_4", "sza_tol": 2.0})

    # Sidecar is <output>.json (full name + .json, so .csv/.png never collide).
    assert sidecar.name == "Libya_4_10_7_1_delta.csv.json"
    assert sidecar.exists()

    rec = json.loads(sidecar.read_text(encoding="utf-8"))
    assert rec["run_config"] == {"site": "Libya_4", "sza_tol": 2.0}
    for key in ("created_utc", "script", "git", "python", "packages"):
        assert key in rec
    assert set(rec["git"]) == {"commit", "branch", "dirty"}


def test_same_stem_csv_and_png_get_distinct_sidecars(tmp_path):
    csv = tmp_path / "Libya_4_10_7_1_matched_pairs.csv"
    png = tmp_path / "Libya_4_10_7_1_matched_pairs.png"
    csv.write_text("x", encoding="utf-8")
    png.write_text("x", encoding="utf-8")

    s_csv = provenance.write_sidecar(csv, {})
    s_png = provenance.write_sidecar(png, {})

    assert s_csv.name == "Libya_4_10_7_1_matched_pairs.csv.json"
    assert s_png.name == "Libya_4_10_7_1_matched_pairs.png.json"
    assert s_csv != s_png


def test_build_manifest_timestamp_is_injectable():
    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    man = provenance.build_manifest({"a": 1}, script="run_x.py", timestamp=ts)
    assert man["created_utc"] == "2026-01-02T03:04:05+00:00"
    assert man["script"] == "run_x.py"
    assert man["run_config"] == {"a": 1}
    assert isinstance(man["packages"], dict)


def test_packages_include_core_stack():
    man = provenance.build_manifest(timestamp=datetime.now(timezone.utc))
    # These are always installed in the env; versions must be captured (not None).
    for pkg in ("numpy", "pandas", "scipy"):
        assert man["packages"].get(pkg) is not None
