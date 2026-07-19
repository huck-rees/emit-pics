"""Tests for the run_config.yaml loader."""

import pytest

from emit_pics.run_config import load_config


def test_load_config_reads_values_and_fills_defaults(tmp_path):
    p = tmp_path / "run_config.yaml"
    p.write_text(
        "site: Libya_4\n"
        'patch: "10_7_1"\n'
        'patch_size: "10"\n'
        "matching: {sza_tol: 3.0, vza_tol: 4.0}\n"
        "drift: {alpha: 0.1, use_aggregated: false}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)

    # explicit values
    assert cfg.site == "Libya_4"
    assert cfg.patch == "10_7_1"
    assert cfg.patch_size == "10"
    assert cfg.sza_tol == 3.0
    assert cfg.vza_tol == 4.0
    assert cfg.alpha == 0.1
    assert cfg.use_aggregated is False

    # defaults for omitted sections
    assert cfg.rad_short_name == "EMITL1BRAD"
    assert cfg.require_qc_review is True
    assert cfg.spectra_stat == "median"


def test_patch_size_coerced_to_string(tmp_path):
    p = tmp_path / "rc.yaml"
    p.write_text("site: Libya_4\npatch: 10_7_1\npatch_size: 10\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.patch_size == "10"     # int in YAML -> str token
    assert isinstance(cfg.patch_size, str)


def test_missing_required_key_raises(tmp_path):
    p = tmp_path / "rc.yaml"
    p.write_text("patch: x\npatch_size: '10'\n", encoding="utf-8")   # no site
    with pytest.raises(ValueError):
        load_config(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_as_dict_roundtrips_fields(tmp_path):
    p = tmp_path / "rc.yaml"
    p.write_text("site: Libya_4\npatch: '10_7_1'\npatch_size: '10'\n", encoding="utf-8")
    d = load_config(p).as_dict()
    assert d["site"] == "Libya_4"
    assert d["patch"] == "10_7_1"
    assert "sza_tol" in d          # flat dict for provenance sidecars
