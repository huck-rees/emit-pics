"""Tests for the run_config.yaml loader."""

import pytest

from emit_pics.run_config import load_config


def test_load_config_reads_values_and_fills_defaults(tmp_path):
    p = tmp_path / "run_config.yaml"
    p.write_text(
        "site: Libya_4\n"
        'patch: "10_7_1"\n'
        'patch_size: "10"\n'
        "matching: {sza_tol: 3.0, vza_tol: 4.0, min_pair_days: 90}\n"
        "drift: {alpha: 0.1, n_bootstrap: 200, seed: 7}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)

    # explicit values
    assert cfg.site == "Libya_4"
    assert cfg.patch == "10_7_1"
    assert cfg.patch_size == "10"
    assert cfg.sza_tol == 3.0
    assert cfg.vza_tol == 4.0
    assert cfg.min_pair_days == 90.0
    assert cfg.alpha == 0.1
    assert cfg.drift_n_bootstrap == 200
    assert cfg.drift_seed == 7

    # defaults for omitted sections
    assert cfg.rad_short_name == "EMITL1BRAD"
    assert cfg.require_qc_review is True
    assert cfg.spectra_stat == "median"
    assert cfg.earth_sun_normalize is True     # radiometry block omitted -> default on
    assert cfg.e0_method == "table"            # default E0 provider (ephemeris table)
    assert cfg.output_variant == ""            # corrected run -> canonical flat layout


def test_earth_sun_normalize_toggle_read(tmp_path):
    p = tmp_path / "rc.yaml"
    p.write_text(
        "site: Libya_4\npatch: '10_7_1'\npatch_size: '10'\n"
        "radiometry: {earth_sun_normalize: false, e0_method: table}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.earth_sun_normalize is False
    assert cfg.e0_method == "table"
    # Correction off -> derived outputs nest under the default variant subfolder.
    assert cfg.output_variant == "no_earthsun"


def test_output_variant_custom_label_when_correction_off(tmp_path):
    p = tmp_path / "rc.yaml"
    p.write_text(
        "site: Libya_4\npatch: '10_7_1'\npatch_size: '10'\n"
        "radiometry: {earth_sun_normalize: false, variant_label: raw_rad}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.output_variant == "raw_rad"
    assert cfg.as_dict()["output_variant"] == "raw_rad"   # rides along in provenance


def test_output_variant_ignored_when_correction_on(tmp_path):
    # The corrected run is ALWAYS the canonical baseline, even if a label is set.
    p = tmp_path / "rc.yaml"
    p.write_text(
        "site: Libya_4\npatch: '10_7_1'\npatch_size: '10'\n"
        "radiometry: {earth_sun_normalize: true, variant_label: raw_rad}\n",
        encoding="utf-8",
    )
    assert load_config(p).output_variant == ""


def test_matching_and_drift_defaults(tmp_path):
    p = tmp_path / "rc.yaml"
    p.write_text("site: Libya_4\npatch: '10_7_1'\npatch_size: '10'\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.min_pair_days == 0.0            # off by default (backward compatible)
    assert cfg.drift_n_bootstrap == 1000
    assert cfg.drift_seed == 0


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


# ---------------------------------------------------------------------- BRDF block
def test_brdf_defaults_when_block_absent(tmp_path):
    p = tmp_path / "rc.yaml"
    p.write_text("site: Libya_4\npatch: '10_7_1'\npatch_size: '10'\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.brdf_site_patches == {}
    assert cfg.brdf_exclude_sites == []
    assert cfg.brdf_label == ""
    assert cfg.brdf_target_wl == 858.0
    assert cfg.brdf_min_scenes == 4
    assert cfg.brdf_n_bootstrap == 1000
    assert cfg.brdf_per_site_boot == 500
    assert cfg.brdf_spectrum_n_boot == 300
    assert cfg.brdf_gamma_cap == 500.0


def test_brdf_roster_drops_null_patches_and_reads_values(tmp_path):
    p = tmp_path / "rc.yaml"
    p.write_text(
        "site: Libya_4\n"
        'patch: "10_7_1"\n'
        'patch_size: "10"\n'
        "brdf:\n"
        "  site_patches: {Libya_4: '10_7_1', Algeria_5: null, Libya_1: '5_3_2'}\n"
        "  exclude_sites: [Mauritania_2]\n"
        "  target_wl: 660.0\n"
        "  min_scenes: 6\n"
        "  spectrum_n_boot: 50\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    # null-patch sites are dropped; only mapped sites survive, coerced to str.
    assert cfg.brdf_site_patches == {"Libya_4": "10_7_1", "Libya_1": "5_3_2"}
    assert cfg.brdf_exclude_sites == ["Mauritania_2"]
    assert cfg.brdf_target_wl == 660.0
    assert cfg.brdf_min_scenes == 6
    assert cfg.brdf_label == ""            # absent -> flat baseline
    assert cfg.brdf_spectrum_n_boot == 50
    # BRDF fields ride along in the flat provenance dict.
    assert load_config(p).as_dict()["brdf_target_wl"] == 660.0


def test_brdf_label_read_and_in_provenance(tmp_path):
    p = tmp_path / "rc.yaml"
    p.write_text(
        "site: Libya_4\npatch: '20_3_2'\npatch_size: '20'\n"
        "brdf: {site_patches: {Libya_4: '20_3_2'}, label: '20km'}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.brdf_label == "20km"
    assert cfg.as_dict()["brdf_label"] == "20km"   # recorded in the sidecar
