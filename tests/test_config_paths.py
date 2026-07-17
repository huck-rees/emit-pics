"""Tests for the canonical site-id token and the library-dedup re-exports."""

import pandas as pd
import pytest

from emit_pics.config import canonical_token, load_sites


# ------------------------------------------------------------- canonical_token
def test_canonical_token_accepts_any_spelling():
    for spelling in ["libya_4", "Libya 4", "Libya_4", "LIBYA 4", " libya 4 "]:
        assert canonical_token(spelling) == "Libya_4"


def test_canonical_token_matches_sitename_for_every_site():
    # The token must equal site_name.replace(" ", "_") for every configured
    # site — this is exactly the historical filename convention.
    for slug, info in load_sites().items():
        expected = info["site_name"].replace(" ", "_")
        assert canonical_token(slug) == expected


def test_canonical_token_unknown_site_raises():
    with pytest.raises(KeyError):
        canonical_token("Atlantis")


# --------------------------------------------------- load_patch_dict / split_products
def test_helpers_are_single_definition_reexported():
    # After dedup, both import paths must resolve to the SAME function object.
    from emit_pics.extract_patch_stats import load_patch_dict as lpd_e
    from emit_pics.extract_patch_stats import split_products as sp_e
    from emit_pics.id_matched_pairs import load_patch_dict as lpd_i
    from emit_pics.id_matched_pairs import split_products as sp_i
    assert lpd_e is lpd_i
    assert sp_e is sp_i


def test_split_products_groups_by_product():
    from emit_pics.id_matched_pairs import split_products
    df = pd.DataFrame({
        "patch_id": ["10_0_0", "10_0_0", "10_0_0", "20_0_0"],
        "product": ["L1B_RAD", "L2A_RFL", "L2A_RFL", "L1B_RAD"],
        "granule_ur": ["radA", "rflA", "rflB", "radX"],
    })
    rad, rfl = split_products(df, "10_0_0")
    assert rad == ["radA"]
    assert rfl == ["rflA", "rflB"]


def test_load_patch_dict_reads_tab_separated(tmp_path):
    from emit_pics.extract_patch_stats import load_patch_dict
    p = tmp_path / "patch_EMIT_dict_Libya_4.csv"
    p.write_text("patch_id\tproduct\tgranule_ur\n10_0_0\tL1B_RAD\tradA\n",
                 encoding="utf-8")
    df = load_patch_dict(p, sep="\t")
    assert list(df.columns) == ["patch_id", "product", "granule_ur"]
    assert df.iloc[0]["granule_ur"] == "radA"
