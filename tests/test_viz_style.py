"""Tests for the figure theme registry (emit_pics.viz.style)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from emit_pics.viz import style


def test_both_themes_exist():
    assert set(style.THEMES) == {"manuscript", "ppt"}


def test_manuscript_double_is_elsevier_185mm():
    with style.theme("manuscript"):
        assert style.figwidth("double") == pytest.approx(185 / 25.4, abs=1e-6)
        assert style.figwidth("single") == pytest.approx(90 / 25.4, abs=1e-6)


def test_theme_sets_rcparams_and_restores_after():
    before = mpl.rcParams["font.size"]
    with style.theme("ppt"):
        assert mpl.rcParams["font.size"] == 15        # slide-scale lettering
        assert mpl.rcParams["pdf.fonttype"] == 42     # base rc still applied
    assert mpl.rcParams["font.size"] == before        # scoped — no leak


def test_manuscript_font_meets_min_and_ppt_is_larger():
    with style.theme("manuscript"):
        assert mpl.rcParams["xtick.labelsize"] >= 7   # Elsevier 7 pt floor
        m = style.scale()
    with style.theme("ppt"):
        p = style.scale()
    assert p > m                                      # ppt scales non-rc sizes up


def test_active_raises_outside_theme():
    with pytest.raises(RuntimeError):
        style.active()
    with pytest.raises(RuntimeError):
        style.figwidth("double")


def test_unknown_theme_and_size_raise():
    with pytest.raises(ValueError):
        with style.theme("poster"):
            pass
    with style.theme("manuscript"):
        with pytest.raises(ValueError):
            style.figwidth("gigantic")


def test_show_title_differs_by_theme():
    with style.theme("manuscript"):
        assert style.show_title() is False    # caption is in LaTeX
    with style.theme("ppt"):
        assert style.show_title() is True


def test_save_writes_theme_formats(tmp_path):
    with style.theme("manuscript"):
        fig = plt.figure(figsize=(2, 2))
        fig.add_subplot(111).plot([0, 1], [0, 1])
        out = style.save(fig, "t", out_dir=tmp_path)
    suffixes = sorted(p.suffix for p in out)
    assert suffixes == [".pdf", ".svg"]
    assert all(p.exists() for p in out)


def test_site_styles_are_stable_and_distinct():
    s = style.site_styles(["B", "A", "C"])
    assert set(s) == {"A", "B", "C"}
    # sorted assignment -> A gets the first palette entry deterministically
    assert s["A"]["color"] == style._SITE_COLORS[0]
    assert s["A"]["marker"] != s["B"]["marker"]
