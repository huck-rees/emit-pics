"""Single source of truth for every data/ and figures/ path in the pipeline.

Every input and output file name is built here, from ``(site, patch/size, ...)``
— so the ``data/`` + ``figures/`` layout and the filename templates live in ONE
place instead of being copied into each script's header. The site is turned into
its canonical filename token via :func:`emit_pics.config.canonical_token`, so any
accepted spelling (``libya_4`` / ``Libya 4`` / ``Libya_4``) resolves to the same
on-disk name, matching the historical convention exactly.

All paths are repo-relative, so a fresh clone works with no per-machine edits.
"""

from functools import lru_cache
from pathlib import Path

from emit_pics.config import canonical_token

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
FIGURES = REPO_ROOT / "figures"

# ------------------------------------------------------------- raw-input directories
# NEVER varied — the extracted per-scene stats/cubes/geometry are the shared raw
# inputs and are identical regardless of the Earth–Sun (radiometry) variant. The
# correction is applied at READ time in the loaders, so these files stay put.
ROIS_DIR = DATA / "ROIs"
PATCH_DIR = ROIS_DIR / "patches"
RAD_DIR = DATA / "radiance"
RFL_DIR = DATA / "reflectance"
MASK_DIR = DATA / "mask_summary"
GEOM_DIR = DATA / "geometry"
CUBE_DIR = DATA / "pixel_cubes"
QC_DIR = DATA / "qc"
MATCHED_DIR = DATA / "matched_pairs"   # matched pairs are geometry-based (variant-free)

_PRODUCT_DIR = {"rad": RAD_DIR, "rfl": RFL_DIR}

# ------------------------------------------------------- derived-output directories
# These hold radiance-sensitive products (deltas, drift, BRDF, and their figures),
# so an alternate radiometry variant (e.g. Earth–Sun correction OFF) nests them all
# under a labelled subfolder to keep it from overwriting the canonical baseline. The
# baseline variant is the empty string, which reproduces the historical flat layout
# byte-for-byte. See :func:`set_variant` and ``RunConfig.output_variant``.
_VARIANT = ""

# Optional extra sub-namespace for the BRDF outputs only (e.g. a patch-size tag
# like "20km"), so a BRDF sensitivity run doesn't overwrite the baseline BRDF
# products (whose filenames are not size-keyed). Nests INSIDE the variant folder.
# Empty string = no extra nesting. See :func:`set_brdf_label` and ``brdf.label``.
_BRDF_LABEL = ""

# Base (un-nested) derived dirs; the public names below are rebuilt from these
# whenever the active variant changes.
_DELTA_PER_PATCH_BASE = DATA / "delta_r" / "per_patch"
_DELTA_AGG_BASE = DATA / "delta_r" / "aggregated"
_DRIFT_BASE = DATA / "drift"
_BRDF_BASE = DATA / "brdf"
_SPECTRA_RFL_BASE = FIGURES / "spectra" / "reflectance"
_SPECTRA_RAD_BASE = FIGURES / "spectra" / "radiance"
_MATCHED_FIG_BASE = FIGURES / "matched_pairs"
_BRDF_FIG_BASE = FIGURES / "brdf"


def _rebuild_derived_dirs() -> None:
    """(Re)compute the variant-nested derived-output dirs + their lookups.

    Idempotent. Called once at import (variant ``""`` -> historical flat layout)
    and again by :func:`set_variant`. Path-builder functions read these module
    globals at call time, so rebinding here redirects every derived output.
    """
    global DELTA_PER_PATCH_DIR, DELTA_AGG_DIR, DRIFT_DIR, BRDF_DIR
    global SPECTRA_RFL_DIR, SPECTRA_RAD_DIR, MATCHED_FIG_DIR, CHANNEL_FIG_DIR
    global BRDF_FIG_DIR, _SPECTRA_DIR, _ALL_OUTPUT_DIRS

    def nest(base: Path) -> Path:
        return base / _VARIANT if _VARIANT else base

    DELTA_PER_PATCH_DIR = nest(_DELTA_PER_PATCH_BASE)
    DELTA_AGG_DIR = nest(_DELTA_AGG_BASE)
    DRIFT_DIR = nest(_DRIFT_BASE)
    BRDF_DIR = nest(_BRDF_BASE)
    SPECTRA_RFL_DIR = nest(_SPECTRA_RFL_BASE)
    SPECTRA_RAD_DIR = nest(_SPECTRA_RAD_BASE)
    MATCHED_FIG_DIR = nest(_MATCHED_FIG_BASE)
    CHANNEL_FIG_DIR = MATCHED_FIG_DIR / "individual_channels"
    BRDF_FIG_DIR = nest(_BRDF_FIG_BASE)

    # BRDF-only extra label nests inside the variant folder (e.g. brdf/20km or
    # brdf/no_earthsun/20km), so patch-size sweeps don't collide with the baseline.
    if _BRDF_LABEL:
        BRDF_DIR = BRDF_DIR / _BRDF_LABEL
        BRDF_FIG_DIR = BRDF_FIG_DIR / _BRDF_LABEL

    _SPECTRA_DIR = {"rad": SPECTRA_RAD_DIR, "rfl": SPECTRA_RFL_DIR}
    _ALL_OUTPUT_DIRS = (
        PATCH_DIR, RAD_DIR, RFL_DIR, MASK_DIR, GEOM_DIR, CUBE_DIR, QC_DIR,
        MATCHED_DIR, DELTA_PER_PATCH_DIR, DELTA_AGG_DIR, DRIFT_DIR, BRDF_DIR, ROIS_DIR,
        SPECTRA_RFL_DIR, SPECTRA_RAD_DIR, MATCHED_FIG_DIR, CHANNEL_FIG_DIR, BRDF_FIG_DIR,
    )


def set_variant(label) -> None:
    """Select the active radiometry output variant (``""`` = canonical baseline).

    A non-empty *label* (e.g. ``"no_earthsun"``) nests only the derived-output
    dirs under ``<dir>/<label>/`` so an alternate run cannot overwrite the
    baseline; raw-input dirs are never affected. Runners call this once, right
    after :func:`emit_pics.run_config.load_config`, with ``cfg.output_variant``.
    """
    global _VARIANT
    _VARIANT = str(label or "")
    _rebuild_derived_dirs()


def active_variant() -> str:
    """The active output-variant label (``""`` for the canonical baseline)."""
    return _VARIANT


def set_brdf_label(label) -> None:
    """Nest the BRDF outputs under an extra sub-namespace (``""`` = none).

    Independent of :func:`set_variant`: a non-empty *label* (e.g. ``"20km"``)
    nests only the BRDF data/figure dirs, inside the variant folder, so a BRDF
    sensitivity run (different patch size, etc.) cannot overwrite the baseline
    BRDF products. The three BRDF runners call this with ``cfg.brdf_label``.
    """
    global _BRDF_LABEL
    _BRDF_LABEL = str(label or "")
    _rebuild_derived_dirs()


def active_brdf_label() -> str:
    """The active BRDF-only output label (``""`` when unset)."""
    return _BRDF_LABEL


_rebuild_derived_dirs()   # initialise the derived dirs at import (baseline layout)


@lru_cache(maxsize=None)
def _tok(site: str) -> str:
    """Canonical filename token for a site (cached — sites.yaml is read once)."""
    return canonical_token(site)


# --------------------------------------------------------------- site-level inputs
def patch_dict(site) -> Path:
    """Patch–granule dictionary CSV (``patch_EMIT_dict_{tok}.csv``)."""
    return PATCH_DIR / f"patch_EMIT_dict_{_tok(site)}.csv"


def patch_bnds(site) -> Path:
    """Patch-boundary GeoPackage (``patches_{tok}_bnds.gpkg``)."""
    return PATCH_DIR / f"patches_{_tok(site)}_bnds.gpkg"


def patch_atts(site) -> Path:
    """Patch attribute table (``patches_{tok}_atts.csv``)."""
    return PATCH_DIR / f"patches_{_tok(site)}_atts.csv"


def granule_footprints(site) -> Path:
    """Granule-footprint GeoPackage (``emit_granule_footprints_{tok}.gpkg``)."""
    return ROIS_DIR / f"emit_granule_footprints_{_tok(site)}.gpkg"


def pics_gpkg() -> Path:
    """Derived PICS box GeoPackage for GIS (``northafrican_pics.gpkg``)."""
    return ROIS_DIR / "northafrican_pics.gpkg"


# ---------------------------------------------------- per-acquisition per-channel
def perchannel(site, patch, ts, product) -> Path:
    """Per-channel stats CSV for one acquisition (``product`` = ``'rad'``|``'rfl'``)."""
    return _PRODUCT_DIR[product] / f"{_tok(site)}_{patch}_{ts}_perchannel_{product}.csv"


def mask_summary(site, patch, ts) -> Path:
    """Per-acquisition MASK-band summary CSV."""
    return MASK_DIR / f"{_tok(site)}_{patch}_{ts}_mask_summary.csv"


def pixel_cube(site, patch, ts) -> Path:
    """Cached raw in-patch pixel cube (NetCDF)."""
    return CUBE_DIR / f"cube_{_tok(site)}_{patch}_{ts}.nc"


# ----------------------------------------------------------------- per-patch tables
def geometry(site, patch) -> Path:
    """Per-scene SZA/VZA geometry table (``geometry_{tok}_{patch}.csv``)."""
    return GEOM_DIR / f"geometry_{_tok(site)}_{patch}.csv"


def qc_sheet(site, patch) -> Path:
    """QC worksheet (``{tok}_{patch}_qc.csv``)."""
    return QC_DIR / f"{_tok(site)}_{patch}_qc.csv"


def qc_pdf(site, patch) -> Path:
    """QC snapshot PDF (``{tok}_{patch}_QC_snapshots.pdf``)."""
    return QC_DIR / f"{_tok(site)}_{patch}_QC_snapshots.pdf"


def matched_pairs(site, patch) -> Path:
    """Matched-pairs CSV (``{tok}_{patch}_matched_pairs.csv``)."""
    return MATCHED_DIR / f"{_tok(site)}_{patch}_matched_pairs.csv"


def delta_per_patch(site, patch) -> Path:
    """Per-patch delta table (``{tok}_{patch}_delta.csv``)."""
    return DELTA_PER_PATCH_DIR / f"{_tok(site)}_{patch}_delta.csv"


def delta_aggregated(site, size) -> Path:
    """Per-PICS aggregated delta table (``{tok}_{size}_delta.csv``)."""
    return DELTA_AGG_DIR / f"{_tok(site)}_{size}_delta.csv"


def drift(size) -> Path:
    """Cross-PICS per-channel drift stats (``{size}_drift.csv``)."""
    return DRIFT_DIR / f"{size}_drift.csv"


def drift_bootstrap(size) -> Path:
    """Matching-bootstrap per-channel drift stats (``{size}_drift_bootstrap.csv``)."""
    return DRIFT_DIR / f"{size}_drift_bootstrap.csv"


# --------------------------------------------------- BRDF pooled multi-site drift
# Pooled across many sites, so these are NOT keyed by a single site/patch token.
# Single-channel products carry the target wavelength; the spectrum spans all bands.
def brdf_design() -> Path:
    """Pooled per-scene design table (site, patch, t, g, rho) — ``brdf_design.csv``."""
    return BRDF_DIR / "brdf_design.csv"


def brdf_leverage() -> Path:
    """Per-site scene-count / geometry-leverage summary (``brdf_leverage.csv``)."""
    return BRDF_DIR / "brdf_leverage.csv"


def brdf_persite_gamma(wl) -> Path:
    """Per-site standalone-gamma table for the shared-drift check (§4b)."""
    return BRDF_DIR / f"brdf_persite_gamma_{wl:.0f}nm.csv"


def brdf_joint_fit(wl) -> Path:
    """Joint pooled fit: per-site alpha/beta + the one shared gamma (§5)."""
    return BRDF_DIR / f"brdf_joint_fit_{wl:.0f}nm.csv"


def brdf_bootstrap_gamma(wl) -> Path:
    """Bootstrapped pooled shared-gamma with CI, per product (§7)."""
    return BRDF_DIR / f"brdf_bootstrap_gamma_{wl:.0f}nm.csv"


def brdf_drift_spectrum() -> Path:
    """Pooled shared-gamma per EMIT channel (``brdf_drift_spectrum.csv``, §8)."""
    return BRDF_DIR / "brdf_drift_spectrum.csv"


# ------------------------------------------------------------------------ figures
def matched_pairs_fig(site, patch) -> Path:
    """SZA/VZA-over-time figure (``{tok}_{patch}_matched_pairs.png``)."""
    return FIGURES / f"{_tok(site)}_{patch}_matched_pairs.png"


def spectra_fig(site, patch, stat, product) -> Path:
    """Survivor-spectra overlay PNG (``{tok}_{patch}_{stat}_{product}_spectra.png``)."""
    return _SPECTRA_DIR[product] / f"{_tok(site)}_{patch}_{stat}_{product}_spectra.png"


def granule_coverage_fig(site, patch_size_km) -> Path:
    """Granule-coverage figure (``{tok}_{km}km_EMITgranule_coverage.png``)."""
    return FIGURES / f"{_tok(site)}_{patch_size_km}km_EMITgranule_coverage.png"


def drift_spectrum_fig(size) -> Path:
    """Drift-spectrum figure (``{size}_drift_spectrum.png``)."""
    return MATCHED_FIG_DIR / f"{size}_drift_spectrum.png"


def drift_bootstrap_spectrum_fig(size) -> Path:
    """Matching-bootstrap drift-spectrum figure (``{size}_drift_bootstrap_spectrum.png``)."""
    return MATCHED_FIG_DIR / f"{size}_drift_bootstrap_spectrum.png"


def channel_fig(size, wavelength) -> Path:
    """Per-channel ΔR-vs-Δt figure (``{size}_{wavelength:04.0f}.png``)."""
    return CHANNEL_FIG_DIR / f"{size}_{wavelength:04.0f}.png"


# --------------------------------------------------- BRDF pooled multi-site figures
def brdf_leverage_fig(wl) -> Path:
    """Pooled leverage / confounding scatter (§4a)."""
    return BRDF_FIG_DIR / f"brdf_leverage_scatter_{wl:.0f}nm.png"


def brdf_persite_gamma_fig(wl) -> Path:
    """Per-site standalone-gamma forest plot (shared-drift check, §4b)."""
    return BRDF_FIG_DIR / f"brdf_persite_gamma_{wl:.0f}nm.png"


def brdf_corrected_fig(wl) -> Path:
    """Quasi-BRDF-corrected collapse onto the shared drift line (§6)."""
    return BRDF_FIG_DIR / f"brdf_corrected_{wl:.0f}nm.png"


def brdf_bootstrap_fig(wl) -> Path:
    """Pooled shared-gamma bootstrap histogram (§7)."""
    return BRDF_FIG_DIR / f"brdf_bootstrap_{wl:.0f}nm.png"


def brdf_drift_spectrum_fig() -> Path:
    """Pooled drift spectrum figure (§8)."""
    return BRDF_FIG_DIR / "brdf_drift_spectrum.png"


def ensure_dirs():
    """Create every output directory (idempotent). Convenient for a runner."""
    for d in _ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
