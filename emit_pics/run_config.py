"""Load ``run_config.yaml`` — the one place the analysis target + knobs live.

A single YAML at the repo root replaces the ``USER-CONFIGURABLE INPUTS`` block
that used to be copied into every script. Scripts do::

    from emit_pics.run_config import load_config
    cfg = load_config()
    ...  cfg.site, cfg.patch, cfg.sza_tol, ...

The YAML is nested for readability; this loader flattens it into a typed
:class:`RunConfig` with defaults, and fails loudly (clear message) if a required
key (``site`` / ``patch`` / ``patch_size``) is missing.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_CONFIG = REPO_ROOT / "run_config.yaml"


@dataclass
class RunConfig:
    """The resolved run configuration (flat, typed)."""

    # --- analysis target (required) ---
    site: str
    patch: str
    patch_size: str
    # --- EMIT products ---
    rad_short_name: str = "EMITL1BRAD"
    rfl_short_name: str = "EMITL2ARFL"
    # --- matched-pair tolerances ---
    sza_tol: float = 2.0
    vza_tol: float = 5.0
    # --- QC gate ---
    require_qc_review: bool = True
    # --- spectra ---
    spectra_stat: str = "median"
    # --- radiometry ---
    #: Divide L1B radiance by the Earth–Sun eccentricity factor E0 at read time
    #: (bring every scene to 1 AU, matching L2A). Applied in the per-scene loaders.
    earth_sun_normalize: bool = True
    #: E0 provider: "table" (ephemeris lookup from config/earth_sun_distance.txt,
    #: default) or "spencer" (closed-form Spencer 1971 / Iqbal 1983). Agree <0.1%.
    e0_method: str = "table"
    # --- drift analysis ---
    alpha: float = 0.05
    use_aggregated: bool = True
    individual_channels: bool = True
    #: {PICS_ID: patch_id} — one representative patch per PICS for the original
    #: cross-PICS pool (publication per-channel figure); disambiguates PICS whose
    #: per_patch/ folder also holds the parallel all-patches experiment's tables.
    drift_cross_pics_patches: dict = field(default_factory=dict)
    # --- BRDF / quasi-BRDF pooled multi-site drift ---
    # (shares the global drift.alpha and qc.require_review knobs)
    brdf_site_patches: dict = field(default_factory=dict)
    brdf_exclude_sites: list = field(default_factory=list)
    brdf_target_wl: float = 858.0
    brdf_min_scenes: int = 4
    brdf_n_bootstrap: int = 1000
    brdf_per_site_boot: int = 500
    brdf_spectrum_n_boot: int = 300
    brdf_gamma_cap: float = 500.0
    # --- extraction ---
    cache_pixels: bool = True
    want_obs: bool = True
    filter_agg_flag: bool = False
    min_valid_frac: float = 0.60
    # --- granule search (upstream) ---
    date_start: str = "2022-08-01"
    date_end: str = "2026-06-30"
    search_patch_size_km: int = 10
    # --- patch generation (upstream) ---
    min_patch_km: int = 10
    increment_km: int = 10

    def as_dict(self) -> dict:
        """Flat dict of every field — used for provenance sidecars (change #3)."""
        return asdict(self)


def load_config(path=None) -> RunConfig:
    """Read ``run_config.yaml`` (or *path*) into a :class:`RunConfig`.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If a required key (``site`` / ``patch`` / ``patch_size``) is missing.
    """
    path = Path(path) if path is not None else DEFAULT_RUN_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            f"run_config.yaml not found: {path}\n"
            "Copy the template at the repo root and set site / patch / patch_size."
        )
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    for key in ("site", "patch", "patch_size"):
        if raw.get(key) in (None, ""):
            raise ValueError(
                f"run_config.yaml is missing required key '{key}' (in {path})."
            )

    def sec(name: str) -> dict:
        return raw.get(name) or {}

    products, matching = sec("products"), sec("matching")
    qc, spectra, drift = sec("qc"), sec("spectra"), sec("drift")
    extract, gs, pg = sec("extract"), sec("granule_search"), sec("patch_gen")
    brdf, radiometry = sec("brdf"), sec("radiometry")

    # BRDF roster: keep only sites with a patch id set (drop null/blank entries).
    brdf_site_patches = {str(s): str(p) for s, p in (brdf.get("site_patches") or {}).items()
                         if p not in (None, "")}
    brdf_exclude_sites = [str(s) for s in (brdf.get("exclude_sites") or [])]

    return RunConfig(
        site=str(raw["site"]),
        patch=str(raw["patch"]),
        patch_size=str(raw["patch_size"]),
        rad_short_name=str(products.get("rad_short_name", "EMITL1BRAD")),
        rfl_short_name=str(products.get("rfl_short_name", "EMITL2ARFL")),
        sza_tol=float(matching.get("sza_tol", 2.0)),
        vza_tol=float(matching.get("vza_tol", 5.0)),
        require_qc_review=bool(qc.get("require_review", True)),
        spectra_stat=str(spectra.get("stat", "median")),
        earth_sun_normalize=bool(radiometry.get("earth_sun_normalize", True)),
        e0_method=str(radiometry.get("e0_method", "table")),
        alpha=float(drift.get("alpha", 0.05)),
        use_aggregated=bool(drift.get("use_aggregated", True)),
        individual_channels=bool(drift.get("individual_channels", True)),
        drift_cross_pics_patches={str(s): str(p) for s, p
                                  in (drift.get("cross_pics_patches") or {}).items()
                                  if p not in (None, "")},
        brdf_site_patches=brdf_site_patches,
        brdf_exclude_sites=brdf_exclude_sites,
        brdf_target_wl=float(brdf.get("target_wl", 858.0)),
        brdf_min_scenes=int(brdf.get("min_scenes", 4)),
        brdf_n_bootstrap=int(brdf.get("n_bootstrap", 1000)),
        brdf_per_site_boot=int(brdf.get("per_site_boot", 500)),
        brdf_spectrum_n_boot=int(brdf.get("spectrum_n_boot", 300)),
        brdf_gamma_cap=float(brdf.get("gamma_cap", 500.0)),
        cache_pixels=bool(extract.get("cache_pixels", True)),
        want_obs=bool(extract.get("want_obs", True)),
        filter_agg_flag=bool(extract.get("filter_agg_flag", False)),
        min_valid_frac=float(extract.get("min_valid_frac", 0.60)),
        date_start=str(gs.get("date_start", "2022-08-01")),
        date_end=str(gs.get("date_end", "2026-06-30")),
        search_patch_size_km=int(gs.get("patch_size_km", 10)),
        min_patch_km=int(pg.get("min_patch_km", 10)),
        increment_km=int(pg.get("increment_km", 10)),
    )
