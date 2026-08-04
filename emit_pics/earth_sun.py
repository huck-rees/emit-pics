"""Earth–Sun distance normalization for EMIT L1B radiance.

EMIT L1B at-sensor radiance is **not** normalized for Earth–Sun distance, so it
carries the seasonal ``1/d²`` solar-flux term (perihelion early January,
aphelion early July; annual span ~±3.4%). L2A reflectance already removes it.
Over a pseudo-invariant site this term leaks into the radiance drift estimate
whenever the acquisition pairs / scene pool are seasonally imbalanced across
their time baselines.

The correction is a single per-scene division of the radiance by the
eccentricity factor ``E0 = (r0/r)² = (d0/d)²`` — the ratio of mean to actual
Earth–Sun distance, squared — bringing every scene to a common 1-AU
illumination. Since ``d0 = 1 AU``, ``E0 = 1/d²`` with ``d`` in AU, so dividing
radiance by ``E0`` is the same as multiplying it by ``d²``. It is **exact**, not
first-order: ``E0`` is spatially and spectrally uniform over a scene, so dividing
the per-channel median (or mean) radiance by ``E0`` equals dividing every pixel
by ``E0``.

Two providers of ``E0`` are offered; both take a datetime (or array of them) and
return ``E0``:

* :func:`table_E0` — a per-day-of-year lookup from ``config/earth_sun_distance.txt``
  (an ephemeris table of ``d`` in AU), returning ``E0 = 1/d²``. This is the
  **pipeline default** (``radiometry.e0_method: table``), so ``E0`` is tied to a
  tabulated ephemeris a reviewer can point at.
* :func:`spencer_E0` — the closed-form Spencer (1971) truncated Fourier series.
  Reference coefficients as reproduced in Iqbal, *An Introduction to Solar
  Radiation* (1983), Eq. 1.2.1 — the accessible citation for the obscure primary
  note J. W. Spencer, *Search* 2(5): 172 (1971). The two providers agree to
  <0.1% (well below the drift signal), so results are unchanged either way.

The bare-function ``method=`` default below stays ``"spencer"`` so the library is
self-contained (no data-file read) when called directly; the pipeline passes
``cfg.e0_method`` (``table`` by default), so production runs use the table.

This module is the single source of the ``E0`` formula: both the pipeline loaders
(:func:`emit_pics.delta_calculations.load_scene_spectrum`,
:func:`emit_pics.brdf_drift.scene_perchannel`) and
``notebooks/earth_sun_distance_diagnostic.ipynb`` import from here, so the
correction is defined once and cannot drift between them.

Pure function library — no side effects on import (the distance table is read
lazily and cached on first use of :func:`table_E0`).
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

#: Per-scene radiance stat columns divided by ``E0`` (reflectance is untouched).
RAD_STAT_COLS = ("median_rad", "mean_rad")

#: Ephemeris table of Earth–Sun distance ``d`` (AU) by day-of-year (1..366).
_DISTANCE_TABLE_PATH = Path(__file__).resolve().parent.parent / "config" / "earth_sun_distance.txt"


def _to_doy(dt):
    """Day-of-year (1..366) as an int array for scalar or array-like *dt*."""
    scalar = isinstance(dt, (str, datetime, pd.Timestamp)) or np.ndim(dt) == 0
    doy = (pd.to_datetime(pd.Series([dt] if scalar else list(dt)))
           .dt.dayofyear.to_numpy().astype(int))
    return doy, scalar


def spencer_E0(dt):
    """Eccentricity factor ``E0 = (d0/d)²`` (Spencer 1971) from acquisition date(s).

    ``E0`` is the multiplicative factor on top-of-atmosphere solar irradiance —
    and hence on un-normalized at-sensor radiance: ``E0 > 1`` near perihelion
    (early January), ``E0 < 1`` near aphelion (early July), spanning ~±3.4% over
    the year. Depends only on day-of-year. Closed-form truncated Fourier series;
    coefficients per Iqbal (1983) Eq. 1.2.1 (after Spencer 1971).

    Parameters
    ----------
    dt : datetime | pandas.Timestamp | str | array-like of those
        One acquisition datetime or many. Strings are parsed by pandas.

    Returns
    -------
    float | numpy.ndarray
        A scalar for scalar input, else a float array aligned with *dt*.
    """
    doy, scalar = _to_doy(dt)
    g = 2.0 * np.pi * (doy.astype(float) - 1.0) / 365.0
    e0 = (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g)
          + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g))
    return float(e0[0]) if scalar else e0


@lru_cache(maxsize=1)
def _distance_table(path=None):
    """Load the ephemeris distance table as ``d[doy]`` (index 0 unused, 1..366).

    Cached so the file is read once per process. The file has two whitespace-
    separated columns: day-of-year and Earth–Sun distance ``d`` in AU.
    """
    p = Path(path) if path is not None else _DISTANCE_TABLE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Earth–Sun distance table not found: {p}\n"
            "Expected config/earth_sun_distance.txt (day-of-year, distance_AU)."
        )
    d = np.full(367, np.nan)          # 1-indexed by day-of-year (1..366)
    with open(p) as f:
        for line in f:
            s = line.split()
            if len(s) == 2:
                d[int(s[0])] = float(s[1])
    return d


def earth_sun_distance_au(dt, *, table_path=None):
    """Earth–Sun distance ``d`` (AU) by day-of-year, from the ephemeris table.

    Perihelion ``d ≈ 0.983`` AU (early January), aphelion ``d ≈ 1.017`` AU (early
    July). Same scalar/array contract as :func:`spencer_E0`.
    """
    doy, scalar = _to_doy(dt)
    d = _distance_table(table_path)[doy]
    return float(d[0]) if scalar else d


def table_E0(dt, *, table_path=None):
    """Eccentricity factor ``E0 = 1/d²`` from the tabulated Earth–Sun distance.

    The ephemeris-table analogue of :func:`spencer_E0` (they agree to <0.1%).
    ``d`` is looked up by day-of-year from ``config/earth_sun_distance.txt``; the
    multiplicative 1-AU normalization is ``d² = 1/E0``.
    """
    d = earth_sun_distance_au(dt, table_path=table_path)
    return 1.0 / np.asarray(d, dtype=float) ** 2 if np.ndim(d) else 1.0 / d ** 2


#: Selectable ``E0`` providers, keyed by ``radiometry.e0_method``.
E0_METHODS = {"spencer": spencer_E0, "table": table_E0}


def resolve_e0(method="spencer"):
    """Return the ``E0`` provider callable for *method* (``"spencer"`` | ``"table"``)."""
    try:
        return E0_METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unknown E0 method {method!r}; choose one of {sorted(E0_METHODS)}."
        )


def normalize_radiance_spectrum(spec, dt, cols=RAD_STAT_COLS, *, method="spencer"):
    """Return a copy of a per-scene spectrum with radiance divided by ``E0(dt)``.

    Brings the scene's radiance to a common 1-AU illumination. Reflectance
    columns are left untouched (L2A already normalizes Earth–Sun distance). Only
    columns in *cols* that are actually present are divided, so this is safe on
    both the wide per-channel CSV (``median_rad``/``mean_rad``) and the merged
    spectrum returned by :func:`load_scene_spectrum`.

    Parameters
    ----------
    spec : pandas.DataFrame
        Per-scene spectrum. ``None`` is passed through unchanged.
    dt : datetime | pandas.Timestamp | str
        The scene's acquisition datetime (drives ``E0``).
    cols : sequence of str
        Radiance stat columns to normalize (default :data:`RAD_STAT_COLS`).
    method : str, keyword-only
        ``E0`` provider: ``"spencer"`` (closed form, default) or ``"table"``
        (ephemeris lookup). See :data:`E0_METHODS`.

    Returns
    -------
    pandas.DataFrame | None
        A copy with the present radiance columns divided by ``E0``; the input
        object is not mutated.
    """
    if spec is None:
        return None
    e0 = resolve_e0(method)(dt)
    out = spec.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c] / e0
    return out
