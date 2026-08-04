"""Earth–Sun distance normalization for EMIT L1B radiance.

EMIT L1B at-sensor radiance is **not** normalized for Earth–Sun distance, so it
carries the seasonal ``1/d²`` solar-flux term (perihelion early January,
aphelion early July; annual span ~±3.4%). L2A reflectance already removes it.
Over a pseudo-invariant site this term leaks into the radiance drift estimate
whenever the acquisition pairs / scene pool are seasonally imbalanced across
their time baselines.

The correction is a single per-scene division of the radiance by the
eccentricity factor ``E0 = (d0/d)²``, bringing every scene to a common 1-AU
illumination. It is **exact**, not first-order: ``E0`` is spatially and
spectrally uniform over a scene, so dividing the per-channel median (or mean)
radiance by ``E0`` equals dividing every pixel by ``E0``.

This module is the single source of the Spencer (1971) ``E0`` formula: both the
pipeline loaders (:func:`emit_pics.delta_calculations.load_scene_spectrum`,
:func:`emit_pics.brdf_drift.scene_perchannel`) and
``notebooks/earth_sun_distance_diagnostic.ipynb`` import :func:`spencer_E0` from
here, so the correction is defined once and cannot drift between them.

Pure function library — no I/O, no side effects on import.
"""

from datetime import datetime

import numpy as np
import pandas as pd

#: Per-scene radiance stat columns divided by ``E0`` (reflectance is untouched).
RAD_STAT_COLS = ("median_rad", "mean_rad")


def spencer_E0(dt):
    """Eccentricity factor ``E0 = (d0/d)²`` (Spencer 1971) from acquisition date(s).

    ``E0`` is the multiplicative factor on top-of-atmosphere solar irradiance —
    and hence on un-normalized at-sensor radiance: ``E0 > 1`` near perihelion
    (early January), ``E0 < 1`` near aphelion (early July), spanning ~±3.4% over
    the year. Depends only on day-of-year.

    Parameters
    ----------
    dt : datetime | pandas.Timestamp | str | array-like of those
        One acquisition datetime or many. Strings are parsed by pandas.

    Returns
    -------
    float | numpy.ndarray
        A scalar for scalar input, else a float array aligned with *dt*.
    """
    scalar = isinstance(dt, (str, datetime, pd.Timestamp)) or np.ndim(dt) == 0
    doy = (pd.to_datetime(pd.Series([dt] if scalar else list(dt)))
           .dt.dayofyear.to_numpy().astype(float))
    g = 2.0 * np.pi * (doy - 1.0) / 365.0
    e0 = (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g)
          + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g))
    return float(e0[0]) if scalar else e0


def normalize_radiance_spectrum(spec, dt, cols=RAD_STAT_COLS):
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

    Returns
    -------
    pandas.DataFrame | None
        A copy with the present radiance columns divided by ``E0``; the input
        object is not mutated.
    """
    if spec is None:
        return None
    e0 = spencer_E0(dt)
    out = spec.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c] / e0
    return out
