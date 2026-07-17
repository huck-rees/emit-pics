"""Golden regression test for the delta -> drift statistics chain.

``build_golden_delta_df`` injects a known linear drift; ``compute_drift_spectrum``
must recover it and produce a per-channel spectrum that is byte-for-byte stable.
The expected spectrum is committed at ``tests/golden/drift_spectrum_rad.csv``.

If a *legitimate* method change alters the numbers, regenerate the golden file:

    python -m tests.regenerate_golden

and review the diff before committing — an unexplained diff here means a refactor
silently changed the science.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from emit_pics.matched_pairs_drift_analysis import compute_drift_spectrum
from tests._helpers import build_golden_delta_df

GOLDEN = Path(__file__).parent / "golden" / "drift_spectrum_rad.csv"


def _spectrum():
    return compute_drift_spectrum(build_golden_delta_df(), "rad")


def test_drift_spectrum_recovers_injected_slopes():
    spec = _spectrum().set_index("band_index")
    np.testing.assert_allclose(spec.loc[0, "slope_pct_yr"], 2.0, rtol=1e-6)
    np.testing.assert_allclose(spec.loc[1, "slope_pct_yr"], -1.0, rtol=1e-6)
    assert np.isnan(spec.loc[2, "slope_pct_yr"])   # absorption band excluded


def test_drift_spectrum_matches_golden_file():
    assert GOLDEN.exists(), (
        f"Golden file missing: {GOLDEN}. Generate it with "
        f"`python -m tests.regenerate_golden`."
    )
    got = _spectrum().reset_index(drop=True)
    want = pd.read_csv(GOLDEN)
    pdt.assert_frame_equal(got, want, check_dtype=False, rtol=1e-8, atol=1e-10)
