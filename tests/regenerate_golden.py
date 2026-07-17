"""(Re)generate the committed golden drift spectrum.

Run from the repo root:  ``python -m tests.regenerate_golden``

Only run this when a *deliberate* method change is expected to alter the output;
review the resulting diff before committing.
"""

from pathlib import Path

from emit_pics.matched_pairs_drift_analysis import compute_drift_spectrum
from tests._helpers import build_golden_delta_df

GOLDEN = Path(__file__).parent / "golden" / "drift_spectrum_rad.csv"


def main():
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    spec = compute_drift_spectrum(build_golden_delta_df(), "rad")
    spec.to_csv(GOLDEN, index=False)
    print(f"wrote {GOLDEN} ({len(spec)} rows)")


if __name__ == "__main__":
    main()
