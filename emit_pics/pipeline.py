"""Ordered pipeline steps + stage resolution for the CLI runner.

The thin CLI in ``scripts/run_pipeline.py`` uses this to run the per-step
scripts in the correct order for the current ``run_config.yaml``. Every stage
operates on all patches of ``cfg.patch_size`` (the batched runners) and on the
run_config rosters (BRDF / Landsat). The per-patch ``run_single_patch_*``
scripts remain available for one-off inspection but are not driven by the
orchestrator. Steps run as subprocesses, so each is isolated and reads
``run_config.yaml`` itself — no arguments are threaded through.

``scripts_for`` (stage -> script list) is pure and unit tested; ``run_script`` /
``run_stage`` do the subprocess execution.
"""

import subprocess
import sys

from emit_pics.paths import REPO_ROOT

SCRIPTS_DIR = REPO_ROOT / "scripts"

STAGE_ORDER = ("setup", "extract", "analyze", "drift", "brdf",
               "landsat", "compare", "figures", "appendix")

#: stage -> ordered list of runner scripts (all batched / roster-driven).
STEPS = {
    "setup": ["run_patch_generator.py", "run_id_granules.py"],
    "extract": ["run_PICS_allpatches_cube_extract.py",
                "run_PICS_allpatches_qc_products.py"],
    "analyze": ["run_PICS_allpatches_id_matched_pairs.py",
                "run_PICS_allpatches_delta_calculations.py",
                "run_PICS_allpatches_aggregate_pairs.py"],
    "drift": ["run_matched_pairs_drift_analysis.py"],
    "brdf": ["run_brdf_assemble.py", "run_brdf_single_channel.py",
             "run_brdf_drift_spectrum.py"],
    "landsat": ["run_landsat_cube_extract.py", "run_landsat_qc_products.py",
                "run_landsat_brdf_drift_spectrum.py",
                "run_landsat_brdf_robustness.py"],
    "compare": ["run_emit_oli2_band_comparison.py"],
    "figures": ["run_publication_figures.py", "run_data_volume_table.py"],
    "appendix": ["run_cos_sza_corr.py", "run_brdf_loo.py"],
}


def scripts_for(stage: str) -> list:
    """Ordered script filenames to run for *stage* (pure).

    Raises ``ValueError`` for an unknown stage.
    """
    if stage not in STEPS:
        raise ValueError(f"Unknown stage '{stage}'. Choose from {list(STEPS)}.")
    return list(STEPS[stage])


def run_script(name: str, *, python: str | None = None) -> int:
    """Run one step script as a subprocess; return its exit code."""
    path = SCRIPTS_DIR / name
    bar = "=" * 70
    print(f"\n{bar}\n>>> {name}\n{bar}", flush=True)
    return subprocess.run([python or sys.executable, str(path)]).returncode


def run_stage(stage: str, *, python: str | None = None):
    """Run every script of *stage* in order; stop at the first non-zero exit.

    Returns ``(returncode, failed_script)`` — ``(0, None)`` on full success.
    """
    for name in scripts_for(stage):
        rc = run_script(name, python=python)
        if rc != 0:
            return rc, name
    return 0, None
