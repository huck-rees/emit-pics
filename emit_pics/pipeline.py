"""Ordered pipeline steps + stage/mode resolution for the CLI runner.

The thin CLI in ``scripts/run_pipeline.py`` uses this to run the existing
per-step scripts in the correct order for the current ``run_config.yaml``. Each
step is a ``(single_patch_script, allpatches_script)`` pair; a ``None`` slot
means the step does not apply in that mode (e.g. the per-PICS aggregation is
allpatches-only). Steps run as subprocesses, so each is isolated and reads
``run_config.yaml`` itself — no arguments are threaded through.

``scripts_for`` (the stage/mode -> script-list resolution) is pure and unit
tested; ``run_script`` / ``run_stage`` do the subprocess execution.
"""

import subprocess
import sys

from emit_pics.paths import REPO_ROOT

SCRIPTS_DIR = REPO_ROOT / "scripts"

MODES = ("single", "allpatches")
STAGE_ORDER = ("setup", "extract", "analyze", "drift")

#: stage -> ordered list of ``(single_patch_script, allpatches_script)``.
STEPS = {
    "setup": [
        ("run_patch_generator.py", "run_patch_generator.py"),
        ("run_id_granules.py", "run_id_granules.py"),
    ],
    "extract": [
        ("run_single_patch_cube_extract.py", "run_PICS_allpatches_cube_extract.py"),
        ("run_single_patch_qc_products.py", "run_PICS_allpatches_qc_products.py"),
    ],
    "analyze": [
        ("run_id_matched_pairs.py", "run_PICS_allpatches_id_matched_pairs.py"),
        ("run_delta_calculations.py", "run_PICS_allpatches_delta_calculations.py"),
        (None, "run_PICS_allpatches_aggregate_pairs.py"),   # per-PICS: allpatches only
    ],
    "drift": [
        ("run_matched_pairs_drift_analysis.py", "run_matched_pairs_drift_analysis.py"),
    ],
}


def scripts_for(stage: str, mode: str) -> list:
    """Ordered script filenames to run for *stage* in *mode* (pure).

    Raises ``ValueError`` for an unknown stage or mode.
    """
    if stage not in STEPS:
        raise ValueError(f"Unknown stage '{stage}'. Choose from {list(STEPS)}.")
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(MODES)}.")
    col = 0 if mode == "single" else 1
    return [pair[col] for pair in STEPS[stage] if pair[col] is not None]


def run_script(name: str, *, python: str | None = None) -> int:
    """Run one step script as a subprocess; return its exit code."""
    path = SCRIPTS_DIR / name
    bar = "=" * 70
    print(f"\n{bar}\n>>> {name}\n{bar}", flush=True)
    return subprocess.run([python or sys.executable, str(path)]).returncode


def run_stage(stage: str, mode: str, *, python: str | None = None):
    """Run every script of *stage* in order; stop at the first non-zero exit.

    Returns ``(returncode, failed_script)`` — ``(0, None)`` on full success.
    """
    for name in scripts_for(stage, mode):
        rc = run_script(name, python=python)
        if rc != 0:
            return rc, name
    return 0, None
