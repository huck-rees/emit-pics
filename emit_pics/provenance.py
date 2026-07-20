"""Write a provenance sidecar next to a pipeline output.

For an output file ``X`` we also write ``X + ".json"`` recording *how it was
made*: a UTC timestamp, the git commit (+ dirty flag + branch), the script that
produced it, the Python + key package versions, and the exact ``run_config``
values used. A CSV on disk can't tell you what produced it; its sidecar can — so
every result is traceable to an exact code state, which is what publication-grade
reproducibility needs.

The sidecar name is ``<output>.json`` (the full output name plus ``.json``, e.g.
``Libya_4_10_7_1_delta.csv.json``) so a ``.csv`` and a same-stem ``.png`` never
collide. Sidecars live beside the (gitignored) data, and travel with it when the
results are archived (e.g. to Zenodo for a paper).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from emit_pics.paths import REPO_ROOT

#: Packages whose versions are worth pinning in the record.
_TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "xarray", "netCDF4", "rasterio",
    "geopandas", "shapely", "matplotlib", "earthaccess",
)


def _git(*args) -> str | None:
    """Run a git command in the repo; return stripped stdout, or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info() -> dict:
    """Current commit hash, branch, and whether the tree is dirty (best-effort)."""
    dirty = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": (dirty != "") if dirty is not None else None,
    }


def _package_versions() -> dict:
    import importlib.metadata as im
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = im.version(pkg)
        except im.PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_manifest(params: dict | None = None, *, script: str | None = None,
                   extra: dict | None = None, timestamp: datetime | None = None) -> dict:
    """Assemble the provenance record (pure except for git/version/time lookups).

    *timestamp* is injectable so tests can be deterministic; it defaults to now
    (UTC).
    """
    ts = timestamp or datetime.now(timezone.utc)
    manifest = {
        "created_utc": ts.isoformat(),
        "script": script or Path(sys.argv[0]).name or None,
        "git": git_info(),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "run_config": params or {},
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_sidecar(output_path, params: dict | None = None, *,
                  script: str | None = None, extra: dict | None = None) -> Path:
    """Write ``<output_path>.json`` describing how *output_path* was produced.

    Parameters
    ----------
    output_path : path-like
        The data/figure file just written.
    params : dict, optional
        The run configuration used (typically ``cfg.as_dict()``).
    script, extra : optional
        Override the recorded script name; attach any extra per-output detail.

    Returns
    -------
    Path
        The sidecar path written.
    """
    output_path = Path(output_path)
    sidecar = output_path.with_name(output_path.name + ".json")
    manifest = build_manifest(params, script=script, extra=extra)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return sidecar
