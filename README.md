# EMIT PICS Temporal Stability

Assessing the radiometric temporal stability of NASA's EMIT imaging spectrometer over North African pseudo-invariant calibration sites (PICS).

## Repository layout

```
emit_pics/      Python package — extraction, processing, analysis modules
config/         YAML site definitions (PICS coordinates, ROI parameters)
scripts/        Command-line entry points
notebooks/      Exploration and analysis notebooks
data/           Downloaded granules and outputs (gitignored)
tests/          Unit tests
```

## Environment setup

```bash
conda env create -f environment.yml
conda activate emit-pics
```

You also need NASA Earthdata credentials. The extraction script authenticates via `earthaccess.login()`, which looks for credentials in `~/.netrc` (or `_netrc` on Windows) and falls back to an interactive prompt. See the [earthaccess docs](https://earthaccess.readthedocs.io/) for details.

## Quick start

1. Fill in site coordinates in `config/sites.yaml`.
2. Run the extraction script:
   ```bash
   python scripts/run_extraction.py --site libya_4 --start 2023-01-01 --end 2023-12-31
   ```
   Add `--dry-run` to search without downloading.
3. Open `notebooks/explore_granule.ipynb` to inspect the downloaded files.

## Current status / next steps

- **Done:** Project skeleton, Earthdata search and download, manifest with provenance metadata.
- **Next:** Implement GLT orthorectification, OBS band extraction (SZA/VZA), ROI mean computation. See stub functions in `emit_pics/extract.py`.
