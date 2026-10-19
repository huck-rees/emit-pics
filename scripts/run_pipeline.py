#!/usr/bin/env python
"""Run the EMIT-PICS pipeline steps in order for the current run_config.yaml.

    python scripts/run_pipeline.py <stage> [--dry-run]

Stages (run in this order):
    setup     patch_generator + id_granules           (once per site; Earthdata)
    extract   cube_extract + qc_products              (Earthdata)
              -> then inspect QC and mark the worksheets reviewed
    analyze   id_matched_pairs + delta_calculations + aggregate_pairs
    drift     matched_pairs_drift_analysis            (cross-PICS)
    brdf      assemble + single_channel + drift_spectrum   (pooled multi-site)
    landsat   cube_extract + qc_products + brdf_drift_spectrum + robustness
              (OLI-2 control; AWS; -> mark the Landsat QC worksheets reviewed)
    compare   emit_oli2_band_comparison               (EMIT vs OLI-2, 7 bands)
    figures   publication_figures + data_volume_table
    appendix  cos_sza_corr + brdf_loo
    all       every stage above, in order

Every stage operates on all patches of cfg.patch_size and on the run_config
rosters. The per-patch run_single_patch_*.py scripts remain for one-off
inspection but are not driven by this orchestrator.

The QC gate is manual: the batched steps skip unreviewed patches, and analyze /
brdf refuse to run on unreviewed worksheets. Each step is a separate subprocess
that reads run_config.yaml itself, so a failure stops the pipeline with that
step's own message. The patch-size sensitivity (Appendix A) needs a separate
20 km run and is not part of `appendix` — see the README.
"""

import argparse
import sys

from emit_pics import pipeline
from emit_pics.run_config import load_config

_BAR = "=" * 70

QC_NOTICE = f"""
{_BAR}
QC GATE — manual review required before the analysis stages.
{_BAR}
Inspect the QC products (data/qc/*_QC_snapshots.pdf), then either:
  * run:  python scripts/run_PICS_allpatches_qc_review.py   (interactive keep/remove), or
  * edit: data/qc/<PICS>_<patch>_qc.csv  (keep=FALSE for bad scenes) and set
          reviewed=TRUE in the top row.
Then continue:  python scripts/run_pipeline.py analyze
{_BAR}""".strip()


def _print_plan() -> None:
    print("Planned steps per stage:")
    for stage in pipeline.STAGE_ORDER:
        names = pipeline.scripts_for(stage)
        print(f"  {stage:9s} -> " + (", ".join(names) if names else "(none)"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the EMIT-PICS pipeline steps in order.")
    parser.add_argument("stage", choices=list(pipeline.STAGE_ORDER) + ["all"])
    parser.add_argument("--config", default=None,
                        help="path to an alternate run_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the steps that would run, without executing")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    print(f"{_BAR}\nEMIT-PICS pipeline | site={cfg.site} patch={cfg.patch} "
          f"size={cfg.patch_size} | stage={args.stage}\n{_BAR}")

    if args.dry_run:
        _print_plan()
        return 0

    stages = list(pipeline.STAGE_ORDER) if args.stage == "all" else [args.stage]

    for stage in stages:
        rc, failed = pipeline.run_stage(stage)
        if rc != 0:
            print(f"\n!! Stage '{stage}' stopped: {failed} exited {rc}.")
            return rc

    if args.stage == "extract":
        print(QC_NOTICE)
    elif args.stage == "setup":
        print("\nSetup done. Next: python scripts/run_pipeline.py extract")
    elif args.stage == "analyze":
        print("\nAnalyze done. Next: python scripts/run_pipeline.py drift")
    elif args.stage == "drift":
        print("\nDrift analysis done. See data/drift/ and figures/matched_pairs/.")
    elif args.stage == "brdf":
        print("\nBRDF pooled drift done. See data/brdf/ and figures/brdf/.")
    elif args.stage == "landsat":
        print("\nLandsat control done. Review the Landsat QC worksheets if running from scratch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
