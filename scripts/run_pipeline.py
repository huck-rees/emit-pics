#!/usr/bin/env python
"""Run the EMIT-PICS pipeline steps in order for the current run_config.yaml.

    python scripts/run_pipeline.py <stage> [--mode single|allpatches] [--dry-run]

Stages (run in this order, one at a time):
    setup     patch_generator + id_granules        (once per site; uses Earthdata)
    extract   cube_extract + qc_products           (uses Earthdata)
              -> then inspect QC and mark the worksheet reviewed
    analyze   id_matched_pairs + delta_calculations [+ aggregate_pairs (allpatches)]
    drift     matched_pairs_drift_analysis         (cross-PICS)
    brdf      assemble + single_channel + drift_spectrum   (pooled multi-site)
    all       setup -> extract, then stop for the manual QC review

Modes:
    --mode allpatches   every patch of cfg.patch_size            (default)
    --mode single       just cfg.patch (one-off inspection)

The `brdf` stage is always pooled across the run_config.yaml `brdf:` roster, so
it ignores --mode (both modes run the same three scripts).

The QC gate is manual and enforced: in single mode `analyze` refuses to run
until the patch's QC worksheet is marked reviewed. Each step is a separate
subprocess that reads run_config.yaml itself, so a failure stops the pipeline
with that step's own message.
"""

import argparse
import sys

from emit_pics import paths, pipeline, qc_gate
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config

_BAR = "=" * 70

QC_NOTICE = f"""
{_BAR}
QC GATE — manual review required before `analyze`.
{_BAR}
Inspect the QC products (data/qc/*_QC_snapshots.pdf), then either:
  * run:  python scripts/run_qc_review.py          (interactive keep/remove), or
  * edit: data/qc/<PICS>_<patch>_qc.csv  (keep=FALSE for bad scenes) and set
          reviewed=TRUE in the top row.
Then continue:  python scripts/run_pipeline.py analyze --mode {{mode}}
{_BAR}""".strip()


def _print_plan(mode: str) -> None:
    print(f"Planned steps per stage (mode = {mode}):")
    for stage in pipeline.STAGE_ORDER:
        names = pipeline.scripts_for(stage, mode)
        print(f"  {stage:8s} -> " + (", ".join(names) if names else "(none)"))


def _qc_gate_blocks(cfg, mode: str) -> bool:
    """In single mode, True if the patch's QC worksheet is not yet reviewed."""
    if mode != "single":
        return False   # batched scripts skip unreviewed patches themselves
    qc = paths.qc_sheet(canonical_token(cfg.site), cfg.patch)
    if qc_gate.is_reviewed(qc):
        return False
    print(f"\n!! QC gate: {qc.name} is not marked reviewed — `analyze` blocked.")
    print(QC_NOTICE.format(mode=mode))
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the EMIT-PICS pipeline steps in order.")
    parser.add_argument("stage", choices=list(pipeline.STAGE_ORDER) + ["all"])
    parser.add_argument("--mode", choices=list(pipeline.MODES), default="allpatches")
    parser.add_argument("--config", default=None,
                        help="path to an alternate run_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the steps that would run, without executing")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    print(f"{_BAR}\nEMIT-PICS pipeline | site={cfg.site} patch={cfg.patch} "
          f"size={cfg.patch_size} | stage={args.stage} mode={args.mode}\n{_BAR}")

    if args.dry_run:
        _print_plan(args.mode)
        return 0

    stages = ["setup", "extract"] if args.stage == "all" else [args.stage]

    for stage in stages:
        if stage == "analyze" and _qc_gate_blocks(cfg, args.mode):
            return 1
        rc, failed = pipeline.run_stage(stage, args.mode)
        if rc != 0:
            print(f"\n!! Stage '{stage}' stopped: {failed} exited {rc}.")
            return rc

    # trailing guidance / next-step hint
    if args.stage in ("extract", "all"):
        print(QC_NOTICE.format(mode=args.mode))
    elif args.stage == "setup":
        print(f"\nSetup done. Next: python scripts/run_pipeline.py extract --mode {args.mode}")
    elif args.stage == "analyze":
        print("\nAnalyze done. Next: python scripts/run_pipeline.py drift")
    elif args.stage == "drift":
        print("\nDrift analysis done. See data/drift/ and figures/matched_pairs/.")
    elif args.stage == "brdf":
        print("\nBRDF pooled drift done. See data/brdf/ and figures/brdf/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
