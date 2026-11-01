#!/usr/bin/env python3
"""
Re-assign the familiarity cells under an alternative OPERATIONAL
DEFINITION of pocket familiarity -- a different similarity metric and
its own threshold -- and ask whether the observed gap survives.

WHY
---
Every result so far rests on one definition of pocket familiarity:
S_p = pocket_qcov against pre-cutoff systems, aggregated over a
system's ligands by minimum, thresholded at 80. The design-factor check
established that the gap is not explained by protein length or pocket
size. It did not establish that pocket_qcov measures pocket novelty
rather than something else that correlates with it.

The same per-ligand table carries two other metrics, sp_fident and
sp_pli. They assign targets very differently: at threshold 90 the NF
cell holds 495 clusters under qcov, 232 under pli and 69 under fident.
A finding that survives a definition which moves this many systems is
about pocket novelty; one that does not is about qcov at 80.

The comparison varies TWO things together: the metric and the
threshold at which it separates familiar from novel. sp_pli at 70 is
not sp_qcov at 80 with one factor changed; each metric has its own
distribution and its own usable threshold, and pairing a metric with a
threshold that empties a cell would test nothing. So this is
robustness to an alternative operationalisation of pocket
familiarity, which is the honest description, and not a controlled
substitution of the metric alone.

WHAT THIS DOES
--------------
1. Recomputes S_p under the alternative metric, with the same
   aggregation and threshold rule, and reports how many TEST systems
   change cell. A metric that reassigns almost nobody cannot test
   anything, and the script says so rather than producing a
   reassuring-looking table.

2. Regroups the per-protein AUPRC already computed -- no refitting --
   under the new assignment, and reports the cell medians beside the
   originals.

WHAT THIS IS NOT
----------------
This is NOT a re-run of the study under a different metric. The probe
was trained on a split built from the qcov assignment, and the systems
were sampled to balance the qcov cells. Regrouping existing scores
asks a narrower question: holding the fitted probe fixed, does the
ordering persist when the cells are drawn differently?

A full re-run would resample, rebuild the split, and refit. That is a
larger piece of work and would need its own extraction for systems not
in the current set. The narrow version is still informative, because
the probe never saw the cell labels: it was trained on pooled training
systems and the cells enter only at evaluation.

Cells are also no longer balanced after reassignment, so a cell with
very few test systems is reported as such rather than compared.

USAGE
-----
    python3 alternative_sp_check.py \\
        --index probe_index.parquet \\
        --per-system per_system_auprc.parquet \\
        --sp-per-ligand sp_per_ligand_boltz1.parquet \\
        --targets targets_with_sp_boltz1.parquet \\
        --metric sp_pli --threshold 70 \\
        --out alternative_sp.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUADRANTS = ("FF", "FN", "NF", "NN")
MIN_CELL = 20


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-system", required=True)
    ap.add_argument("--sp-per-ligand", required=True,
                    help="per-ligand similarity scores, carrying the "
                         "alternative metrics")
    ap.add_argument("--targets",
                    help="target table carrying fam_f, if the index "
                         "does not")
    ap.add_argument("--metric", default="sp_pli",
                    choices=["sp_qcov", "sp_fident", "sp_pli"])
    ap.add_argument("--threshold", type=float, default=70.0)
    ap.add_argument("--agg", default="min", choices=["min", "max"],
                    help="how a system's per-ligand scores combine "
                         "(default min, as in the primary analysis)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    idx = pd.read_parquet(args.index)
    per = pd.read_parquet(args.per_system)
    lig = pd.read_parquet(args.sp_per_ligand)

    dups = per.duplicated(["system_id", "feature_set", "layer"],
                          keep=False)
    if dups.any():
        ex = per.loc[dups, ["system_id", "feature_set",
                            "layer"]].head(3).to_dict("records")
        sys.exit(f"duplicate per-system scores, e.g. {ex}")
    if args.metric not in lig.columns:
        sys.exit(f"{Path(args.sp_per_ligand).name} has no column "
                 f"'{args.metric}'; it has {list(lig.columns)}")

    # The family axis is held fixed. Only the operational definition of
    # pocket familiarity -- the similarity metric together with its
    # threshold -- is varied.
    if "fam_f" in idx.columns:
        fam = idx[["system_id", "fam_f"]].drop_duplicates()
    elif args.targets:
        tg = pd.read_parquet(args.targets)
        if "fam_f" not in tg.columns:
            sys.exit(f"{Path(args.targets).name} has no fam_f column")
        fam = tg[["system_id", "fam_f"]].drop_duplicates()
    else:
        sys.exit("neither --index nor --targets carries fam_f; the "
                 "family axis cannot be held fixed")

    agg = lig.groupby("system_id")[args.metric]
    sp_new = (agg.min() if args.agg == "min" else agg.max()).rename("sp_new")
    new = (idx[["system_id", "quadrant", "split"]]
           .merge(fam, on="system_id", how="left", validate="one_to_one")
           .merge(sp_new, on="system_id", how="left",
                  validate="one_to_one"))

    # A missing family flag must not reach np.where: NaN is truthy
    # there, which would quietly send a target down the familiar-family
    # branch.
    if new["fam_f"].isna().any():
        bad = new.loc[new["fam_f"].isna(), "system_id"].tolist()
        sys.exit(f"{len(bad)} systems have no fam_f after the merge, "
                 f"e.g. {bad[:5]}.\nThe family axis cannot be held "
                 f"fixed.")
    new["fam_f"] = new["fam_f"].astype(bool)
    # And it must agree with the family letter already encoded in the
    # original quadrant, or "held fixed" is not true.
    from_quadrant = new["quadrant"].str[0].eq("F")
    if not np.array_equal(new["fam_f"].to_numpy(),
                          from_quadrant.to_numpy()):
        n_bad = int((new["fam_f"] != from_quadrant).sum())
        sys.exit(f"fam_f disagrees with the family axis in the original "
                 f"quadrant for {n_bad} systems.\nRefusing to call the "
                 f"family axis fixed.")

    missing = int(new.sp_new.isna().sum())
    if missing:
        print(f"note: {missing} of {len(new)} systems have no "
              f"{args.metric} score and are dropped")
        new = new[new.sp_new.notna()]
    if new.empty:
        sys.exit("no systems carry the alternative metric")

    pocket_familiar = new.sp_new >= args.threshold
    new["quadrant_new"] = np.where(
        new.fam_f,
        np.where(pocket_familiar, "FF", "FN"),
        np.where(pocket_familiar, "NF", "NN"))

    test = new[new.split == "test"]
    moved = int((test.quadrant != test.quadrant_new).sum())
    print(f"\nmetric {args.metric}, {args.agg} over ligands, threshold "
          f"{args.threshold:g}")
    print(f"test systems: {len(test)}, reassigned: {moved} "
          f"({100 * moved / len(test):.1f}%)")
    if moved < 0.05 * len(test):
        print("\n  WARNING: this metric moves almost nobody, so "
              "agreement between the\n  two assignments is not "
              "evidence that the finding is metric-independent.")

    print(f"\n{'from':>6} -> {'to':<6}{'n':>6}")
    for a in QUADRANTS:
        for b in QUADRANTS:
            k = int(((test.quadrant == a) & (test.quadrant_new == b)).sum())
            if k and a != b:
                print(f"{a:>6} -> {b:<6}{k:>6}")

    # Regroup the scores already computed. The probe is untouched: it
    # was trained on pooled training systems and never saw a cell
    # label, so regrouping is legitimate even though the cells were not
    # balanced for this metric.
    # per_system_auprc.parquet already carries quadrant, so merging it
    # against the reassignment table would give quadrant_x and
    # quadrant_y and every reference below would fail. Take only the
    # score columns from the left; the reassignment table is
    # authoritative for both the original and the new cell.
    sc = per[["system_id", "feature_set", "layer", "auprc"]].merge(
        test[["system_id", "quadrant", "quadrant_new"]],
        on="system_id", how="inner", validate="many_to_one")
    results = {"metric": args.metric, "threshold": args.threshold,
               "agg": args.agg, "n_test": int(len(test)),
               "n_reassigned": moved, "by_feature": {}}

    for feature, sub in sc.groupby("feature_set"):
        # Several layers under one feature set would be pooled into a
        # single median and then labelled with whichever layer happened
        # to come first.
        layers = sorted(sub["layer"].unique())
        if len(layers) != 1:
            sys.exit(f"{feature} carries several scored layers "
                     f"{layers}; this audit needs exactly one per "
                     f"feature set.")
        layer = int(layers[0])
        print(f"\n{feature} (layer {layer})")
        print(f"{'cell':6}{'n orig':>8}{'median':>9}   "
              f"{'n new':>7}{'median':>9}")
        cells = {}
        for q in QUADRANTS:
            o = sub[sub.quadrant == q]
            n_ = sub[sub.quadrant_new == q]
            om = f"{o.auprc.median():.3f}" if len(o) else "--"
            nm = (f"{n_.auprc.median():.3f}" if len(n_) >= MIN_CELL
                  else f"({len(n_)} too few)")
            print(f"{q:6}{len(o):>8}{om:>9}   {len(n_):>7}{nm:>9}")
            cells[q] = {
                "n_original": int(len(o)),
                "median_original": (round(float(o.auprc.median()), 4)
                                    if len(o) else None),
                "n_new": int(len(n_)),
                "median_new": (round(float(n_.auprc.median()), 4)
                               if len(n_) >= MIN_CELL else None),
            }
        results["by_feature"][feature] = {"layer": layer, "cells": cells}

    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print("\n  The probe is unchanged; only the cell assignment is. The\n"
          "  question is whether the ordering persists when the cells\n"
          "  are drawn by a metric that moves this many systems.\n")
    print("  This is not a re-run under the alternative definition: the\n"
          "  sample was balanced for qcov at 80 and the split was built\n"
          "  from it, so the new cells are unbalanced and some may be\n"
          "  too small to compare.\n")
    print("  Report this as robustness to an alternative "
          "operationalisation of\n  pocket familiarity (metric and "
          "threshold together), not to a change\n  of metric alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
