#!/usr/bin/env python3
"""
Does the FN deficit live in its covalent systems, or is it spread
across the cell?

THE QUESTION THIS SEPARATES
---------------------------
The composition check found FN holds covalent ligands more often than
NF, 0.254 against 0.110, with the interval excluding zero. That is a
difference in what the cells CONTAIN. It does not say the covalent
systems are the hard ones.

Two worlds produce that same enrichment:

  CONCENTRATED. FN's covalent systems score badly and its
  non-covalent systems score like NF. Then the cell looks hard
  because it holds hard cases, and "covalent sites are hard to read
  from this representation" is a mechanism with a name and a
  testable consequence.

  UNIFORM. Both halves of FN score low. Then covalent enrichment is a
  property of the cell that happens to be true and does not explain
  the deficit, and the composition result stays a confounder rather
  than becoming an explanation.

The difference between those is the difference between a correlation
and a lead.

DECLARED BEFORE RUNNING
-----------------------
The quantity is the within-cell gap: median AUPRC of a cell's
covalent systems minus median AUPRC of its non-covalent systems. It is
computed for FN, which is the cell in question, and for NF, which is
the matched control. A within-cell gap that appears in FN and not in
NF means something about FN; one that appears in both is about
covalent ligands generally and is a different claim.

Intervals come from resampling receptor-family clusters, as elsewhere.

THE SAMPLE IS SMALL AND THAT IS STATED IN ADVANCE
-------------------------------------------------
FN holds about 122 test systems and roughly a quarter are covalent, so
the comparison is something like 30 against 92. The interval will be
wide. Both halves are reported whatever they show, and a gap whose
interval includes zero is reported as not resolving rather than as
evidence of no difference: a wide interval around a large point
estimate is ignorance, not a null.

The script refuses to compare a half below a declared floor, because a
median over a handful of proteins is not evidence of the same kind as
a median over ninety.

That floor may well bind. NF holds covalent ligands in about 11 per
cent of roughly 136 systems, which is around fifteen -- the floor
itself. If the comparison comes back unmade, the honest reading is
that the covalent account could not be resolved at the declared
minimum subgroup size, and the floor stays where it was set. Lowering
it after seeing the count would be choosing the threshold that lets
the test run.

USAGE
-----
    python3 covalent_split_check.py \\
        --index probe_index.parquet \\
        --per-system per_system_auprc.parquet \\
        --plinder-dir ~/.local/share/plinder/2026-09 \\
        --out covalent_split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FLAG = "ligand_is_covalent"
CELLS = ("FN", "NF")          # the cell in question, and its control
OTHER = ("FF", "NN")          # reported, not part of the contrast
MIN_HALF = 15                 # fixed here, not passed in


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-system", required=True,
                    help="per-system AUPRC at the peak layer, from "
                         "per_system_scores.py")
    ap.add_argument("--plinder-dir", required=True,
                    help="the release directory the study used; read "
                         "as files, since the package substitutes "
                         "another release silently")
    ap.add_argument("--feature", default="s")
    ap.add_argument("--layer", type=int,
                    help="the layer the MAIN analysis fixed. Required "
                         "when the score file holds more than one, "
                         "and not inferred: choosing the layer by its "
                         "median here would pick the best test layer "
                         "for this secondary analysis, when the point "
                         "is to interrogate the representation the "
                         "primary result already named.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.n_boot <= 0:
        sys.exit("--n-boot must be positive")
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    idx = pd.read_parquet(args.index)
    scores = pd.read_parquet(args.per_system)
    for c in ("system_id", "quadrant", "split", "family_cluster"):
        if c not in idx.columns:
            sys.exit(f"--index has no '{c}'")
    if "auprc" not in scores.columns:
        sys.exit("--per-system has no 'auprc'")
    if "feature_set" in scores.columns:
        scores = scores[scores.feature_set == args.feature]
        if scores.empty:
            sys.exit(f"no scores for feature set '{args.feature}'")
    selected_layer = None
    if "layer" in scores.columns:
        layers = sorted(int(x) for x in scores.layer.unique())
        # Validated whether the file holds one layer or many. A single
        # -layer file with the wrong --layer would otherwise be
        # accepted and the wrong number recorded as provenance.
        if args.layer is not None and args.layer not in layers:
            sys.exit(f"requested layer {args.layer} is absent from "
                     f"--per-system; it holds {layers}.")
        if len(layers) > 1:
            if args.layer is None:
                sys.exit(f"--per-system holds {len(layers)} layers; "
                         f"supply the layer the main analysis fixed "
                         f"with --layer.\nSelecting it here by median "
                         f"would choose the best test layer for this "
                         f"analysis.")
            selected_layer = args.layer
            scores = scores[scores.layer == selected_layer]
        else:
            selected_layer = layers[0]
        print(f"layer {selected_layer} (fixed externally)")

    test = idx[idx.split == "test"][
        ["system_id", "quadrant", "family_cluster"]]

    # One score per system, and a score for every system. An inner
    # merge would drop an unscored system quietly, and a duplicated
    # one would weight it twice; both change a median without
    # announcing it.
    dup = scores.system_id.duplicated(keep=False)
    if dup.any():
        sys.exit(f"{scores.loc[dup, 'system_id'].nunique()} systems "
                 f"have more than one score row after filtering on "
                 f"feature set and layer, e.g. "
                 f"{scores.loc[dup, 'system_id'].head(3).tolist()}.")
    ann_p = (Path(args.plinder_dir).expanduser() / "index"
             / "annotation_table.parquet")
    if not ann_p.exists():
        sys.exit(f"{ann_p} does not exist.")
    pl = pd.read_parquet(ann_p, columns=["system_id", FLAG])

    # Covalent at the SYSTEM level: any ligand of the system is
    # covalent. The same "any ligand" rule the composition check used,
    # so the two analyses are splitting on the same thing.
    cov = pl.groupby("system_id")[FLAG].max()
    d = test.merge(scores[["system_id", "auprc"]], on="system_id",
                   how="inner", validate="one_to_one")
    if len(d) != len(test):
        sys.exit(f"only {len(d):,} of {len(test):,} test systems have "
                 f"an AUPRC score. A median over the scored subset is "
                 f"not\nthe quantity this compares.")
    d = d.merge(cov.rename("covalent"), on="system_id", how="left",
                validate="one_to_one")
    if d.covalent.isna().any():
        n = int(d.covalent.isna().sum())
        sys.exit(f"{n} test systems carry no covalent flag; the "
                 f"release is probably not the study's.")
    d["covalent"] = d.covalent.astype(bool)
    print(f"test systems scored: {len(d):,}")

    rng = np.random.default_rng(args.seed)
    clusters = d.family_cluster.unique()
    by_cl = {c: g for c, g in d.groupby("family_cluster")}

    def halves(frame, cell):
        s = frame[frame.quadrant == cell]
        return (s[s.covalent], s[~s.covalent])

    def gap(frame, cell):
        a, b = halves(frame, cell)
        if len(a) < 1 or len(b) < 1:
            return float("nan")
        return float(a.auprc.median() - b.auprc.median())

    print(f"\n{'cell':6}{'covalent n':>12}{'median':>9}"
          f"{'other n':>10}{'median':>9}{'within gap':>12}")
    results = {}
    for cell in CELLS + OTHER:
        a, b = halves(d, cell)
        g = gap(d, cell)
        small = len(a) < MIN_HALF or len(b) < MIN_HALF
        results[cell] = {
            "n_covalent": int(len(a)),
            "n_other": int(len(b)),
            "median_covalent": (round(float(a.auprc.median()), 4)
                                if len(a) else None),
            "median_other": (round(float(b.auprc.median()), 4)
                             if len(b) else None),
            "within_gap": None if np.isnan(g) else round(g, 4),
            "below_floor": bool(small)}
        print(f"{cell:6}{len(a):>12,}"
              f"{a.auprc.median() if len(a) else float('nan'):>9.3f}"
              f"{len(b):>10,}"
              f"{b.auprc.median() if len(b) else float('nan'):>9.3f}"
              f"{g:>+12.3f}" + ("  (small)" if small else ""))

    # The contrast: does FN's within-cell gap differ from NF's? A gap
    # in both is about covalent ligands; a gap in FN alone is about FN.
    draws = np.empty(args.n_boot)
    for i in range(args.n_boot):
        rep = pd.concat([by_cl[c] for c in
                         rng.choice(clusters, len(clusters),
                                    replace=True)], ignore_index=True)
        draws[i] = gap(rep, CELLS[0]) - gap(rep, CELLS[1])
    draws = draws[~np.isnan(draws)]
    if len(draws) < args.n_boot * 0.5:
        print(f"\n  {args.n_boot - len(draws):,} of {args.n_boot:,} "
              f"replicates had an empty half and were dropped. The "
              f"interval\n  below rests on the remainder and should "
              f"be read as unstable.")
    est = gap(d, CELLS[0]) - gap(d, CELLS[1])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    excl = bool(lo > 0 or hi < 0)

    print(f"\n{CELLS[0]} within-gap minus {CELLS[1]} within-gap")
    print(f"  {est:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    # When the floor binds, the comparison was not eligible. The
    # estimate and interval are still recorded, because deleting them
    # would hide what was computed, but comparison_made says they are
    # not to be read as a result.
    floor_hit = any(results[c]["below_floor"] for c in CELLS)
    if floor_hit:
        verdict = (f"a half falls below {MIN_HALF} systems, so the "
                   f"comparison is not\n  made: a median over a "
                   f"handful of proteins is not evidence of the same "
                   f"kind")
    elif excl and est < 0:
        verdict = ("FN's covalent systems are worse than its others "
                   "by more than NF's are,\n  which is what a "
                   "concentrated deficit looks like. A candidate "
                   "mechanism,\n  on a small sample")
    elif excl:
        verdict = ("the difference runs the other way: FN's covalent "
                   "systems are RELATIVELY\n  better, which the "
                   "concentration account does not predict")
    else:
        verdict = ("the interval includes zero. On this sample the "
                   "deficit does not resolve\n  as concentrated in "
                   "the covalent systems, which is not the same as "
                   "showing\n  it is spread evenly: the interval is "
                   "wide and the halves are small")
    print(f"\n  {verdict}.")

    results["_meta"] = {
        "feature": args.feature, "flag": FLAG,
        "contrast": f"{CELLS[0]} within-gap - {CELLS[1]} within-gap",
        "comparison_made": not floor_hit,
        "estimate": round(est, 4), "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "excludes_zero": (None if floor_hit else excl),
        "excludes_zero_note":
            ("a half fell below the declared minimum subgroup size, "
             "so the comparison was not eligible; the estimate and "
             "interval are recorded but are not a result"
             if floor_hit else None),
        "layer": selected_layer,
        "min_half": MIN_HALF, "n_boot": args.n_boot,
        "seed": args.seed, "resampled": "family_cluster",
        "n_replicates_used": int(len(draws)),
        "plinder_dir": str(Path(args.plinder_dir).expanduser())}
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print("\n  A within-cell gap says where the deficit sits, not why. "
          "Covalent ligands\n  attaching at sites that need not "
          "present as pockets is a story this test\n  is consistent "
          "with, not one it establishes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
