#!/usr/bin/env python3
"""
Quadrant feasibility check for the two-axis pocket-generalization design.

WHAT THIS IS FOR
----------------
The study assigns every evaluation target to one of four regimes from two
familiarity axes -- protein-family familiarity S_f and pocket familiarity
S_p:

                      familiar pocket   novel pocket
    familiar family         FF               FN
    novel family            NF               NN

Two of the study's own kill criteria are decided here, BEFORE any
representation is extracted:

  KC1  S_f and S_p are so strongly correlated that all four quadrants
       cannot be populated with meaningful sample sizes.
  KC2  The NN regime is too small for stable pocket-level evaluation.

Both are properties of the dataset and the chosen thresholds alone. Neither
requires a model, a GPU, or a single forward pass. Running this first means
a design that cannot work is found now rather than after an extraction run.

WHY A SWEEP RATHER THAN ONE ANSWER
----------------------------------
The quadrant assignment depends on two thresholds that the protocol requires
to be fixed in advance. Fixing them well requires knowing how the counts
behave across plausible values -- but choosing them to maximise a downstream
result would be exactly the post-hoc selection the study is designed to
avoid. The intended use is therefore: sweep, inspect the counts, pick a
threshold pair on sample-size grounds alone, record it, and never revisit it
once performance numbers exist.

THE COUNT THAT MATTERS IS NOT THE ROW COUNT
-------------------------------------------
Several pockets can belong to one protein, and several proteins to one
sequence cluster. Those are not independent evaluation units: the study's
own statistical plan calls for cluster-aware resampling for exactly this
reason. A quadrant holding 400 systems drawn from 6 clusters supports far
less than one holding 400 systems from 200 clusters. Every count below is
therefore reported three ways -- systems, distinct proteins, and distinct
family clusters -- and the kill-criterion verdict is taken on CLUSTERS.

INPUT
-----
A table (parquet or csv) with one row per candidate evaluation system and,
at minimum, columns giving:

    an id                  --id-col            (default: system_id)
    family familiarity     --sf-col            continuous; higher = more familiar
    pocket familiarity     --sp-col            continuous; higher = more familiar
    a family cluster id    --cluster-col       for cluster-aware counting
    a protein id           --protein-col       optional, for a middle count

For PLINDER, S_f is naturally built from its MMseqs2 protein similarity and
S_p from its Foldseek pocket similarity, each taken as the maximum
similarity from the target to any system available before the temporal
cutoff. Computing those maxima is upstream of this script: it consumes the
per-target values, whatever their provenance, so the same code serves any
dataset.

The cutoff itself must sit AFTER the frozen trunk's training cutoff, or
"novel" is not novel to the model. This script cannot check that and does
not try to; it is recorded in the output manifest for the record.

USAGE
-----
    # see what the output looks like before the real table exists
    python3 quadrant_feasibility.py --synthetic

    # a real table, sweeping the default threshold grid
    python3 quadrant_feasibility.py --input plinder_targets.parquet \\
        --sf-col max_seqid_pre --sp-col max_pocket_sim_pre \\
        --cluster-col family_cluster --out feasibility.json

    # one pre-registered threshold pair
    python3 quadrant_feasibility.py --input t.parquet \\
        --sf-thresholds 0.30 --sp-thresholds 0.50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas --break-system-packages")


# Minimum distinct family clusters in a quadrant for it to be considered
# usable. Protein-level bootstrap intervals over fewer than this are too
# coarse to support a primary endpoint. Override with --min-clusters.
DEFAULT_MIN_CLUSTERS = 30

QUADRANTS = ["FF", "FN", "NF", "NN"]


# ----------------------------------------------------------------------
# Synthetic table, for seeing the output shape before real data exists
# ----------------------------------------------------------------------

def make_synthetic(n=4000, rho=0.55, seed=0):
    """A stand-in target table with a tunable S_f/S_p correlation.

    rho is the thing KC1 is about. Real pocket data has S_f and S_p
    positively correlated -- a protein from a well-represented family
    usually has a well-represented pocket -- and the question is whether
    that correlation still leaves the off-diagonal cells populated. Vary
    --synthetic-rho to see how the verdict moves.
    """
    rng = np.random.default_rng(seed)
    cov = [[1.0, rho], [rho, 1.0]]
    z = rng.multivariate_normal([0, 0], cov, size=n)
    # Map to [0,1] similarity-like scales via the normal CDF.
    from math import erf, sqrt
    cdf = np.vectorize(lambda v: 0.5 * (1 + erf(v / sqrt(2))))
    sf, sp = cdf(z[:, 0]), cdf(z[:, 1])

    # Clusters: a heavy-tailed assignment, so a few clusters hold many
    # systems and most hold one or two -- the real shape, and the reason
    # row counts overstate usable sample size.
    n_clusters = max(40, n // 12)
    weights = 1.0 / np.arange(1, n_clusters + 1) ** 1.1
    weights /= weights.sum()
    cluster = rng.choice(n_clusters, size=n, p=weights)
    protein = cluster * 1000 + rng.integers(0, 6, size=n)

    return pd.DataFrame({
        "system_id": [f"sys_{i:06d}" for i in range(n)],
        "S_f": sf, "S_p": sp,
        "family_cluster": [f"c{c}" for c in cluster],
        "protein_id": [f"p{p}" for p in protein],
    })


# ----------------------------------------------------------------------
# Core counting
# ----------------------------------------------------------------------

def assign_quadrants(df, sf_col, sp_col, sf_thr, sp_thr):
    """Label each row FF/FN/NF/NN.

    A target is FAMILIAR on an axis when its similarity to the pre-cutoff
    pool is at or above the threshold. Note the asymmetry this creates and
    that the manuscript must state: 'novel' here means 'below a stated
    similarity threshold', not 'demonstrably absent from pretraining'.
    """
    fam_f = df[sf_col] >= sf_thr
    fam_p = df[sp_col] >= sp_thr
    q = np.where(fam_f & fam_p, "FF",
        np.where(fam_f & ~fam_p, "FN",
        np.where(~fam_f & fam_p, "NF", "NN")))
    return pd.Series(q, index=df.index)


def count_cell(sub, cluster_col, protein_col):
    return {
        "systems": int(len(sub)),
        "proteins": int(sub[protein_col].nunique()) if protein_col else None,
        "clusters": int(sub[cluster_col].nunique()),
    }


def evaluate_pair(df, sf_col, sp_col, cluster_col, protein_col,
                  sf_thr, sp_thr, min_clusters):
    """Counts and kill-criterion verdicts for one threshold pair."""
    q = assign_quadrants(df, sf_col, sp_col, sf_thr, sp_thr)
    cells = {name: count_cell(df[q == name], cluster_col, protein_col)
             for name in QUADRANTS}

    nn_clusters = cells["NN"]["clusters"]
    weakest = min(c["clusters"] for c in cells.values())
    empty = [n for n in QUADRANTS if cells[n]["systems"] == 0]

    # KC1 asks whether ALL FOUR cells are usable, so it is decided by the
    # weakest cell, not by NN alone -- a design with a healthy NN but an
    # empty FN cannot separate which axis drives the gap, which is what FN
    # and NF exist to do.
    kc1_pass = (not empty) and weakest >= min_clusters
    kc2_pass = nn_clusters >= min_clusters

    return {
        "sf_threshold": float(sf_thr),
        "sp_threshold": float(sp_thr),
        "cells": cells,
        "weakest_cell_clusters": int(weakest),
        "nn_clusters": int(nn_clusters),
        "empty_cells": empty,
        "kc1_all_quadrants_usable": bool(kc1_pass),
        "kc2_nn_usable": bool(kc2_pass),
        "usable": bool(kc1_pass and kc2_pass),
    }


def correlation_report(df, sf_col, sp_col):
    """S_f vs S_p association, on both the raw and rank scales.

    Spearman is the more useful summary of the two, because it is not
    disturbed by the arbitrary scale of either similarity measure. But it
    summarizes monotone association only; it does NOT by itself determine
    quadrant occupancy. Two datasets with identical Spearman correlation can
    produce very different FF/FN/NF/NN counts, because occupancy depends on
    the joint distribution and on where the thresholds fall within it. The
    sweep below is what answers the feasibility question; these coefficients
    describe how redundant the two axes are, which is a related but distinct
    thing.
    """
    sf, sp = df[sf_col].to_numpy(), df[sp_col].to_numpy()
    ok = np.isfinite(sf) & np.isfinite(sp)
    sf, sp = sf[ok], sp[ok]
    pearson = float(np.corrcoef(sf, sp)[0, 1])
    rs = pd.Series(sf).rank().to_numpy()
    rp = pd.Series(sp).rank().to_numpy()
    spearman = float(np.corrcoef(rs, rp)[0, 1])
    return {"n": int(ok.sum()), "pearson": pearson, "spearman": spearman}


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def print_cells(res, protein_col):
    hdr = f"    {'':4} {'systems':>9} {'proteins':>9} {'clusters':>9}"
    print(hdr)
    for name in QUADRANTS:
        c = res["cells"][name]
        prot = "--" if c["proteins"] is None else f"{c['proteins']:,}"
        flag = "  <-- NN" if name == "NN" else ""
        print(f"    {name:4} {c['systems']:>9,} {prot:>9} "
              f"{c['clusters']:>9,}{flag}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="parquet or csv target table")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on a generated stand-in table instead")
    ap.add_argument("--synthetic-rho", type=float, default=0.55,
                    help="S_f/S_p correlation of the synthetic table "
                         "(default 0.55)")
    ap.add_argument("--synthetic-n", type=int, default=4000)
    ap.add_argument("--id-col", default="system_id")
    ap.add_argument("--sf-col", default="S_f")
    ap.add_argument("--sp-col", default="S_p")
    ap.add_argument("--cluster-col", default="family_cluster")
    ap.add_argument("--protein-col", default="protein_id")
    ap.add_argument("--sf-thresholds", type=float, nargs="+",
                    default=[0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--sp-thresholds", type=float, nargs="+",
                    default=[0.30, 0.40, 0.50, 0.60])
    ap.add_argument("--min-clusters", type=int, default=DEFAULT_MIN_CLUSTERS,
                    help=f"distinct family clusters a quadrant needs to be "
                         f"usable (default {DEFAULT_MIN_CLUSTERS})")
    ap.add_argument("--cutoff-date", default=None,
                    help="temporal cutoff, recorded in the manifest. It must "
                         "fall AFTER the frozen trunk's own training cutoff; "
                         "this script records but cannot verify that.")
    ap.add_argument("--trunk", default=None,
                    help="frozen trunk name, recorded in the manifest")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)

    # --- load ----------------------------------------------------------
    if args.synthetic:
        df = make_synthetic(args.synthetic_n, args.synthetic_rho)
        args.sf_col, args.sp_col = "S_f", "S_p"
        args.cluster_col, args.protein_col = "family_cluster", "protein_id"
        source = (f"SYNTHETIC (n={args.synthetic_n}, "
                  f"rho={args.synthetic_rho})")
    elif args.input:
        p = Path(args.input)
        if not p.exists():
            sys.exit(f"No such file: {p}")
        df = (pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq")
              else pd.read_csv(p))
        source = str(p)
    else:
        sys.exit("Give --input TABLE, or --synthetic to see the output shape.")

    # Fail loudly and helpfully on column mismatch: dataset schemas differ
    # and guessing silently would quietly analyse the wrong column.
    need = [args.sf_col, args.sp_col, args.cluster_col]
    if args.protein_col and args.protein_col in df.columns:
        need.append(args.protein_col)
    else:
        args.protein_col = None
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"Missing column(s): {', '.join(missing)}", file=sys.stderr)
        print(f"\nAvailable columns ({len(df.columns)}):", file=sys.stderr)
        for c in list(df.columns)[:60]:
            print(f"  {c}", file=sys.stderr)
        sys.exit(1)

    # Coerce the two axes to numeric before anything touches them. A string
    # column (an identity written as "73.2%", say) otherwise reaches
    # np.isfinite and raises an opaque TypeError from inside NumPy.
    for col in (args.sf_col, args.sp_col):
        if not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() == 0:
                sys.exit(f"Column '{col}' is not numeric and could not be "
                         f"parsed as a number. First values: "
                         f"{list(df[col].head(3))}")
            print(f"NOTE: column '{col}' was not numeric; coerced "
                  f"({int(coerced.isna().sum() - df[col].isna().sum())} "
                  f"value(s) became NaN).", file=sys.stderr)
            df[col] = coerced

    before = len(df)
    df = df[np.isfinite(df[args.sf_col]) & np.isfinite(df[args.sp_col])]
    dropped = before - len(df)
    if len(df) == 0:
        sys.exit("No rows left with finite S_f and S_p.")

    print("=" * 70)
    print("QUADRANT FEASIBILITY: two-axis pocket-generalization design")
    print("=" * 70)
    print(f"source        : {source}")
    print(f"systems       : {len(df):,}" +
          (f"  ({dropped:,} dropped for non-finite S_f/S_p)" if dropped else ""))
    print(f"S_f column    : {args.sf_col}")
    print(f"S_p column    : {args.sp_col}")
    print(f"cluster column: {args.cluster_col}")
    print(f"usable cell   : >= {args.min_clusters} distinct family clusters")
    if args.trunk:
        print(f"trunk         : {args.trunk}")
    if args.cutoff_date:
        print(f"cutoff        : {args.cutoff_date}  (must postdate the "
              f"trunk's training cutoff)")

    # --- axis ranges, and whether the thresholds live inside them -------
    # A silent unit mismatch is the most dangerous failure this script can
    # have: sequence identity supplied on a 0-100 scale against thresholds
    # meant for 0-1 puts every target on one side of the cut, collapses two
    # quadrants, and produces a confident KILL verdict that is an artefact
    # of units rather than a fact about the data. Printing the observed
    # range next to the requested thresholds makes that visible at a glance.
    ranges = {}
    print()
    for label, col, thrs in (("S_f", args.sf_col, args.sf_thresholds),
                             ("S_p", args.sp_col, args.sp_thresholds)):
        v = df[col].to_numpy()
        lo, med, hi = float(v.min()), float(np.median(v)), float(v.max())
        ranges[label] = {"min": lo, "median": med, "max": hi}
        fracs = [(t, float((v >= t).mean())) for t in thrs]
        ranges[label]["familiar_fraction_by_threshold"] = {
            f"{t:g}": f for t, f in fracs}
        shown = "  ".join(f"{t:g}:{f:.0%}" for t, f in fracs)
        print(f"{label} range      : {lo:.4g} .. {hi:.4g}  "
              f"(median {med:.4g})")
        print(f"{label} familiar at : {shown}")
        # Range membership is not the right test -- a threshold can sit
        # inside the observed range and still put 99% of targets on one
        # side. What matters is the SPLIT each threshold induces.
        lopsided = [(t, f) for t, f in fracs if f > 0.95 or f < 0.05]
        if len(lopsided) == len(thrs):
            print(f"  WARNING: every {label} threshold splits the data more "
                  f"extremely than 95/5.\n  Two quadrants will be nearly "
                  f"empty at every setting, and the resulting\n  kill verdict "
                  f"would reflect the thresholds rather than the data. If "
                  f"{label}\n  is on a 0-100 scale while the thresholds "
                  f"assume 0-1 (median here is\n  {med:.4g}), rescale or pass "
                  f"thresholds on the matching scale.")
        elif lopsided:
            print(f"  NOTE: {len(lopsided)} of {len(thrs)} {label} thresholds "
                  f"split more extremely than 95/5.")

    # --- KC1 part one: the correlation itself --------------------------
    corr = correlation_report(df, args.sf_col, args.sp_col)
    print(f"\nS_f vs S_p association: Pearson {corr['pearson']:+.3f}, "
          f"Spearman {corr['spearman']:+.3f}")
    print("  Spearman summarizes monotone association between the two "
          "familiarity measures.\n  Quadrant feasibility is determined by "
          "the joint distribution and the chosen\n  thresholds, not by this "
          "coefficient; read the sweep below for that.")
    if corr["spearman"] > 0.9:
        print("  WARNING: near-collinear. The two axes are close to one "
              "measurement,\n  and FN/NF will be thin at every threshold.")

    # --- sweep ---------------------------------------------------------
    print(f"\n{'-' * 70}")
    print(f"Threshold sweep ({len(args.sf_thresholds)} x "
          f"{len(args.sp_thresholds)} pairs), cluster counts per quadrant")
    print(f"{'-' * 70}")
    print(f"{'S_f':>6} {'S_p':>6} {'FF':>8} {'FN':>8} {'NF':>8} {'NN':>8}"
          f"   verdict")

    sweep = []
    for sf_thr in args.sf_thresholds:
        for sp_thr in args.sp_thresholds:
            r = evaluate_pair(df, args.sf_col, args.sp_col,
                              args.cluster_col, args.protein_col,
                              sf_thr, sp_thr, args.min_clusters)
            sweep.append(r)
            cl = {n: r["cells"][n]["clusters"] for n in QUADRANTS}
            if r["usable"]:
                verdict = "usable"
            elif not r["kc2_nn_usable"] and not r["kc1_all_quadrants_usable"]:
                verdict = "KC1+KC2 fail"
            elif not r["kc2_nn_usable"]:
                verdict = "KC2 fail (NN thin)"
            else:
                verdict = "KC1 fail (a cell thin)"
            print(f"{sf_thr:>6.2f} {sp_thr:>6.2f} {cl['FF']:>8,} "
                  f"{cl['FN']:>8,} {cl['NF']:>8,} {cl['NN']:>8,}   {verdict}")

    # --- verdict -------------------------------------------------------
    usable = [r for r in sweep if r["usable"]]
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print("=" * 70)

    if not usable:
        best = max(sweep, key=lambda r: r["weakest_cell_clusters"])
        print(f"  NO threshold pair populates all four quadrants with "
              f">= {args.min_clusters} clusters.")
        print(f"  Closest: S_f={best['sf_threshold']:.2f}, "
              f"S_p={best['sp_threshold']:.2f}, weakest cell "
              f"{best['weakest_cell_clusters']} clusters")
        print_cells(best, args.protein_col)
        print("\n  This is kill criterion 1 or 2 firing. The options are to "
              "widen the\n  candidate pool, relax --min-clusters with an "
              "explicit justification,\n  or narrow the design to the "
              "continuous S_f/S_p analysis, which does\n  not require four "
              "populated cells, and drop the 2x2 endpoint.")
    else:
        # Among usable pairs, identify the most balanced design rather than
        # the largest NN: the four-cell comparison is only as strong as its
        # weakest cell, and maximising NN alone tends to starve FN or NF.
        #
        # This is deliberately reported as MOST BALANCED, not as
        # "recommended". Balance is a sample-size property, and a threshold
        # pair chosen to maximise it has no external meaning: a reviewer can
        # fairly ask whether "novel family" and "novel pocket" were defined
        # to manufacture workable cells. The scientific thresholds should
        # have a defensible interpretation of their own -- a pre-declared
        # sequence-identity cutoff, a pocket-similarity cutoff taken from
        # the benchmark definition -- and this diagnostic only says whether
        # such a choice leaves enough data to proceed.
        best = max(usable, key=lambda r: (r["weakest_cell_clusters"],
                                          r["nn_clusters"]))
        print(f"  {len(usable)} of {len(sweep)} threshold pairs are usable.")
        print(f"\n  Most balanced pair: S_f={best['sf_threshold']:.2f}, "
              f"S_p={best['sp_threshold']:.2f}")
        print_cells(best, args.protein_col)
        print(f"\n  weakest cell: {best['weakest_cell_clusters']} clusters; "
              f"NN: {best['nn_clusters']} clusters")
        print("\n  This is a FEASIBILITY diagnostic, not a scientific "
              "threshold recommendation.\n  Balance is a sample-size "
              "property; the thresholds you adopt should have an\n  external "
              "justification (a pre-declared identity cutoff, a benchmark's "
              "own\n  pocket-similarity definition), with this sweep showing "
              "only that the\n  resulting cells are large enough to analyse. "
              "Record the chosen pair, do\n  not revisit it once any "
              "performance number exists, and report the full\n  sweep so "
              "the choice is visibly independent of results.")

    print("\n  Note for the manuscript: 'novel' here means below a stated "
          "similarity\n  threshold against the pre-cutoff pool, not "
          "demonstrably unseen during\n  pretraining. The NN cell inherits "
          "that limitation and should say so.")

    if args.out:
        payload = {
            "source": source,
            "n_systems": int(len(df)),
            "n_dropped_nonfinite": int(dropped),
            "columns": {"sf": args.sf_col, "sp": args.sp_col,
                        "cluster": args.cluster_col,
                        "protein": args.protein_col},
            "min_clusters": args.min_clusters,
            "axis_ranges": ranges,
            "trunk": args.trunk,
            "cutoff_date": args.cutoff_date,
            "correlation": corr,
            "sweep": sweep,
            "n_usable_pairs": len(usable),
            # Named for what it is: the pair with the largest weakest cell.
            # Not a recommended scientific threshold -- see the note printed
            # alongside it.
            "most_balanced_threshold_pair": best,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
