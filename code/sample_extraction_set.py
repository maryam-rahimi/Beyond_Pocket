#!/usr/bin/env python3
"""
Choose the systems to extract representations for.

Extracting all 25,695 targets is neither affordable nor necessary. At the
median length of 402 residues, the per-residue track and a pooled pair
summary cost about 30 MB per system across 48 layers, so the full set
would need roughly 760 GB before any full pair tensor is stored at all --
and a full pair tensor at one layer alone would exceed the quota. This
script picks a stratified subset instead.

WHY A SUBSET IS ENOUGH
----------------------
The depth analysis estimates an onset per familiarity regime, so what it
needs is enough systems PER QUADRANT for that estimate to be stable, not
the whole pool. A few hundred per cell is the same order as a published
evaluation set in this area. Spending the budget on full layer resolution
rather than on more proteins is the right trade when onset is the primary
endpoint and the quadrant performance comparison is, at least in part, a
replication of results already reported elsewhere.

CLUSTER-AWARE SAMPLING
----------------------
Systems are not independent: many share a receptor-family cluster, and
family sizes are heavily skewed. Sampling systems uniformly would fill a
cell with near-duplicates from a handful of large families, so the
effective sample size would be far smaller than the nominal one and the
bootstrap would understate uncertainty. This script therefore samples
CLUSTERS first and then one system per cluster, which prevents
within-cell over-representation by large receptor families.

Note what that does NOT guarantee: sampling is per cell, so one family
can still contribute a system to FF and another to FN if pocket
familiarity differs across its members. Any later train/test split must
therefore be made cluster-disjoint in its own right; this step does not
do it.

LENGTH CAP
----------
Pair-tensor memory grows with the square of sequence length. The longest
target here is 6,729 residues, whose pair tensor is over 300x the median's
and would not fit on the GPU, let alone on disk. A cap is therefore
unavoidable; it is applied identically to every quadrant and the number of
systems it removes from each is reported, because a cap that bit unevenly
across cells would confound length with familiarity.

PRE-REGISTRATION
----------------
The threshold, the per-cell target, the length cap and the seed are all
fixed here and recorded in the manifest. They are properties of the
sampling design, chosen before any representation is extracted, and
re-running with different values should produce a differently named
manifest rather than overwriting this one.

USAGE
-----
    python3 sample_extraction_set.py \\
        --targets targets_with_sp_boltz1.parquet \\
        --labels pocket_labels_boltz1.parquet \\
        --out extraction_manifest.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUADRANTS = ["FF", "FN", "NF", "NN"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True,
                    help="parquet with system_id, fam_f, S_p, family_cluster")
    ap.add_argument("--labels", required=True,
                    help="pocket label parquet, for length and eligibility")
    ap.add_argument("--threshold", type=int, default=80,
                    help="S_p threshold defining pocket familiarity "
                         "(default 80, the best-balanced point of the sweep)")
    ap.add_argument("--per-cell", type=int, default=500,
                    help="systems to draw per quadrant (default 500)")
    ap.add_argument("--max-length", type=int, default=800,
                    help="longest protein to extract, in residues "
                         "(default 800)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite. Pass another --out.")

    tg = pd.read_parquet(args.targets)
    lb = pd.read_parquet(args.labels)
    for c in ("system_id", "fam_f", "S_p", "family_cluster"):
        if c not in tg.columns:
            sys.exit(f"target table lacks '{c}'")
    for c in ("system_id", "length", "eligible", "n_pocket"):
        if c not in lb.columns:
            sys.exit(f"label table lacks '{c}'")

    df = tg.merge(lb[["system_id", "length", "eligible", "n_pocket"]],
                  on="system_id", how="inner")
    print(f"targets with labels: {len(df):,}")

    # --- filters, each reported ----------------------------------------
    n0 = len(df)
    df = df[df.eligible]
    print(f"  eligible                : {len(df):,} "
          f"(-{n0 - len(df):,})")

    n1 = len(df)
    df = df[df.n_pocket > 0]
    print(f"  has pocket residues     : {len(df):,} (-{n1 - len(df):,})")

    # Familiarity and clustering must be KNOWN before a system can be
    # assigned to a quadrant. This fails closed rather than filtering,
    # because a manifest is part of the pre-registered dataset definition
    # and a missing value here is a silent miscategorisation: NaN >= 80
    # evaluates to False, so an unknown S_p would quietly become
    # "pocket-novel" -- the very cell whose scarcity the design is most
    # concerned about.
    # Unknown FAMILIARITY is fatal: NaN >= 80 evaluates to False, so a
    # missing S_p would quietly become "pocket-novel" -- the cell whose
    # scarcity the design is most concerned about. There is no safe
    # default, so the run stops.
    fatal = {c: int(df[c].isna().sum()) for c in ("fam_f", "S_p")}
    if any(fatal.values()):
        sys.exit(f"missing familiarity fields, refusing to assign "
                 f"quadrants: {fatal}\nThere is no safe default for an "
                 f"unknown axis value; resolve these upstream.")

    # A missing CLUSTER is different. The quadrant is still known; what is
    # unknown is the receptor-family composition, because some chain had
    # no sequence-cluster assignment. Such systems cannot take part in
    # cluster-aware sampling, so they are excluded here -- deliberately,
    # counted, and per cell, since an exclusion that fell unevenly across
    # cells would bias the comparison.
    no_cluster = df.family_cluster.isna()
    if no_cluster.any():
        by_cell = (df.loc[no_cluster]
                   .assign(_q=df.loc[no_cluster, "fam_f"]
                           .map({True: "F", False: "N"}).astype(str)
                           + (df.loc[no_cluster, "S_p"] >= args.threshold)
                           .map({True: "F", False: "N"}).astype(str))
                   ._q.value_counts().to_dict())
        print(f"  no family_cluster       : {len(df) - int((~no_cluster).sum()):,} "
              f"excluded (incomplete receptor-family signature)")
        print(f"    by quadrant: {by_cell}")
        df = df[~no_cluster].copy()
    print(f"  complete design fields  : {len(df):,}")

    n2 = len(df)
    long_by_cell = {}
    df["quadrant"] = (df.fam_f.map({True: "F", False: "N"}).astype(str)
                      + (df.S_p >= args.threshold)
                      .map({True: "F", False: "N"}).astype(str))
    for q in QUADRANTS:
        sub = df[df.quadrant == q]
        long_by_cell[q] = int((sub.length > args.max_length).sum())
    df = df[df.length <= args.max_length]
    print(f"  length <= {args.max_length:<13}: {len(df):,} "
          f"(-{n2 - len(df):,})")
    print(f"    removed per quadrant: {long_by_cell}")
    print("    (reported because a cap biting unevenly across cells would "
          "confound\n     length with familiarity)")

    # --- sample ----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    picked, cell_report = [], {}
    print(f"\nsampling {args.per_cell} per quadrant at threshold "
          f"{args.threshold}, seed {args.seed}")
    for q in QUADRANTS:
        sub = df[df.quadrant == q]
        # np.asarray: .unique() on a string column returns a pandas
        # array, which rng.shuffle warns about and does not guarantee to
        # permute correctly.
        clusters = np.asarray(sub.family_cluster.dropna().unique(),
                              dtype=object)
        rng.shuffle(clusters)

        chosen = []
        for cl in clusters:
            if len(chosen) >= args.per_cell:
                break
            cand = sub[sub.family_cluster == cl]
            # One system per cluster, the shortest, since pair-tensor cost
            # is quadratic and the choice within a cluster is arbitrary on
            # every other axis. This makes the result a COST-AWARE sample
            # rather than a uniform one, so the per-cell length
            # distributions below should be compared before analysis: a
            # systematic length difference across cells would be a
            # confound, not a nuisance.
            chosen.append(cand.loc[cand.length.idxmin()])

        got = pd.DataFrame(chosen) if chosen else sub.head(0)
        short = args.per_cell - len(got)
        if short > 0:
            # Fewer clusters than requested systems. Taking a second
            # system from a cluster is worse than a smaller cell: it
            # inflates the nominal sample without adding independent
            # information. The shortfall is reported instead.
            print(f"  {q}: only {len(clusters):,} clusters available, "
                  f"cell short by {short}")
        picked.append(got)
        cell_report[q] = {
            "systems_available": int(len(sub)),
            "length_p25": int(got.length.quantile(.25)) if len(got) else 0,
            "length_p75": int(got.length.quantile(.75)) if len(got) else 0,
            "clusters_available": int(len(clusters)),
            "sampled": int(len(got)),
            "median_length": int(got.length.median()) if len(got) else 0,
            "median_pocket": int(got.n_pocket.median()) if len(got) else 0,
            "removed_by_length_cap": long_by_cell[q],
        }

    sel = pd.concat(picked, ignore_index=True)

    print(f"\n{'cell':6}{'avail':>9}{'clusters':>10}{'sampled':>9}"
          f"{'len p25':>9}{'med len':>9}{'len p75':>9}{'med pocket':>12}")
    for q in QUADRANTS:
        r = cell_report[q]
        print(f"{q:6}{r['systems_available']:>9,}{r['clusters_available']:>10,}"
              f"{r['sampled']:>9,}{r['length_p25']:>9,}"
              f"{r['median_length']:>9,}{r['length_p75']:>9,}"
              f"{r['median_pocket']:>12,}")
    meds = [cell_report[q]["median_length"] for q in QUADRANTS
            if cell_report[q]["sampled"]]
    if meds and max(meds) > 1.5 * min(meds):
        print("\n  WARNING: median length differs by more than 1.5x across "
              "cells. Because\n  the shortest system per cluster is taken, "
              "length is correlated with cell\n  membership here and must "
              "be controlled for in the analysis.")

    # --- cost ------------------------------------------------------------
    L = sel.length.to_numpy(dtype=np.int64)
    n_layers, cs, cz = 48, 384, 128
    s_bytes = int((L * cs * 2 * n_layers).sum())
    # Pooled pair summary: row mean, column mean and self-pair diagonal,
    # three channels of width cz per residue, as in the zPocket readout.
    pz_bytes = int((L * 3 * cz * 2 * n_layers).sum())
    fullz_final = int((L * L * cz * 2).sum())

    print(f"\nestimated storage for {len(sel):,} systems, fp16:")
    print(f"  s, all {n_layers} layers          : {s_bytes/1e9:6.1f} GB")
    print(f"  pooled z, all {n_layers} layers   : {pz_bytes/1e9:6.1f} GB")
    print(f"  full z, final layer only    : {fullz_final/1e9:6.1f} GB")
    print(f"  total                       : "
          f"{(s_bytes+pz_bytes+fullz_final)/1e9:6.1f} GB")

    manifest = {
        "threshold": args.threshold, "per_cell": args.per_cell,
        "max_length": args.max_length, "seed": args.seed,
        "n_selected": int(len(sel)), "cells": cell_report,
        "storage_estimate_gb": {
            "s_all_layers": round(s_bytes / 1e9, 2),
            "pooled_z_all_layers": round(pz_bytes / 1e9, 2),
            "full_z_final_layer": round(fullz_final / 1e9, 2),
        },
    }
    sel[["system_id", "quadrant", "fam_f", "S_p", "family_cluster",
         "length", "n_pocket"]].to_parquet(out)
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out}")
    print(f"Wrote {out}.manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
