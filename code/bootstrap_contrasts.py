#!/usr/bin/env python3
"""
Bootstrap intervals for the regime medians and the contrasts between
them.

WHY
---
Every regime comparison reported so far is a point estimate on 122 to
152 test proteins. The claim that carries the finding -- that the
familiar-family/novel-pocket regime sits below the regime novel on
both axes -- is a 2.6-fold difference estimated from 122 and 124
proteins, and nothing has been attached to it that says how much a
different sample of proteins would move it.

WHAT IS RESAMPLED
-----------------
Family clusters, not proteins. Proteins from one receptor family are
not independent: they share sequence, fold and often pocket, so
resampling them individually would treat near-duplicates as separate
evidence and give intervals that are too narrow. A cluster is drawn
with replacement and all of its proteins come with it.

The split was built to be cluster-disjoint for the same reason, so the
resampling unit matches the unit the design already respects.

PAIRED CONTRASTS
----------------
A contrast between two regimes is computed within each replicate,
from the same resample, rather than from two independently resampled
estimates. The regimes share nothing by construction -- they are
disjoint sets of proteins -- but the resample is one draw of the
world, and computing the difference inside it is what makes the
interval an interval on the difference.

WHAT AN INTERVAL HERE DOES AND DOES NOT MEAN
---------------------------------------------
It is a percentile interval on the median AUPRC of the test proteins
in a regime, under resampling of the clusters that produced them. It
carries no correction for the number of contrasts examined, and the
script reports how many were computed so that is visible. Intervals
excluding zero are not thereby significant findings at any particular
level; they are intervals.

USAGE
-----
    python3 bootstrap_contrasts.py \\
        --index probe_index.parquet \\
        --per-system per_system_auprc.parquet \\
        --out bootstrap_contrasts.json \\
        --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUADRANTS = ("FF", "FN", "NF", "NN")
# Every ordered pair, so the sign of each contrast is fixed in advance
# rather than chosen after seeing which way it ran.
CONTRASTS = (("FF", "FN"), ("NF", "NN"), ("FF", "NN"), ("NF", "FF"),
             ("NN", "FN"), ("NF", "FN"))


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-system", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="two-sided; 0.05 gives a 95%% interval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.n_boot <= 0:
        sys.exit("--n-boot must be positive")
    if not 0 < args.alpha < 1:
        sys.exit("--alpha must lie strictly between 0 and 1")

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    idx = pd.read_parquet(args.index)
    per = pd.read_parquet(args.per_system)
    if "family_cluster" not in idx.columns:
        sys.exit("--index has no family_cluster; the resampling unit "
                 "is the cluster, not the protein")
    dup = per.duplicated(["system_id", "feature_set", "layer"],
                         keep=False)
    if dup.any():
        sys.exit(f"duplicate per-system rows, e.g. "
                 f"{per.loc[dup].head(3).to_dict('records')}")

    df = per[["system_id", "feature_set", "layer", "auprc"]].merge(
        idx[["system_id", "quadrant", "family_cluster"]],
        on="system_id", how="inner", validate="many_to_one")
    if df.empty:
        sys.exit("no rows after joining scores to the index")

    # A missing cluster id is fatal, and quietly so without this check:
    # Series.unique() keeps NaN while groupby() drops it, so a drawn
    # NaN would either raise a KeyError or silently shrink the
    # population the resample is drawn from. Cluster identity defines
    # the resampling unit, so a system without one cannot be
    # bootstrapped. Checked after the join, against the systems
    # actually scored.
    if df.family_cluster.isna().any():
        bad = df.loc[df.family_cluster.isna(),
                     "system_id"].unique()[:5].tolist()
        sys.exit(f"family_cluster is missing for scored systems, e.g. "
                 f"{bad}.\nThe cluster bootstrap is undefined for "
                 f"them.")

    rng = np.random.default_rng(args.seed)
    results = {"n_boot": args.n_boot, "alpha": args.alpha,
               "seed": args.seed, "resampled": "family_cluster",
               "n_contrasts_per_feature": len(CONTRASTS),
               "by_feature": {}}
    lo_q, hi_q = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)

    for feature, sub_all in df.groupby("feature_set"):
        layers = sorted(sub_all.layer.unique())
        if len(layers) != 1:
            sys.exit(f"{feature} carries several scored layers "
                     f"{layers}; this needs exactly one per feature "
                     f"set.")
        layer = int(layers[0])
        sub = sub_all

        # One row per cluster, holding its proteins' scores, so a draw
        # takes a cluster and everything in it.
        clusters = sub.family_cluster.unique()
        by_cluster = {c: g for c, g in sub.groupby("family_cluster")}
        n_clusters = len(clusters)

        boot_med = {q: np.empty(args.n_boot) for q in QUADRANTS}
        for b in range(args.n_boot):
            drawn = rng.choice(clusters, size=n_clusters, replace=True)
            rep = pd.concat([by_cluster[c] for c in drawn],
                            ignore_index=True)
            g = rep.groupby("quadrant").auprc.median()
            for q in QUADRANTS:
                boot_med[q][b] = g.get(q, np.nan)

        print(f"\n{feature} (layer {layer}), {n_clusters} clusters, "
              f"{args.n_boot} replicates")
        print(f"{'cell':6}{'n':>5}{'median':>9}{'95% interval':>20}")
        cells = {}
        observed = {}
        for q in QUADRANTS:
            obs = sub[sub.quadrant == q]
            if obs.empty:
                continue
            m = float(obs.auprc.median())
            observed[q] = m
            draws = boot_med[q][~np.isnan(boot_med[q])]
            lo, hi = np.percentile(draws, [lo_q, hi_q])
            cells[q] = {"n": int(len(obs)), "median": round(m, 4),
                        "lo": round(float(lo), 4),
                        "hi": round(float(hi), 4),
                        "n_valid_draws": int(len(draws))}
            print(f"{q:6}{len(obs):>5}{m:>9.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>20}")

        print(f"\n{'contrast':12}{'estimate':>10}{'95% interval':>20}"
              f"{'excludes 0':>12}")
        contrasts = {}
        for a, b_ in CONTRASTS:
            if a not in cells or b_ not in cells:
                continue
            # From the unrounded medians: differencing the rounded
            # ones can move the contrast by up to 1e-4.
            est = observed[a] - observed[b_]
            d = boot_med[a] - boot_med[b_]
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [lo_q, hi_q])
            excl = bool(lo > 0 or hi < 0)
            contrasts[f"{a}-{b_}"] = {
                "estimate": round(est, 4), "lo": round(float(lo), 4),
                "hi": round(float(hi), 4), "excludes_zero": excl}
            print(f"{a}-{b_:9}{est:>+10.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}"
                  f"{'yes' if excl else 'no':>12}")

        results["by_feature"][feature] = {
            "layer": layer, "n_clusters": n_clusters,
            "cells": cells, "contrasts": contrasts}

    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print(f"\n  {len(CONTRASTS)} contrasts per feature set, "
          f"{len(CONTRASTS) * len(results['by_feature'])} in total, "
          f"with no\n  multiplicity correction. An interval excluding "
          f"zero is an interval, not a\n  test at a stated level.")
    print("\n  Clusters are the resampling unit because proteins "
          "within a family share\n  sequence and fold; resampling "
          "proteins would count near-duplicates as\n  independent "
          "evidence and give intervals that are too narrow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
