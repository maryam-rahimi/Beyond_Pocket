#!/usr/bin/env python3
"""
Is covalent status associated with lower pocket decodability, pooled
across the four regimes?

WHY THIS AND NOT THE WITHIN-CELL SPLIT
--------------------------------------
The split asked whether the FN deficit concentrates in FN's covalent
systems, by comparing FN's within-cell gap against NF's. That contrast
did not resolve: -0.256, interval -0.429 to +0.242, with NF's covalent
half at exactly the declared floor of fifteen.

But the descriptive numbers underneath it were striking. Covalent
systems scored 0.113 against 0.579 in FF, 0.044 against 0.529 in NN,
and 0.116 against 0.318 in FN. Three large gaps in the same direction.
NF was the exception at +0.054, on fifteen systems.

That pattern is not about FN. It suggests covalent systems are harder
everywhere, which would also explain why the difference of differences
cancelled. This script estimates that pooled effect directly, so the
claim can be stated with an interval rather than as four numbers a
reader is invited to compare by eye.

WHAT IS ESTIMATED
-----------------
Two quantities, both pooled over the four regimes:

  ADJUSTED. The coefficient on covalent status in a least-squares fit
  of AUPRC on covalent status and regime, the regime entered as
  indicators. This is the covalent effect holding the regime fixed,
  which matters because the regimes differ both in difficulty and in
  how often they are covalent: an unadjusted difference would mix the
  two.

  UNADJUSTED. The difference in MEANS, reported alongside. Means and
  not medians: the adjusted quantity is a least-squares coefficient,
  so a median difference beside it would be a different estimand and
  the pair could not be read as before and after adjustment. The
  per-regime medians are reported too, as description, because they
  connect to the rest of the paper -- but they are not the comparator.

Intervals come from resampling receptor-family clusters, as elsewhere,
with the fit recomputed inside each replicate.

THIS IS EXPLORATORY
-------------------
The covalent flag was one of a pre-specified set compared between FN
and NF. Pooling it across regimes is a different question, asked after
seeing that the within-cell gaps pointed the same way in three cells.
It is reported as exploratory and is not a pre-registered test.

WHAT A LARGE EFFECT WOULD AND WOULD NOT MEAN
--------------------------------------------
That covalent systems are harder to predict under this probe. Not that
the representation fails on them for a reason specific to covalency: a
covalent ligand attaches by forming a bond and often sits at a site
that does not present as a cavity, so a low score may reflect that
these labels mark something other than a pocket in the usual sense.
That is a property of the task and the label, not only of the model.

USAGE
-----
    python3 covalent_pooled_check.py \\
        --index probe_index.parquet \\
        --per-system per_system_auprc.parquet \\
        --plinder-dir ~/.local/share/plinder/2026-09 \\
        --feature s --layer 18 \\
        --out covalent_pooled.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FLAG = "ligand_is_covalent"
QUADRANTS = ("FF", "FN", "NF", "NN")


def fit(frame):
    """Covalent coefficient from AUPRC ~ covalent + regime.

    Regime enters as indicators with FF as the reference, so the
    coefficient is the covalent effect within a regime rather than a
    mixture of that and the regimes' differing covalent prevalence.
    Returns nan when a replicate lacks either covalent class, which a
    cluster bootstrap can produce.
    """
    y = frame.auprc.to_numpy(float)
    cov = frame.covalent.to_numpy(float)
    if len(np.unique(cov)) < 2:
        return float("nan")
    cols = [np.ones(len(frame)), cov]
    for q in QUADRANTS[1:]:                  # FF is the reference
        cols.append((frame.quadrant == q).to_numpy(float))
    X = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(beta[1])


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-system", required=True)
    ap.add_argument("--plinder-dir", required=True)
    ap.add_argument("--feature", default="s")
    ap.add_argument("--layer", type=int,
                    help="the layer the main analysis fixed; required "
                         "when the score file holds more than one")
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
    if "feature_set" in scores.columns:
        scores = scores[scores.feature_set == args.feature]
        if scores.empty:
            sys.exit(f"no scores for feature set '{args.feature}'")

    selected_layer = None
    if "layer" in scores.columns:
        layers = sorted(int(x) for x in scores.layer.unique())
        if args.layer is not None and args.layer not in layers:
            sys.exit(f"requested layer {args.layer} is absent from "
                     f"--per-system; it holds {layers}.")
        if len(layers) > 1:
            if args.layer is None:
                sys.exit(f"--per-system holds {len(layers)} layers; "
                         f"supply the layer the main analysis fixed "
                         f"with --layer.")
            selected_layer = args.layer
            scores = scores[scores.layer == selected_layer]
        else:
            selected_layer = layers[0]
        print(f"layer {selected_layer} (fixed externally)")

    dup = scores.system_id.duplicated(keep=False)
    if dup.any():
        sys.exit(f"{scores.loc[dup, 'system_id'].nunique()} systems "
                 f"have more than one score row after filtering.")

    test = idx[idx.split == "test"][
        ["system_id", "quadrant", "family_cluster"]]
    if test.family_cluster.isna().any():
        sys.exit(f"{int(test.family_cluster.isna().sum())} test "
                 f"systems have no family_cluster; the cluster "
                 f"bootstrap is undefined for them.")
    ann_p = (Path(args.plinder_dir).expanduser() / "index"
             / "annotation_table.parquet")
    if not ann_p.exists():
        sys.exit(f"{ann_p} does not exist.")
    cov = (pd.read_parquet(ann_p, columns=["system_id", FLAG])
           .groupby("system_id")[FLAG].max())

    d = test.merge(scores[["system_id", "auprc"]], on="system_id",
                   how="inner", validate="one_to_one")
    if len(d) != len(test):
        sys.exit(f"only {len(d):,} of {len(test):,} test systems have "
                 f"an AUPRC score.")
    d = d.merge(cov.rename("covalent"), on="system_id", how="left",
                validate="one_to_one")
    if d.covalent.isna().any():
        sys.exit(f"{int(d.covalent.isna().sum())} test systems carry "
                 f"no covalent flag; the release is probably not the "
                 f"study's.")
    d["covalent"] = d.covalent.astype(bool)

    n_cov = int(d.covalent.sum())
    print(f"test systems {len(d):,}: {n_cov:,} covalent, "
          f"{len(d) - n_cov:,} not")
    print(f"\n{'regime':8}{'covalent n':>12}{'median':>9}"
          f"{'other n':>10}{'median':>9}")
    per_cell = {}
    for q in QUADRANTS:
        s = d[d.quadrant == q]
        a, b = s[s.covalent], s[~s.covalent]
        per_cell[q] = {
            "n_covalent": int(len(a)), "n_other": int(len(b)),
            "median_covalent": (round(float(a.auprc.median()), 4)
                                if len(a) else None),
            "median_other": (round(float(b.auprc.median()), 4)
                             if len(b) else None)}
        print(f"{q:8}{len(a):>12,}"
              f"{a.auprc.median() if len(a) else float('nan'):>9.3f}"
              f"{len(b):>10,}"
              f"{b.auprc.median() if len(b) else float('nan'):>9.3f}")

    adj = fit(d)
    unadj = float(d[d.covalent].auprc.mean()
                  - d[~d.covalent].auprc.mean())

    rng = np.random.default_rng(args.seed)
    clusters = d.family_cluster.unique()
    by_cl = {c: g for c, g in d.groupby("family_cluster")}
    ba, bu = np.empty(args.n_boot), np.empty(args.n_boot)
    for i in range(args.n_boot):
        rep = pd.concat([by_cl[c] for c in
                         rng.choice(clusters, len(clusters),
                                    replace=True)], ignore_index=True)
        ba[i] = fit(rep)
        a, b = rep[rep.covalent], rep[~rep.covalent]
        bu[i] = (a.auprc.mean() - b.auprc.mean()
                 if len(a) and len(b) else float("nan"))
    ba, bu = ba[~np.isnan(ba)], bu[~np.isnan(bu)]
    alo, ahi = np.percentile(ba, [2.5, 97.5])
    ulo, uhi = np.percentile(bu, [2.5, 97.5])

    print(f"\ncovalent association with mean AUPRC, pooled")
    print(f"  adjusted for regime  {adj:+.3f}  "
          f"[{alo:+.3f}, {ahi:+.3f}]")
    print(f"  unadjusted           {unadj:+.3f}  "
          f"[{ulo:+.3f}, {uhi:+.3f}]")
    excl = bool(alo > 0 or ahi < 0)
    # "On average under an additive model" and not "in every regime".
    # The model fits one coefficient because it contains no
    # covalent-by-regime interaction, so it assumes a common
    # association rather than demonstrating one -- and the per-regime
    # medians above show NF running the other way.
    if excl and adj < 0:
        verdict = ("after adjusting for familiarity regime, covalent "
                   "systems have lower mean\n  AUPRC on average "
                   "under the additive model. The model fits one "
                   "coefficient\n  and so assumes a common "
                   "association; it does not establish one, and the "
                   "per-\n  regime medians above do not all point "
                   "the same way")
    elif excl:
        verdict = ("covalent systems have HIGHER mean AUPRC after "
                   "adjustment, which the\n  descriptive gaps did "
                   "not suggest")
    else:
        verdict = ("the interval includes zero: pooled across "
                   "regimes, covalent status does\n  not resolve as "
                   "an association on this sample")
    print(f"\n  {verdict}.")

    res = {"feature": args.feature, "layer": selected_layer,
           "flag": FLAG, "exploratory": True,
           "n_systems": int(len(d)), "n_covalent": n_cov,
           "per_cell": per_cell,
           "estimand": "association with MEAN AUPRC; observational "
                       "subgrouping, not an effect",
           "adjusted": {"estimate": round(adj, 4),
                        "lo": round(float(alo), 4),
                        "hi": round(float(ahi), 4),
                        "excludes_zero": excl,
                        "model": "auprc ~ covalent + regime, FF "
                                 "reference; additive, no "
                                 "covalent-by-regime interaction, so "
                                 "a common association is assumed "
                                 "rather than shown"},
           "unadjusted": {"statistic": "difference in means",
                          "estimate": round(unadj, 4),
                          "lo": round(float(ulo), 4),
                          "hi": round(float(uhi), 4)},
           "n_boot": args.n_boot, "seed": args.seed,
           "n_replicates_used": int(len(ba)),
           "resampled": "family_cluster",
           "plinder_dir": str(Path(args.plinder_dir).expanduser())}
    out.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out}")
    print("\n  Exploratory: pooling was decided after seeing the "
          "within-cell gaps point\n  the same way in three regimes. A "
          "covalent ligand binds by forming a bond\n  and often sits "
          "where no cavity presents itself, so a low score may say "
          "the\n  label marks something other than a pocket rather "
          "than that the model fails.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
