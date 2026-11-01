#!/usr/bin/env python3
"""
Calibrate the null threshold tau for each layer.

Refits the entire probe pipeline on permuted labels, many times, and
reports the upper quantile of the resulting AUPRCs. That quantile is
tau: the level a depth curve must exceed before any claim is made that
the representation carries pocket information at that layer.

Without this, a depth curve can only be read against the positive rate,
which is the AUPRC of a random ranking of a single protein. That is the
wrong comparison: the pipeline fits a 384-dimensional probe on tens of
thousands of residues and aggregates across proteins, and such a
procedure scores above the positive rate on pure noise. tau measures how
far above.

PERMUTATION IS WITHIN PROTEIN
-----------------------------
Each protein's label vector is shuffled among its own residues. This
preserves every protein's positive rate and the number of proteins, and
destroys only which residues are positive -- the thing the probe is
supposed to learn.

Permuting across proteins instead would also destroy the per-protein
prevalence, so proteins with large pockets would donate positives to
proteins with small ones. The null would then be easier than the real
task for reasons unrelated to the probe, tau would come out too low, and
the onset rule would fire early. Given that this project's calibration
work exists precisely because a naive rule fires nineteen layers early,
an avoidably lax null would be the wrong mistake to make.

WHAT IS PERMUTED
----------------
--permute both (default) permutes training AND test labels: the fully
null pipeline, where no real label information enters at any stage. This
is the null that matches the claim being made, which is about the
procedure end to end.

The alternatives are available and answer narrower questions. `train`
leaves the test labels real and asks what a probe that learned nothing
scores against true pockets. `test` leaves training real and asks what a
genuinely trained probe scores against scrambled truth. They are not
interchangeable, and `both` is the one reported.

PERMUTATIONS ARE SYNCHRONISED ACROSS LAYERS
-------------------------------------------
Replicate b uses the SAME within-protein shuffle at every layer and for
every feature set. Drawing a fresh shuffle per layer would give valid
marginal thresholds and nothing else: replicate 17 at layer 0 would be
unrelated to replicate 17 at layer 1, so the 48 values could not be read
as a curve.

That would be fatal for this study's primary endpoint, which is not a
per-layer test but the first layer beginning five consecutive
exceedances. The false-positive rate of that rule depends entirely on
how correlated neighbouring layers are -- it exists because a
three-layer rule fires nineteen stages early under transformer-like
depth correlation -- and unsynchronised nulls destroy exactly the
correlation structure that needs measuring.

Synchronised replicates give null TRAJECTORIES:

    N_0^(b), N_1^(b), ..., N_47^(b)

from which two things follow. The marginal tau at each layer, as
before; and, directly, how often a null trajectory produces five
consecutive crossings. The second is the empirical check on the onset
rule itself, and it is unavailable from marginal thresholds alone.

Replicate-level AUPRCs are therefore written alongside tau, so the rule
can be audited against the null that calibrated it.

THE THRESHOLD IS AN ORDER STATISTIC
-----------------------------------
tau is the k-th smallest of the B null values, with

    k = ceil((1 - alpha) (B + 1))

not an interpolated quantile. For B = 100 and alpha = 0.05 that is the
96th ordered value, giving an exceedance probability of at most
(100 + 1 - 96) / 101 = 0.0495. Interpolating between null observations
invents a value the null never produced, and for a finite permutation
test the order statistic is both exact and conservative.

COST
----
This is the expensive step: a full null needs n_perm complete refits at
every layer. At roughly fifteen seconds a fit, 100 permutations across
48 layers is about twenty hours for one feature set.

An ISOLATED sparse-layer calibration is pilot-only: interpolating tau
between calibrated layers can create or erase a five-layer run
outright, and the endpoint is defined on exactly such runs.

Computing the layers SEPARATELY is different, and is how the primary
analysis is run. A job array calibrates one layer per task, and
merge_null.py then requires all of them before it will write a
threshold file; the onset audit requires the complete trunk again. A
slice is therefore a valid piece of the production null, and is not on
its own a basis for an onset.

Given a fixed budget, 100 permutations at all 48 layers is worth more
than 200 at a subset: the trajectory structure is what the onset rule
depends on. The default of 100 is the floor at which the threshold's
own variability stops contributing materially to the onset.

USAGE
-----
The primary run takes its scope from the curves it calibrates, so
neither --layers nor --features is given:

    python3 calibrate_null.py \\
        --index probe_index.parquet \\
        --labels pocket_labels.parquet \\
        --curves-meta depth_curves.parquet.meta.json \\
        --out null_tau.parquet \\
        --n-perm 100 --c <FROZEN_C>

--curves-meta is required. It supplies the layers and feature sets and
checks C, max_iter and class weighting, so the null cannot calibrate a
different analysis from the one it is a null for. --pilot runs without
it, for exploratory passes whose output must not be used to threshold
anything.

A sparse-layer run that is never merged cannot support the
pre-registered onset, which requires five adjacent stages:

    python3 calibrate_null.py ... --layers 0 8 16 24 32 40 47

Computing the layers separately and merging them can. That is how the
primary analysis runs, one task per layer:

    python3 calibrate_null.py ... --layers 17 \\
        --out null_s/null_layer_17.parquet

    python3 merge_null.py --indir null_s/ --out null_tau.parquet \\
        --n-layers 48

merge_null.py refuses until every layer is present, so a slice is a
valid piece of the production null and never a basis for an onset on
its own.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.exceptions import ConvergenceWarning
except ImportError as e:
    sys.exit(f"needs scikit-learn: {e}")

QUADRANTS = ("FF", "FN", "NF", "NN")
CLASS_WEIGHT = "balanced"
ALPHA = 0.05           # tau is the (1 - ALPHA) quantile of the null


# The bfloat16 decode lives in load_representations.py and is imported
# rather than copied: a local copy would not receive a correction made
# there, and reading the stored bits as numbers instead of
# reinterpreting them yields values that look entirely like activations.
try:
    from load_representations import load_layer
except ImportError as e:  # noqa: BLE001
    sys.exit(f"needs load_representations.py alongside this script: {e}")


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", nargs="*",
                    choices=["s", "zpool", "both"],
                    help="omit to take the feature sets from "
                         "--curves-meta, which is what makes the null "
                         "follow the run it calibrates")
    ap.add_argument("--layers", nargs="*", type=int,
                    help="layers to calibrate. A subset is allowed, "
                         "and is how the array runs: one layer per "
                         "task, merged afterwards. Production onset "
                         "needs the merged complete calibration, so "
                         "an isolated subset that is never merged is "
                         "pilot-only.")
    ap.add_argument("--n-perm", type=int, default=100,
                    help="null replicates, synchronised across layers "
                         "(default 100)")
    ap.add_argument("--permute", default="both",
                    choices=["both", "train", "test"])
    ap.add_argument("--c", type=float, required=True,
                    help="MUST equal the C used for the real curves: a "
                         "null fitted with a different probe is not a "
                         "null for those curves")
    ap.add_argument("--curves-meta",
                    help="the real run's .meta.json. REQUIRED unless "
                         "--pilot is passed: C, max_iter, class "
                         "weighting, layers and feature sets are all "
                         "taken or checked from it, so the null cannot "
                         "calibrate a different analysis from the one "
                         "it is a null for.")
    ap.add_argument("--pilot", action="store_true",
                    help="run without --curves-meta. For exploratory "
                         "passes only: the result is NOT a null for any "
                         "particular set of curves and must not be used "
                         "to threshold them.")
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite. Pass another --out.")
    if args.n_perm < 20:
        sys.exit(f"--n-perm {args.n_perm} is too few for a "
                 f"{100*(1-ALPHA):.0f}th percentile; the threshold would "
                 f"rest on one or two order statistics.")

    # A null is only a null for the probe that produced the curves, and
    # an optional check is one that can be omitted on the run that
    # matters. The metadata is therefore required, and the analysis
    # scope -- which layers, which feature sets -- is taken from it
    # rather than restated, so the two cannot drift apart.
    if not args.curves_meta and not args.pilot:
        sys.exit(
            "--curves-meta is required: a null calibrates a specific "
            "set of curves, and without\ntheir metadata there is "
            "nothing to check the probe specification against.\n"
            "Pass --pilot for an exploratory run whose output must not "
            "be used as a threshold.")
    if args.pilot and not args.features:
        sys.exit("--pilot needs --features, since there is no "
                 "--curves-meta to take them from")

    # "both" is the null the reported claim is made against: the fully
    # null pipeline, with no real label information at any stage.
    # "train" and "test" answer narrower diagnostic questions, and a
    # production run using one would produce an entirely normal-looking
    # threshold for a different question.
    if not args.pilot and args.permute != "both":
        sys.exit(
            f"the primary null requires --permute both, not "
            f"'{args.permute}'.\n'train' leaves test labels real and "
            f"asks what an untrained probe scores against true "
            f"pockets;\n'test' leaves training real and asks what a "
            f"trained probe scores against scrambled truth. Both are "
            f"diagnostics\nand are available with --pilot.")

    if args.curves_meta:
        cm = json.loads(Path(args.curves_meta).read_text())
        mismatch = {}
        if abs(float(cm.get("C", -1)) - args.c) > 1e-12:
            mismatch["C"] = (cm.get("C"), args.c)
        if int(cm.get("max_iter", -1)) != args.max_iter:
            mismatch["max_iter"] = (cm.get("max_iter"), args.max_iter)
        if cm.get("class_weight") != CLASS_WEIGHT:
            mismatch["class_weight"] = (cm.get("class_weight"),
                                        CLASS_WEIGHT)
        if mismatch:
            sys.exit("the null would use a different probe from the "
                     "curves it calibrates:\n" + "\n".join(
                         f"  {k}: curves={a} null={b}"
                         for k, (a, b) in mismatch.items()))
        print(f"probe spec matches {Path(args.curves_meta).name}")

        # Scope follows the real run. Taken when not given, checked for
        # equality when given: a null over a subset of the curves'
        # layers or feature sets would leave the rest unthresholded.
        # Malformed or stale metadata must not silently become "no
        # constraint": without these, a file missing "layers" would
        # leave the null free to calibrate a different scope than the
        # curves it is for.
        if not cm.get("feature_sets"):
            sys.exit(f"{Path(args.curves_meta).name} has no "
                     f"feature_sets; the curves' scope is unknown")
        if not cm.get("layers"):
            sys.exit(f"{Path(args.curves_meta).name} has no layers; "
                     f"the curves' scope is unknown")
        meta_features = list(cm["feature_sets"])
        meta_layers = [int(x) for x in cm["layers"]]

        # A requested scope must be a SUBSET of the curves', not equal
        # to it. Equality would be right for a single run, but the null
        # is run as a job array: one task per layer, because 100
        # permutations at 48 layers is some 700 hours serially. Each
        # task calibrates one slice, and requiring equality refuses
        # every one of them.
        #
        # What equality was protecting -- that the merged null covers
        # the curves it thresholds -- is enforced where it can be:
        # merge_null.py requires every layer 0..n-1 before it will
        # write, and the onset audit requires the complete trunk again.
        # A slice that is never merged simply produces no onset.
        if args.features is None:
            args.features = meta_features
            print(f"feature sets from curves-meta: {args.features}")
        else:
            extra = sorted(set(args.features) - set(meta_features))
            if extra:
                sys.exit(f"feature sets not present in the curves: "
                         f"{extra}.\nThe curves cover "
                         f"{meta_features}.")
            if set(args.features) != set(meta_features):
                print(f"calibrating a SUBSET of the curves' feature "
                      f"sets: {args.features} of {meta_features}")
        if args.layers is None and meta_layers:
            args.layers = meta_layers
            print(f"layers from curves-meta: {len(meta_layers)} layers")
        elif args.layers is not None and meta_layers:
            extra = sorted(set(args.layers) - set(meta_layers))
            if extra:
                sys.exit(f"layers not present in the curves: {extra}")
            if set(args.layers) != set(meta_layers):
                print(f"calibrating a SUBSET of the curves' layers: "
                      f"{len(args.layers)} of {len(meta_layers)}")

    idx = pd.read_parquet(args.index)
    lab = pd.read_parquet(args.labels)
    if lab.system_id.duplicated().any():
        sys.exit("duplicate system_id values in the label table")
    labels = {r.system_id: np.frombuffer(r.labels, dtype=np.int8)
              for r in lab.itertuples()}
    # The null must be computed on exactly the systems and alignments
    # the real curves used; a mismatch here would compare a curve to a
    # threshold derived from different data.
    missing = [s_ for s_ in idx.system_id if s_ not in labels]
    if missing:
        sys.exit(f"{len(missing)} indexed systems have no label vector, "
                 f"e.g. {missing[:3]}")
    for r in idx.itertuples():
        if len(labels[r.system_id]) != int(r.length):
            sys.exit(f"{r.system_id}: label vector "
                     f"{len(labels[r.system_id])} != index length "
                     f"{int(r.length)}")
    n_layers = int(idx.n_layers.iloc[0])
    layers = args.layers if args.layers else list(range(n_layers))
    bad = [l for l in layers if not (0 <= l < n_layers)]
    if bad:
        sys.exit(f"--layers out of range for {n_layers} layers: {bad}")

    train = idx[idx.split == "train"].reset_index(drop=True)
    test = idx[idx.split == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        sys.exit("split is unusable")

    total = len(args.features) * len(layers) * args.n_perm
    print(f"systems : {len(train):,} train, {len(test):,} test")
    print(f"layers  : {len(layers)}  features: {args.features}")
    print(f"null    : {args.n_perm} permutations, within protein, "
          f"permuting {args.permute}")
    print(f"tau     : the {100*(1-ALPHA):.0f}th percentile of the "
          f"permuted AUPRCs")
    print(f"fits    : {total:,} (about "
          f"{total*15/3600:.1f} h at 15 s each)\n")

    # Replicate b's shuffle is regenerated from a per-replicate seed, so
    # it is identical at every layer and for every feature set without
    # having to hold 100 x n_proteins index arrays in memory.
    def perm_labels(ys, b):
        r = np.random.default_rng(args.seed * 1_000_003 + b)
        return [r.permutation(y) for y in ys]

    rows, reps = [], []
    t0 = time.time()
    done = 0

    for which in args.features:
        for layer in layers:
            Xtr = np.concatenate(
                [load_layer(Path(r.path), layer, which)
                 for r in train.itertuples()]).astype(np.float32)
            ytr_true = [labels[r.system_id] for r in train.itertuples()]
            Xte = {r.system_id: load_layer(Path(r.path), layer, which)
                   for r in test.itertuples()}

            # Standardisation depends only on the features, not on the
            # labels, so it is computed once rather than per replicate.
            mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
            sd[sd < 1e-6] = 1.0
            Xtr_scaled = (Xtr - mu) / sd
            del Xtr
            Xte_scaled = {k: (v - mu) / sd for k, v in Xte.items()}
            del Xte

            null_by_q = {q: [] for q in QUADRANTS}
            for b in range(args.n_perm):
                if args.permute in ("both", "train"):
                    ytr = np.concatenate(perm_labels(ytr_true, b))
                else:
                    ytr = np.concatenate(ytr_true)

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    clf = LogisticRegression(
                        C=args.c, max_iter=args.max_iter,
                        class_weight=CLASS_WEIGHT,
                        random_state=args.seed).fit(Xtr_scaled, ytr)
                if any(issubclass(w.category, ConvergenceWarning)
                       for w in caught):
                    raise RuntimeError(
                        f"null probe did not converge at {which} layer "
                        f"{layer}, replicate {b}. A null fitted "
                        f"differently from the real probe is not a null "
                        f"for it.")

                te_rng = np.random.default_rng(
                    args.seed * 7_919 + b)
                per_q = {q: [] for q in QUADRANTS}
                for r in test.itertuples():
                    y = labels[r.system_id]
                    if args.permute in ("both", "test"):
                        y = te_rng.permutation(y)
                    if y.sum() == 0 or y.sum() == len(y):
                        continue
                    p = clf.decision_function(Xte_scaled[r.system_id])
                    per_q[r.quadrant].append(
                        float(average_precision_score(y, p)))
                for q in QUADRANTS:
                    if per_q[q]:
                        m = float(np.median(per_q[q]))
                        null_by_q[q].append(m)
                        # Replicate identity is kept: these rows are the
                        # null TRAJECTORIES the onset rule is audited
                        # against, and discarding them would leave only
                        # marginal thresholds.
                        reps.append({"feature_set": which,
                                     "layer": int(layer), "quadrant": q,
                                     "perm_id": b, "auprc_median": m})
                done += 1

            for q in QUADRANTS:
                vals = np.sort(np.asarray(null_by_q[q]))
                if vals.size == 0:
                    continue
                # Order statistic, not an interpolated quantile: for a
                # finite permutation test the k-th smallest of B values
                # with k = ceil((1-alpha)(B+1)) is exact and
                # conservative, and never invents a value the null did
                # not produce.
                k = int(np.ceil((1 - ALPHA) * (vals.size + 1)))
                k = min(k, vals.size)
                rows.append({
                    "feature_set": which, "layer": int(layer),
                    "quadrant": q, "n_perm": int(vals.size),
                    "tau_order_k": k,
                    "tau": float(vals[k - 1]),
                    "null_median": float(np.median(vals)),
                    "null_max": float(vals[-1]),
                })
            del Xtr_scaled, Xte_scaled, ytr_true

            el = time.time() - t0
            print(f"  {which} layer {layer:>2} | "
                  f"tau {rows[-1]['tau']:.3f} (k={rows[-1]['tau_order_k']}"
                  f"/{rows[-1]['n_perm']}) | {el/60:.1f} min, "
                  f"{done}/{total} fits")

    res = pd.DataFrame(rows)
    res.to_parquet(out)
    rep_path = Path(str(out).replace(".parquet", "") + "_replicates.parquet")
    pd.DataFrame(reps).to_parquet(rep_path)
    Path(str(out) + ".meta.json").write_text(json.dumps({
        "n_perm": args.n_perm, "permute": args.permute,
        "alpha": ALPHA, "within_protein": True,
        "permutations_synchronised_across_layers": True,
        "threshold": "order statistic k = ceil((1-alpha)(B+1))",
        "C": args.c, "max_iter": args.max_iter, "seed": args.seed,
        "layers": [int(x) for x in layers],
        "features": list(args.features),
        "statistic": "median AUPRC across test proteins",
        "hours": round((time.time() - t0) / 3600, 2),
    }, indent=2))

    print(f"\n{'feat':7}{'cell':6}{'tau min':>9}{'tau max':>9}"
          f"{'null med':>10}")
    for which in args.features:
        for q in QUADRANTS:
            sub = res[(res.feature_set == which) & (res.quadrant == q)]
            if sub.empty:
                continue
            print(f"{which:7}{q:6}{sub.tau.min():>9.3f}"
                  f"{sub.tau.max():>9.3f}{sub.null_median.median():>10.3f}")
    print(f"\nWrote {out}")
    print(f"Wrote {rep_path}")
    print("\n  A curve point counts as above chance only where it exceeds\n"
          "  tau at that layer, and the onset rule then requires five\n"
          "  consecutive such layers. Because replicate b uses the same\n"
          "  shuffle at every layer, the replicate file holds null\n"
          "  TRAJECTORIES: how often five consecutive crossings arise\n"
          "  from correlated noise can be measured directly from it,\n"
          "  rather than assumed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
