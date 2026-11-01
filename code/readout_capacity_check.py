#!/usr/bin/env python3
"""
Does the FN deficit narrow when the readout is given nonlinear
capacity?

THE QUESTION, AND WHAT THIS CAN AND CANNOT ANSWER
-------------------------------------------------
Every result so far comes from one readout: a linear probe at fixed
regularisation. A low score under it is consistent with two different
worlds. The information may be absent from the representation, or it
may be present in a form a linear map cannot reach. Those call for
different interventions and the linear probe cannot tell them apart.

The full test is a capacity ladder on the pair tensor -- axial mixing
and Pairformer-style blocks -- and it is not available here. The
extraction kept z only as its row mean, column mean and self-pair
diagonal, so the L x L structure those heads operate on is gone. What
remains possible is the first rung above linear: a multilayer
perceptron on the same features.

So this answers "is the FN deficit specific to LINEARITY" and not "is
it a readout bottleneck". A deficit that narrows here is evidence for
the second; a deficit that does not narrow leaves it open, because the
heads that would mix across residue pairs were never tried.

PRE-SPECIFIED BEFORE RUNNING
----------------------------
The quantity of interest is the FN-to-NF gap, fixed because those two
cells are matched on protein length, pocket size, positive rate and
alignment depth. The comparison is that gap under the linear probe
against the same gap under the MLP, on identical features, split and
layer.

NARROWING is a reduction in the gap. The reduction is read with a
cluster bootstrap interval, as elsewhere, and reported whatever its
sign. A gap that widens is as informative as one that shrinks and is
not to be explained away.

The architecture is fixed here rather than tuned: one hidden layer,
width equal to the input, ReLU, early stopping on a held-out slice of
TRAINING clusters. Tuning capacity against test performance would
make a better number and answer nothing, since the question is
whether capacity helps rather than how much can be extracted.

USAGE
-----
    python3 readout_capacity_check.py \\
        --index probe_index.parquet \\
        --labels pocket_labels.parquet \\
        --curves depth_curves.parquet \\
        --curves-meta depth_curves.parquet.meta.json \\
        --out readout_capacity.json
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
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.exceptions import ConvergenceWarning
except ImportError as e:
    sys.exit(f"needs scikit-learn: {e}")

try:
    from load_representations import load_layer
except ImportError as e:  # noqa: BLE001
    sys.exit(f"needs load_representations.py alongside this script: {e}")

CLASS_WEIGHT = "balanced"
# Fixed here, not passed in: the early-stopping rule is part of the
# pre-specified design and tuning it would make the comparison a
# search.
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 5
MLP_VAL_CLUSTER_FRAC = 0.15
# The per-epoch validation curve and the epoch the stopping rule
# actually chose. Both are kept, because they can differ: the rule
# requires an improvement over the running best of more than the
# tolerance, so a curve creeping upward by less than that keeps its
# first epoch while the curve's argmax moves on. Reporting the argmax
# as the selected epoch would describe a model that was never fitted.
MLP_TRACE = []
MLP_SELECTED = []       # one entry, the epoch the rule chose
PAIR = ("FN", "NF")
QUADRANTS = ("FF", "FN", "NF", "NN")


def fit_score(kind, Xtr, ytr, wtr, val_blocks, Xfull, yfull, wfull,
              test, labels, layer, feature, mu, sd, mu_sel, sd_sel,
              C, max_iter, seed):
    """Fit one readout and score every test protein with it.

    Both readouts see balanced classes. The linear probe gets this
    through class_weight; scikit-learn's MLP accepts neither
    class_weight nor sample_weight, so it is given a training set
    balanced by resampling instead -- positives drawn with
    replacement until the two classes are equal in number.

    These are not identical. Weighting rescales each residue's
    contribution to the loss; resampling duplicates residues. The two
    target the same class-balanced objective in expectation, but
    induce different finite-sample optimisation dynamics: duplicated
    rows, minibatching and an adaptive optimiser do not reproduce the
    weighted updates, and the positives carry a smaller effective
    sample than their count suggests. The alternative was to drop weighting from
    both, and that was rejected because it would move the linear
    baseline away from the probe every other result in this study
    uses, leaving the comparison disconnected from the rest.

    Comparing a balanced linear loss against an unweighted nonlinear
    one would have varied two things at once, which is the failure
    this avoids.
    """
    if kind == "linear":
        clf = LogisticRegression(C=C, max_iter=max_iter,
                                 class_weight=CLASS_WEIGHT,
                                 random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit((Xtr - mu) / sd, ytr)
        if any(issubclass(w.category, ConvergenceWarning)
               for w in caught):
            raise RuntimeError(
                f"the linear probe did not converge at {feature} "
                f"layer {layer}. A readout that hit its optimisation "
                f"limit cannot be the baseline for a capacity "
                f"comparison.")
        return score_test(clf, "linear", test, labels, layer, feature,
                          mu, sd)

    # Two stages, and the second is what makes the comparison fair.
    #
    # STAGE 1 selects the number of epochs, using a slice of training
    # clusters held out by family. scikit-learn's own
    # validation_fraction splits the residues it is handed, and those
    # residues come from proteins sharing families, so its validation
    # set would not be independent of its training set. partial_fit is
    # the only interface that accepts a validation set of our own.
    #
    # STAGE 2 throws that model away and refits from scratch on ALL
    # training clusters for exactly that many epochs. Without it the
    # MLP would be fitted on 85% of the training families while the
    # linear probe saw 100%, and a difference in the gap could be the
    # smaller sample rather than the capacity -- the same confound as
    # mismatched class weights, arriving by another route.
    # Selection-stage standardisation comes from the FITTING slice
    # alone. The held-out clusters are meant to be untouched while the
    # epoch count is chosen, and feature means computed partly from
    # them would let them influence the preprocessing that selects it.
    # Unlabeled, and far weaker than test leakage, but the slice is
    # either held out or it is not.
    # Balanced by resampling, since the estimator takes no weights.
    # Drawn once with a fixed seed rather than per epoch, so every
    # epoch sees the same training set and the epoch count means what
    # it usually does.
    Ztr, ytr_b = balance(Xtr, ytr, mu_sel, sd_sel, seed)
    sel = MLPClassifier(hidden_layer_sizes=(Xtr.shape[1],),
                        activation="relu", max_iter=1,
                        warm_start=True, random_state=seed)
    # The whole per-epoch trace is kept and reported. An early stop
    # at epoch 1 and a validation curve that was still climbing within
    # the improvement tolerance look identical from the best value
    # alone, and they mean different things: the first is a model that
    # converged, the second one that was halted. The trace lets a
    # reader tell them apart.
    best, best_epoch, stall = -np.inf, None, 0
    trace = []
    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            sel.partial_fit(Ztr, ytr_b, classes=np.array([0, 1]))
        # The criterion is the MEDIAN PER-PROTEIN AUPRC on the
        # held-out proteins, matching the study's evaluation unit.
        # Pooling their residues and taking one AP would weight a
        # 700-residue protein over four times a 150-residue one, so
        # the stopping rule would be length-weighted while the
        # reported endpoint is protein-weighted. This study has gone
        # to some trouble to remove length from the FN-NF comparison;
        # reintroducing it through epoch selection would be careless.
        ap = median_val_auprc(sel, val_blocks, mu_sel, sd_sel)
        # Stored unrounded. The tolerance is 1e-5 and rounding to
        # five decimals would erase differences at exactly the scale
        # this trace exists to expose.
        trace.append(float(ap))
        if ap > best + 1e-5:
            best, best_epoch, stall = ap, epoch, 0
        else:
            stall += 1
            if stall >= MLP_PATIENCE:
                break
    if best_epoch is None:
        raise RuntimeError("the MLP never improved on the held-out "
                           "clusters; no epoch count was selected")
    stopped_early = stall >= MLP_PATIENCE
    print(f"  mlp: epoch {best_epoch} selected on held-out clusters "
          f"(median per-protein AUPRC {best:.4f}), "
          f"{'early stopping' if stopped_early else 'EPOCH LIMIT'}")
    print(f"  mlp: validation trace "
          f"{[round(v, 6) for v in trace]}")
    if best_epoch == 1 and len(trace) > 1:
        rise = max(trace[1:]) - trace[0]
        print(f"  mlp: selected the first epoch. The best later value "
              f"is {rise:+.5f} from it;\n       a value just under "
              f"the 1e-5 tolerance would mean the model was still\n"
              f"       improving when it was stopped.")
    MLP_TRACE.extend(trace)
    MLP_SELECTED.append(int(best_epoch))
    if not stopped_early:
        raise RuntimeError(
            f"the MLP reached the epoch limit of {MLP_MAX_EPOCHS} "
            f"without the early-stopping rule firing. It was still "
            f"improving, so its capacity was not exhausted and the "
            f"comparison would understate it.")

    # Stage 2: a fresh model, same seed and architecture, all training
    # clusters, exactly best_epoch epochs -- and the FULL-training
    # standardisation, the same one the linear probe and the test
    # scoring use.
    clf = MLPClassifier(hidden_layer_sizes=(Xfull.shape[1],),
                        activation="relu", max_iter=1,
                        warm_start=True, random_state=seed)
    Zfull, yfull_b = balance(Xfull, yfull, mu, sd, seed)
    for _ in range(best_epoch):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.partial_fit(Zfull, yfull_b, classes=np.array([0, 1]))
    print(f"  mlp: refitted on {len(yfull_b):,} balanced residues "
          f"drawn from {len(yfull):,} for {best_epoch} epochs")
    return score_test(clf, "mlp", test, labels, layer, feature, mu, sd)


def balance(X, y, mu, sd, seed):
    """Standardise, then balance the classes by resampling.

    The minority class is drawn with replacement until it matches the
    majority. This substitutes for the class weighting the linear
    probe receives, because scikit-learn's MLP accepts neither
    class_weight nor sample_weight.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        raise RuntimeError("one class is absent; cannot balance")
    minority, majority = ((pos, neg) if len(pos) < len(neg)
                          else (neg, pos))
    drawn = rng.choice(minority, size=len(majority), replace=True)
    take = np.concatenate([majority, drawn])
    rng.shuffle(take)
    return ((X[take] - mu) / sd).astype(np.float32), y[take]


def median_val_auprc(clf, blocks, mu, sd):
    """Median per-protein AUPRC over the held-out validation proteins.

    Protein boundaries are kept rather than concatenated, because the
    quantity this study reports is a median over proteins and the
    stopping rule should optimise the same thing.
    """
    vals = []
    for X, y in blocks:
        p = clf.predict_proba((X - mu) / sd)[:, 1]
        vals.append(average_precision_score(y, p))
    return float(np.median(vals)) if vals else float("nan")


def score_test(clf, kind, test, labels, layer, feature, mu, sd):

    """Per-protein AUPRC on the held-out proteins."""
    rows = []
    for r in test.itertuples():
        y = labels[r.system_id]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        X = load_layer(Path(r.path), layer, feature)
        p = (clf.decision_function((X - mu) / sd) if kind == "linear"
             else clf.predict_proba((X - mu) / sd)[:, 1])
        rows.append({"system_id": r.system_id,
                     "quadrant": r.quadrant,
                     "family_cluster": r.family_cluster,
                     "auprc": float(average_precision_score(y, p))})
    return pd.DataFrame(rows)


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--curves", required=True)
    ap.add_argument("--curves-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature", default="s",
                    help="the primary feature set; the full pair "
                         "tensor is not on disk, so the richer heads "
                         "cannot be run")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.n_boot <= 0:
        sys.exit("--n-boot must be positive")
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    cm = json.loads(Path(args.curves_meta).read_text())
    for k in ("C", "max_iter", "class_weight"):
        if k not in cm:
            sys.exit(f"{Path(args.curves_meta).name} has no '{k}'")
    if cm["class_weight"] != CLASS_WEIGHT:
        sys.exit(f"class weighting differs from the curves")
    C, max_iter = float(cm["C"]), int(cm["max_iter"])

    idx = pd.read_parquet(args.index)
    if "family_cluster" not in idx.columns:
        sys.exit("--index has no family_cluster")
    # A null cluster cannot be held out by family and cannot be
    # resampled as one, so it is fatal rather than silently dropped.
    if idx.family_cluster.isna().any():
        bad = idx.loc[idx.family_cluster.isna(),
                      "system_id"].head(5).tolist()
        sys.exit(f"{int(idx.family_cluster.isna().sum())} systems have "
                 f"no family_cluster, e.g. {bad}.\nBoth the "
                 f"early-stopping split and the bootstrap are "
                 f"undefined for them.")
    lab = pd.read_parquet(args.labels)
    labels = {r.system_id: np.frombuffer(r.labels, dtype=np.int8)
              for r in lab.itertuples()}
    curves = pd.read_parquet(args.curves)
    sub = curves[curves.feature_set == args.feature]
    if sub.empty:
        sys.exit(f"no curves for feature set '{args.feature}'")
    layer = int(sub.groupby("layer").auprc_median.median().idxmax())

    train_all = idx[idx.split == "train"].reset_index(drop=True)
    test = idx[idx.split == "test"].reset_index(drop=True)

    # The MLP's early-stopping set is held out BY CLUSTER, so no
    # family appears in both. Carved from the training split only; the
    # test split is untouched and the linear probe still trains on
    # every training protein, as it does elsewhere in the study.
    rng0 = np.random.default_rng(args.seed)
    tr_clusters = train_all.family_cluster.dropna().unique()
    n_val = max(1, int(round(MLP_VAL_CLUSTER_FRAC * len(tr_clusters))))
    val_clusters = set(rng0.choice(tr_clusters, n_val, replace=False))
    is_val = train_all.family_cluster.isin(val_clusters)
    train = train_all[~is_val].reset_index(drop=True)
    val = train_all[is_val].reset_index(drop=True)
    print(f"early-stopping split: {len(val):,} proteins over "
          f"{n_val:,} clusters held out of training")
    print(f"feature {args.feature}, peak layer {layer}")
    print(f"systems {len(train):,} train, {len(test):,} test")

    def stack(df):
        X = np.concatenate(
            [load_layer(Path(r.path), layer, args.feature)
             for r in df.itertuples()]).astype(np.float32)
        y = np.concatenate(
            [labels[r.system_id] for r in df.itertuples()]
        ).astype(np.int8)
        return X, y

    Xall, yall = stack(train_all)          # the linear probe's data
    Xtr, ytr = stack(train)                # the MLP's fitting data
    # The early-stopping set keeps its protein boundaries, so the
    # criterion can be a median over proteins rather than one pooled
    # score.
    val_blocks = []
    for r in val.itertuples():
        y = labels[r.system_id]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        val_blocks.append(
            (load_layer(Path(r.path), layer, args.feature)
             .astype(np.float32), y))
    if not val_blocks:
        sys.exit("no usable proteins in the early-stopping slice")
    # Two standardisations, used at different stages. mu_sel/sd_sel
    # come from the fitting slice only and are used while the epoch
    # count is being selected; mu/sd come from all training systems
    # and are used for the final MLP refit, the linear probe, and all
    # test scoring, so the two readouts are preprocessed identically
    # where it counts.
    mu, sd = Xall.mean(axis=0), Xall.std(axis=0)
    sd[sd < 1e-6] = 1.0
    mu_sel, sd_sel = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd_sel[sd_sel < 1e-6] = 1.0

    # The linear probe's balanced objective is preserved for the MLP
    # by deterministic class resampling, because this scikit-learn
    # version accepts neither class_weight nor sample_weight for
    # MLPClassifier. The weights are computed only to report the
    # imbalance; nothing consumes them.
    n0, n1 = int((ytr == 0).sum()), int((ytr == 1).sum())
    if not (n0 and n1):
        sys.exit("one class is absent from the MLP training slice")
    wtr = wall = None          # the MLP is balanced by resampling
    print(f"residues: {len(yall):,} for the linear probe, "
          f"{len(ytr):,} fitting the MLP; {len(val_blocks):,} "
          f"proteins select its epoch")
    print(f"positive rate {yall.mean():.3f}; the linear probe is "
          f"weighted, the MLP\nbalanced by resampling to "
          f"{2 * int((yall == 0).sum()):,} residues")

    scored, t0 = {}, time.time()
    for kind in ("linear", "mlp"):
        A, b = (Xall, yall) if kind == "linear" else (Xtr, ytr)
        scored[kind] = fit_score(kind, A, b, wtr, val_blocks,
                                 Xall, yall, wall, test, labels,
                                 layer, args.feature, mu, sd,
                                 mu_sel, sd_sel,
                                 C, max_iter, args.seed)
        g = scored[kind].groupby("quadrant").auprc.median()
        print(f"\n{kind:7} " + "  ".join(
            f"{q} {g.get(q, float('nan')):.3f}" for q in QUADRANTS)
            + f"   | {(time.time() - t0) / 60:.1f} min")

    # The pre-specified quantity: the FN-to-NF gap under each readout,
    # and whether it narrows. Bootstrapped over family clusters, with
    # both readouts evaluated inside the same resample so the
    # difference of gaps is a difference and not two independent
    # estimates.
    rng = np.random.default_rng(args.seed)
    merged = scored["linear"][["system_id", "quadrant",
                               "family_cluster", "auprc"]].rename(
        columns={"auprc": "lin"}).merge(
        scored["mlp"][["system_id", "auprc"]].rename(
            columns={"auprc": "mlp"}), on="system_id",
        validate="one_to_one")
    clusters = merged.family_cluster.unique()
    by_cl = {c: d for c, d in merged.groupby("family_cluster")}

    def gap(d, col):
        a = d.loc[d.quadrant == PAIR[1], col].median()
        b = d.loc[d.quadrant == PAIR[0], col].median()
        return float(a - b)

    g_lin, g_mlp = gap(merged, "lin"), gap(merged, "mlp")
    draws = np.empty(args.n_boot)
    for b in range(args.n_boot):
        rep = pd.concat([by_cl[c] for c in
                         rng.choice(clusters, len(clusters),
                                    replace=True)], ignore_index=True)
        draws[b] = gap(rep, "mlp") - gap(rep, "lin")
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    change = g_mlp - g_lin

    print(f"\n{PAIR[1]} minus {PAIR[0]} gap")
    print(f"  linear {g_lin:+.3f}")
    print(f"  mlp    {g_mlp:+.3f}")
    print(f"  change {change:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    excl = bool(lo > 0 or hi < 0)
    if excl and change < 0:
        verdict = ("the gap narrows under nonlinear capacity, which "
                   "is evidence that part of\n  the FN deficit is a "
                   "property of the linear readout")
    elif excl:
        verdict = ("the gap WIDENS under nonlinear capacity, which "
                   "the readout-bottleneck\n  account does not "
                   "predict")
    else:
        verdict = ("the interval contains zero: nonlinear capacity "
                   "does not measurably change\n  the gap. The "
                   "pair-mixing heads were not tried, so a readout "
                   "bottleneck\n  is narrowed, not excluded")
    print(f"\n  {verdict}.")

    res = {"feature": args.feature, "layer": layer,
           "pair": list(PAIR), "n_boot": args.n_boot,
           "seed": args.seed, "C": C,
           "readouts_compared": ["linear", "mlp"],
           "heads_not_available": ["axial", "pairformerK"],
           "reason": "the extraction retained z only as row mean, "
                     "column mean and self-pair diagonal, so the "
                     "L x L structure these heads operate on is not "
                     "on disk",
           "cells": {k: {q: round(float(
               v.loc[v.quadrant == q, "auprc"].median()), 4)
               for q in QUADRANTS} for k, v in scored.items()},
           "mlp_validation_trace": MLP_TRACE,
           # The epoch the rule chose, and separately the epoch the
           # curve peaked at. They agree only when the peak beat the
           # running best by more than the tolerance.
           "mlp_epochs_selected": (MLP_SELECTED[0] if MLP_SELECTED
                                   else None),
           "mlp_trace_argmax_epoch": (int(np.argmax(MLP_TRACE) + 1)
                                      if MLP_TRACE else None),
           "mlp_trace_argmax_auprc": (float(max(MLP_TRACE))
                                      if MLP_TRACE else None),
           "mlp_later_gain_if_epoch1": (
               float(max(MLP_TRACE[1:]) - MLP_TRACE[0])
               if MLP_SELECTED and MLP_SELECTED[0] == 1
               and len(MLP_TRACE) > 1 else None),
           "gap_linear": round(g_lin, 4),
           "gap_mlp": round(g_mlp, 4),
           "change": round(change, 4),
           "lo": round(float(lo), 4), "hi": round(float(hi), 4),
           "excludes_zero": excl}
    out.write_text(json.dumps(res, indent=2))
    scored["mlp"].to_parquet(str(out).replace(".json", "_mlp.parquet"))
    print(f"\nWrote {out}")
    print("\n  This compares linear against one nonlinear head on the "
          "same features.\n  It is not the capacity ladder: the heads "
          "that mix across residue pairs\n  need the full pair "
          "tensor, which this extraction did not keep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
