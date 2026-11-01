#!/usr/bin/env python3
"""
Save per-protein AUPRC at a small set of layers.

WHEN TO USE THIS, AND WHEN NOT TO
---------------------------------
run_depth_probes.py now writes a per-system sidecar beside the curves,
holding the individual protein scores from the SAME fitted classifiers
that produced the medians. Where that sidecar exists, use it: scores
taken from the original fit are stronger evidence than scores from a
refit with nominally identical settings.

This script is for the case where it does not exist -- a curve file
produced before that output was added, or recovered from a run whose
sidecar was lost. It refits at a small number of layers and reports
per-protein AUPRC.

The design-factor check does need it: it asks whether per-protein AUPRC
is explained by protein length and pocket size rather than by anything
the representation encodes, and that is a regression over proteins.

Rather than re-running 144 layers to recover values the curves threw
away, this refits at ONE layer per feature set -- by default each
feature set's own peak, which is where the reported contrast lives.
Three fits instead of 144.

The probe is identical to the curve run: same C, same max_iter, same
class weighting, same cluster-disjoint split, same standardisation
fitted on training residues only. The configuration is checked against
the curve run's metadata, so a score produced under a different probe
cannot be mixed with the curves it is meant to explain.

USAGE
-----
    python3 per_system_scores.py \\
        --index probe_index.parquet \\
        --labels pocket_labels.parquet \\
        --curves depth_curves.parquet \\
        --curves-meta depth_curves.parquet.meta.json \\
        --out per_system_auprc.parquet
"""

from __future__ import annotations

import argparse
import hashlib
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

try:
    from load_representations import load_layer
except ImportError as e:  # noqa: BLE001
    sys.exit(f"needs load_representations.py alongside this script: {e}")

CLASS_WEIGHT = "balanced"


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--curves", required=True,
                    help="finished depth curves, used to locate each "
                         "feature set's peak layer")
    ap.add_argument("--curves-meta", required=True,
                    help="the curve run's .meta.json; C, max_iter and "
                         "class weighting are taken from it so the "
                         "probe here is the same probe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs="*", type=int,
                    help="layers to score. The default picks each "
                         "feature set's peak as the layer maximising "
                         "the median across quadrants of the "
                         "per-quadrant medians -- a defensible summary "
                         "but one choice among several. Pass the layer "
                         "the primary analysis actually reports, if it "
                         "defines the peak differently.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    # The probe must match the one that produced the curves, or the
    # per-protein scores would explain a different analysis.
    cm = json.loads(Path(args.curves_meta).read_text())
    for k in ("C", "max_iter", "class_weight"):
        if k not in cm:
            sys.exit(f"{Path(args.curves_meta).name} has no '{k}'")
    if cm["class_weight"] != CLASS_WEIGHT:
        sys.exit(f"class weighting differs: curves={cm['class_weight']} "
                 f"here={CLASS_WEIGHT}")
    if "seed" in cm and int(cm["seed"]) != int(args.seed):
        sys.exit(f"seed differs: curves={cm['seed']} here={args.seed}")
    C, max_iter = float(cm["C"]), int(cm["max_iter"])
    print(f"probe from {Path(args.curves_meta).name}: C={C}, "
          f"max_iter={max_iter}, class_weight={CLASS_WEIGHT}")

    # Matching C and max_iter does not establish that these are the
    # same inputs. A curve produced from one index and labels, scored
    # here against another, would agree on every probe setting and
    # explain the wrong analysis. The curve run records input hashes in
    # its .state.json; when that file is present they are enforced, and
    # when it is not, the gap is stated rather than passed over.
    def _sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    state_file = Path(str(args.curves) + ".state.json")
    if state_file.exists():
        cfg = json.loads(state_file.read_text()).get("config", {})
        checks = (("index_sha256", args.index, "--index"),
                  ("labels_sha256", args.labels, "--labels"))
        absent = [k for k, _, _ in checks if k not in cfg]
        if absent:
            # Skipping an absent hash and then reporting a match would
            # claim a verification that never happened. Three outcomes
            # are possible here and each is said plainly: verified, not
            # verifiable because the hashes are absent, or not
            # verifiable because the state file is.
            print(f"WARNING: {state_file.name} exists but lacks "
                  f"{', '.join(absent)}, so the input\n  files cannot "
                  f"be verified against the curve run.")
        else:
            for key, path, what in checks:
                if cfg[key] != _sha(path):
                    sys.exit(f"{what} differs from the file used to "
                             f"produce the curves.\nRefusing to "
                             f"explain one analysis with scores from "
                             f"another.")
            print(f"input hashes match {state_file.name}")
    else:
        print(f"WARNING: {state_file.name} is absent, so --index and "
              f"--labels cannot be\n  checked against the files that "
              f"produced the curves. Verify by hand that they\n  are "
              f"the same, or regenerate the curves with a version that "
              f"writes the state file.")

    idx = pd.read_parquet(args.index)
    lab = pd.read_parquet(args.labels)
    labels = {r.system_id: np.frombuffer(r.labels, dtype=np.int8)
              for r in lab.itertuples()}
    curves = pd.read_parquet(args.curves)

    train = idx[idx.split == "train"].reset_index(drop=True)
    test = idx[idx.split == "test"].reset_index(drop=True)
    print(f"systems: {len(train):,} train, {len(test):,} test\n")

    # One layer per feature set: its own peak, since that is where the
    # contrast being explained was reported.
    targets = []
    for feature in sorted(curves.feature_set.unique()):
        sub = curves[curves.feature_set == feature]
        if args.layers:
            for l in args.layers:
                targets.append((feature, int(l)))
        else:
            peak = int(sub.groupby("layer").auprc_median.median().idxmax())
            targets.append((feature, peak))
            print(f"  {feature:6} peak layer {peak}")

    rows = []
    t0 = time.time()
    for feature, layer in targets:
        Xtr = np.concatenate(
            [load_layer(Path(r.path), layer, feature)
             for r in train.itertuples()]).astype(np.float32)
        ytr = np.concatenate(
            [labels[r.system_id] for r in train.itertuples()]
        ).astype(np.int8)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd < 1e-6] = 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = LogisticRegression(
                C=C, max_iter=max_iter, class_weight=CLASS_WEIGHT,
                random_state=args.seed).fit((Xtr - mu) / sd, ytr)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError(
                f"probe did not converge at {feature} layer {layer}, "
                f"though the curve run reported it converging. The two "
                f"are no longer the same fit.")
        del Xtr, ytr

        for r in test.itertuples():
            y = labels[r.system_id]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            X = load_layer(Path(r.path), layer, feature)
            p = clf.decision_function((X - mu) / sd)
            rows.append({"system_id": r.system_id,
                         "feature_set": feature, "layer": layer,
                         "quadrant": r.quadrant,
                         "auprc": float(average_precision_score(y, p))})
        print(f"  {feature:6} layer {layer:>2}: {len(test):,} proteins "
              f"scored | {(time.time() - t0) / 60:.1f} min")

    res = pd.DataFrame(rows)
    res.to_parquet(out)
    Path(str(out) + ".meta.json").write_text(json.dumps({
        "C": C, "max_iter": max_iter, "class_weight": CLASS_WEIGHT,
        "seed": args.seed,
        "layers_scored": [{"feature_set": f, "layer": l}
                          for f, l in targets],
        "curves_meta": str(args.curves_meta),
        "n_rows": int(len(res)),
    }, indent=2))

    print(f"\n{'feature':8}{'layer':>7}{'n':>6}{'median':>9}")
    for feature, layer in targets:
        sub = res[(res.feature_set == feature) & (res.layer == layer)]
        print(f"{feature:8}{layer:>7}{len(sub):>6}"
              f"{sub.auprc.median():>9.3f}")
    print(f"\nWrote {out}")
    print("\n  These are the individual protein scores behind the curve "
          "medians.\n  design_factor_check.py reads them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
