#!/usr/bin/env python3
"""
Fit a pocket probe at every trunk layer and score it per familiarity
regime.

Produces one tidy row per (feature set, layer, quadrant): the held-out
AUPRC of a linear probe trained to separate pocket from non-pocket
residues using that layer's representation. Those rows are the depth
curves; the onset estimator reads them.

This script does not estimate an onset and does not compute a null. Both
need these curves first, and keeping them separate means the curve can be
inspected before any threshold is applied to it.

ONE PROBE, EVALUATED PER REGIME
-------------------------------
A single probe is trained on all training systems and then scored
separately on each quadrant's test systems. The alternative -- a probe
per quadrant -- would confound the question: NN has fewer systems than
FF, so a per-quadrant probe would do worse there partly because it had
less data to learn from, and the result would be unreadable as a
statement about the representation.

A shared probe asks the question the study is about: given a readout
fitted on whatever was available, how well does the representation
support pocket prediction in each regime? That is also the situation a
practitioner faces.

SCORING AT THE PROTEIN LEVEL
----------------------------
AUPRC is computed per protein and then aggregated, rather than pooling
every residue into one ranking. Pooling would let a few long proteins
dominate, and would implicitly compare residues across proteins, which
is not the prediction anyone makes.

The aggregate reported is the MEDIAN across proteins, with the mean also
recorded. Pocket sizes vary by an order of magnitude, and a handful of
proteins with very small pockets can swing a mean a long way; the median
is the pre-registered primary.

TRAINING IS RESIDUE-WEIGHTED
----------------------------
Training pools residues across proteins, so a 700-residue protein
contributes roughly twice as many observations as a 350-residue one, to
the classifier and to the standardisation statistics alike. Evaluation
corrects for this by scoring each protein separately, but the fit itself
does not. That is the pre-registered primary analysis; equalising
protein weights is recorded as a planned sensitivity check rather than
applied here, because changing it after seeing curves would be a choice
made on the result.

STANDARDISATION
---------------
Features are standardised per layer, with the mean and scale fitted on
TRAINING systems only and applied unchanged to test systems. Fitting on
everything would leak test distribution into the transform. Per layer,
because activation scale varies with depth -- the smoke test found
layer 47 magnitudes an order of magnitude above early layers -- and a
single global scaling would make early layers look flat for a reason
that has nothing to do with pocket information.

REGULARISATION
--------------
Fixed across layers AND across feature sets, not tuned per layer.
Tuning per layer would let the probe's capacity vary with depth and
confound "the signal appeared" with "the probe got more freedom".

The selection rule, pre-registered:

    C is selected once from a prespecified grid using TRAINING data
    only, as the largest value for which the probe converges at all 48
    layers and for all three feature sets. No held-out AUPRC and no
    onset statistic is examined during this selection. The selected C is
    then fixed for every layer, feature set and familiarity regime;
    neighbouring convergent values are reported as sensitivity analyses.

This is a numerical ADMISSIBILITY rule, not hyperparameter tuning. It
depends on the training data, which is unremarkable -- so does the
standardisation -- and what matters is that it never consults the
outcome. Inspecting depth curves and then choosing the C with the
nicest onset would be a different thing entirely.

Selection waits for a training pool containing all four regimes. A C
chosen on one quadrant may fail to converge on another, leaving a choice
between re-selecting mid-study and failing, and FF may be numerically
easier than NN.

ON NON-CONVERGENCE
------------------
With finite L2 regularisation the objective is generally well behaved,
so failure to converge does not by itself imply separability or
unbounded weights: poor conditioning, redundant directions, or simply
an insufficient iteration budget will do it. The two causes call for
different fixes and are distinguishable -- --diagnose crosses max_iter
with C at a few layers, and if a large iteration budget alone suffices,
the problem was the budget and the estimator need not be redefined.

ON CLASS WEIGHTING
------------------
`balanced` is the primary setting. At a 7.8% positive rate sklearn's
weights are roughly w+ 6.4 and w- 0.54, a ratio near 12. This changes
what the classifier learns, not merely its calibration, so it is not a
knob to turn in response to a convergence failure; unweighted fitting is
recorded as a sensitivity analysis instead.

Note that AUPRC is NOT insensitive to class prior -- its random-ranking
baseline is exactly the positive prevalence, which is why
`baseline_median` is reported alongside every curve point.

USAGE
-----
    python3 run_depth_probes.py --index probe_index.parquet \\
        --out depth_curves.parquet
    python3 run_depth_probes.py --index probe_index.parquet \\
        --out curves_quick.parquet --layers 0 8 16 24 32 40 47
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

FEATURE_SETS = ("s", "zpool", "both")
QUADRANTS = ("FF", "FN", "NF", "NN")

# Pre-registered probe settings. Fixed before any depth curve was seen.
# C is selected by the pre-registered convergence rule on the full
# training pool, never by held-out performance.
C_INVERSE_REG = 1.0
MAX_ITER = 5000
CLASS_WEIGHT = "balanced"


# The bfloat16 decode lives in load_representations.py and is imported
# rather than copied: a local copy would not receive a correction made
# there, and reading the stored bits as numbers instead of
# reinterpreting them yields values that look entirely like activations.
try:
    from load_representations import load_layer
except ImportError as e:  # noqa: BLE001
    sys.exit(f"needs load_representations.py alongside this script: {e}")


def load_labels(labels_path: Path) -> dict:
    """system_id -> binary per-residue vector."""
    lab = pd.read_parquet(labels_path)
    return {r.system_id: np.frombuffer(r.labels, dtype=np.int8)
            for r in lab.itertuples()}


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True,
                    help="probe_index.parquet from assemble_probe_data.py")
    ap.add_argument("--labels", required=True,
                    help="pocket_labels parquet, for the y vectors")
    ap.add_argument("--out",
                    help="output parquet; not needed with --diagnose or "
                         "--sweep-c, which write nothing")
    ap.add_argument("--features", nargs="*", default=list(FEATURE_SETS),
                    choices=list(FEATURE_SETS))
    ap.add_argument("--layers", nargs="*", type=int,
                    help="subset of layers, for a quick pass")
    ap.add_argument("--c", type=float, default=C_INVERSE_REG,
                    help=f"inverse regularisation strength (default "
                         f"{C_INVERSE_REG}). The same value is used at "
                         f"every layer; see --sweep-c to choose it.")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--sweep-c", nargs="*", type=float,
                    help="try these C values at the given layers, across "
                         "every requested feature set, and report which "
                         "converge everywhere; exits without writing "
                         "curves")
    ap.add_argument("--sweep-state",
                    help="JSON file recording EVERY individual fit as "
                         "it completes, so a sweep killed by a walltime "
                         "loses only the fit then in flight. Without "
                         "it, hours of converged fits are discarded "
                         "because a C is only resolved once all 48 "
                         "layers of all three feature sets have passed.")
    ap.add_argument("--resume-from",
                    help="partial curve parquet from an interrupted "
                         "run. Its (feature set, layer) rows are kept "
                         "and not refitted. Each layer is appended to "
                         "--out as it completes, so --out of a killed "
                         "run is itself a valid --resume-from.")
    ap.add_argument("--diagnose", action="store_true",
                    help="cross max_iter with C at the given layers to "
                         "separate an iteration-budget problem from a "
                         "regularisation-sensitive one, then exit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # --diagnose and --sweep-c exit before writing, so they need no
    # output path and must not trip the overwrite guard.
    diagnostic = bool(args.diagnose or args.sweep_c)
    if not diagnostic:
        if not args.out:
            sys.exit("--out is required unless running --diagnose or "
                     "--sweep-c")
        if args.resume_from and not Path(args.resume_from).exists():
            sys.exit(f"--resume-from does not exist: "
                     f"{args.resume_from}")
        out = Path(args.out)
        if out.exists():
            # Continuing an interrupted run rewrites its own output in
            # place, so that case is allowed -- but only when
            # --resume-from names exactly that file, compared after
            # resolution, so a mistyped path cannot overwrite a
            # finished collection.
            if not args.resume_from:
                sys.exit(f"{out} exists; refusing to overwrite.\nPass "
                         f"--resume-from {out} to continue it, or "
                         f"choose another --out.")
            if Path(args.resume_from).resolve() != out.resolve():
                sys.exit(f"--out exists ({out}) but --resume-from names "
                         f"a different file\n({args.resume_from}). "
                         f"Refusing to overwrite.")

    idx = pd.read_parquet(args.index)
    labels = load_labels(Path(args.labels))
    n_layers = int(idx.n_layers.iloc[0])
    layers = args.layers if args.layers else list(range(n_layers))

    bad_layers = [l for l in layers if not (0 <= l < n_layers)]
    if bad_layers:
        sys.exit(f"--layers out of range for a {n_layers}-layer trunk: "
                 f"{bad_layers}")

    train = idx[idx.split == "train"].reset_index(drop=True)
    test = idx[idx.split == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        sys.exit(f"split is unusable: {len(train)} train, {len(test)} "
                 f"test. Re-run the assembler with a different "
                 f"--test-frac or seed.")
    print(f"systems : {len(train):,} train, {len(test):,} test")
    print(f"layers  : {len(layers)} of {n_layers}")
    print(f"features: {', '.join(args.features)}")
    print(f"probe   : logistic, C={args.c}, max_iter={args.max_iter}, "
          f"class_weight={CLASS_WEIGHT} (same C at every layer)")

    missing = [s for s in idx.system_id if s not in labels]
    if missing:
        sys.exit(f"{len(missing)} systems have no label vector, "
                 f"e.g. {missing[:3]}")

    # Redundant with the assembler, which already enforces this. Kept
    # because a label vector of the wrong length would align the probe
    # to the wrong residues, and that is worth two lines of defence
    # rather than one.
    for r in idx.itertuples():
        if len(labels[r.system_id]) != int(r.length):
            sys.exit(f"{r.system_id}: label vector is "
                     f"{len(labels[r.system_id])} residues but the index "
                     f"says {int(r.length)}. Re-run the assembler.")

    absent = [q for q in QUADRANTS if (test.quadrant == q).sum() == 0]
    if absent:
        print(f"  note: {absent} have no test systems and will be "
              f"missing from the curves")

    for q in QUADRANTS:
        n = int((test.quadrant == q).sum())
        if 0 < n < 10:
            print(f"  note: {q} has only {n} test systems; its per-layer "
                  f"AUPRC will be noisy")

    # --- shared fitter for the diagnostic modes -------------------------
    def fit_once(which, layer, c, max_iter):
        """Fit on training data only; return True if it converged."""
        Xtr, ytr = [], []
        for r in train.itertuples():
            Xtr.append(load_layer(Path(r.path), layer, which))
            ytr.append(labels[r.system_id])
        Xtr = np.concatenate(Xtr).astype(np.float32)
        ytr = np.concatenate(ytr).astype(np.int8)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd < 1e-6] = 1.0
        Xtr = (Xtr - mu) / sd
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            LogisticRegression(C=c, max_iter=max_iter,
                               class_weight=CLASS_WEIGHT,
                               random_state=args.seed).fit(Xtr, ytr)
        del Xtr, ytr
        return not any(issubclass(w.category, ConvergenceWarning)
                       for w in caught)

    # --- diagnose: budget or regularisation? ----------------------------
    if args.diagnose:
        which = args.features[0]
        iters = [1000, 5000, 20000]
        cs = [1.0, 0.3, 0.1]
        print(f"\ndiagnostic matrix on '{which}', layers {layers}")
        print("Separates an ITERATION-BUDGET problem from a "
              "REGULARISATION-SENSITIVE one.\nIf C=1 converges at a large "
              "budget, the estimator need not be redefined.\n")
        print(f"{'C':>8}" + "".join(f"{f'iter={i}':>14}" for i in iters))
        for c in cs:
            cells = []
            for it in iters:
                bad = [l for l in layers if not fit_once(which, l, c, it)]
                cells.append("converges" if not bad
                             else f"fails@{len(bad)}")
            print(f"{c:>8g}" + "".join(f"{x:>14}" for x in cells))
        print("\n  'fails@n' = failed at n of the tested layers.")
        return 0

    # --- sweep: choose C by convergence, across all feature sets --------
    if args.sweep_c:
        # The selection rule is "the largest C converging at ALL layers
        # and for ALL feature sets". A sweep over a subset would report
        # "converges everywhere" for an everywhere that is seven layers
        # of one feature set, and that value could then be frozen as if
        # it satisfied the rule. Enforced rather than documented.
        if set(args.features) != set(FEATURE_SETS):
            sys.exit(
                f"--sweep-c selects the frozen C and must cover every "
                f"feature set {list(FEATURE_SETS)}, not "
                f"{args.features}.\nA C valid for 's' alone can fail "
                f"for 'both', which has twice the features.")
        if layers != list(range(n_layers)):
            sys.exit(
                f"--sweep-c must cover all {n_layers} layers; got "
                f"{len(layers)}.\nThe rule requires convergence "
                f"everywhere, and a subset cannot establish it. Use "
                f"--diagnose for an exploratory pass.")
        print(f"\nsweeping C over {args.sweep_c}")
        print(f"layers {layers}, feature sets {args.features}, "
              f"max_iter {args.max_iter}")
        print("Selection is by CONVERGENCE on training data only, never "
              "by held-out\nperformance, and must hold for every feature "
              "set: a C valid for 's'\nalone can fail for 'both', which "
              "has twice the features.\n")
        # Checkpoint every (C, feature set, layer) fit, not every C.
        # A C that fails is resolved by its first failing layer, but a C
        # that succeeds needs all 144 fits, and those successes are the
        # expensive work. Recording only at C granularity would discard
        # 143 converged fits if the walltime landed on the last one.
        state_path = Path(args.sweep_state) if args.sweep_state else None
        # Everything that determines a verdict goes in the config, and
        # the inputs are bound by content hash: a probe_index.parquet
        # replaced at the same path would otherwise let old convergence
        # verdicts be reused against new data.
        def _sha(path):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            return h.hexdigest()

        config = {"features": sorted(args.features), "layers": layers,
                  "max_iter": args.max_iter,
                  "class_weight": CLASS_WEIGHT, "seed": args.seed,
                  "index_sha256": _sha(args.index),
                  "labels_sha256": _sha(args.labels)}
        cells: dict = {}
        if state_path and state_path.exists():
            try:
                saved = json.loads(state_path.read_text())
            except Exception as e:  # noqa: BLE001
                sys.exit(f"cannot read --sweep-state {state_path.name}: "
                         f"{type(e).__name__}: {e}. Refusing to proceed "
                         f"on an unreadable record; delete it to start "
                         f"over.")
            if saved.get("config") != config:
                diff = [k for k in config
                        if saved.get("config", {}).get(k) != config[k]]
                sys.exit(f"{state_path.name} was written for a "
                         f"different sweep; {diff} differ.\nA "
                         f"convergence verdict does not carry across "
                         f"those. Use another --sweep-state.")
            # Validate rather than coerce: bool("false") is True, so
            # a string where a JSON boolean belongs would silently
            # invert a verdict. The file is written by this script and
            # should never contain one, which is exactly why a
            # violation means something is wrong rather than that a
            # default is needed.
            raw = saved.get("cells", {})
            bad = [k for k, v in raw.items() if not isinstance(v, bool)]
            if bad:
                sys.exit(f"{state_path.name} has non-boolean verdicts "
                         f"for {bad[:5]}; refusing to interpret them.")
            cells = {str(k): v for k, v in raw.items()}
            if cells:
                print(f"  resuming: {len(cells)} individual fits "
                      f"already completed\n", flush=True)

        def cell_key(c, which, layer):
            return f"{c:.17g}|{which}|{int(layer)}"

        def save_state():
            if state_path is None:
                return
            tmp = state_path.with_suffix(state_path.suffix + ".tmp")
            tmp.write_text(json.dumps({"config": config, "cells": cells},
                                      indent=2, sort_keys=True))
            tmp.replace(state_path)

        results = {}
        for c in sorted(args.sweep_c, reverse=True):
            failures, c_failed = {}, False
            print(f"\nC={c:g}", flush=True)
            for which in args.features:
                bad = []
                for layer in layers:
                    key = cell_key(c, which, layer)
                    if key in cells:
                        ok = cells[key]
                        print(f"  {which:6s} layer {layer:2d}: "
                              f"{'PASS' if ok else 'FAIL'} [saved]",
                              flush=True)
                    else:
                        ok = fit_once(which, layer, c, args.max_iter)
                        cells[key] = bool(ok)
                        save_state()
                        print(f"  {which:6s} layer {layer:2d}: "
                              f"{'PASS' if ok else 'FAIL'}", flush=True)
                    if not ok:
                        bad.append(layer)
                        failures[which] = bad
                        c_failed = True
                        break
                if c_failed:
                    break
            results[c] = failures
            if failures:
                detail = ", ".join(f"{k} at layers {v}"
                                   for k, v in failures.items())
                print(f"  C={c:<8g} FAILS: {detail}", flush=True)
            else:
                print(f"  C={c:<8g} converges everywhere", flush=True)

        good = [c for c, f in results.items() if not f]
        print()
        if good:
            print(f"  Largest C converging everywhere: {max(good)}")
            print(f"  Re-run with --c {max(good)}, and keep that value "
                  f"for every layer,\n  feature set and regime. Report "
                  f"neighbouring convergent values as\n  sensitivity "
                  f"analyses.")
        else:
            print("  Nothing on this grid converged everywhere. Extend "
                  "the grid downward\n  or raise --max-iter; do not "
                  "proceed with a C that fails at some\n  depths or for "
                  "some feature sets.")
        return 0

    # Each layer is written as it finishes. A curve run is 144 fits,
    # each followed by scoring every test protein, and the sweep showed
    # that much work exceeding a 12-hour walltime. Holding results in
    # memory until the end would discard all of it.
    # The configuration that produced these curve points, written
    # BEFORE any fitting. The run's .meta.json is only written at the
    # end, so a walltime-killed partial curve would otherwise carry no
    # record of the C, seed or inputs behind it -- and could then be
    # resumed under a different probe without anything noticing.
    def _sha_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    curve_config = {
        "C": float(args.c), "max_iter": int(args.max_iter),
        "class_weight": CLASS_WEIGHT, "seed": int(args.seed),
        "features": list(args.features),
        "layers": [int(x) for x in layers],
        "index_sha256": _sha_file(args.index),
        "labels_sha256": _sha_file(args.labels),
    }
    state_path = Path(str(out) + ".state.json")
    if args.resume_from:
        if not state_path.exists():
            sys.exit(f"{state_path.name} is missing, so the "
                     f"configuration behind the partial curve cannot "
                     f"be established.\nRefusing to resume.")
        saved = json.loads(state_path.read_text())
        if saved.get("config") != curve_config:
            diff = [k for k in curve_config
                    if saved.get("config", {}).get(k) != curve_config[k]]
            sys.exit(f"the partial curve was produced under a different "
                     f"configuration; {diff} differ.\nRefusing to mix "
                     f"curve points fitted with different probes.")
    else:
        tmp = Path(str(state_path) + ".tmp")
        tmp.write_text(json.dumps({"config": curve_config}, indent=2))
        tmp.replace(state_path)

    rows = []
    sys_rows = []
    sys_path = Path(str(out) + ".per_system.parquet")
    done_cells = set()
    if args.resume_from:
        prev = pd.read_parquet(args.resume_from)
        need = {"feature_set", "layer", "quadrant", "auprc_median"}
        if not need.issubset(prev.columns):
            sys.exit(f"--resume-from lacks {sorted(need - set(prev.columns))}")
        rows = prev.to_dict("records")
        # A cell counts as done only if it carries exactly one row per
        # quadrant that has test systems. One stray row would otherwise
        # make a partially written layer look complete and leave a hole
        # in the curve.
        expected_q = {q for q in QUADRANTS if (test.quadrant == q).any()}
        done_cells = set()
        for (feat, lay), g in prev.groupby(["feature_set", "layer"]):
            qs = list(g["quadrant"])
            if len(qs) != len(set(qs)):
                sys.exit(f"duplicate quadrant rows in the resumed cell "
                         f"{feat}, layer {lay}")
            if set(qs) != expected_q:
                sys.exit(f"incomplete resumed cell {feat}, layer {lay}: "
                         f"found {sorted(set(qs))}, expected "
                         f"{sorted(expected_q)}")
            done_cells.add((feat, int(lay)))
        # The sidecar must be restored too. flush_rows rewrites it
        # whole, so resuming with sys_rows empty would replace a file
        # holding 138 cells with one holding the handful this job adds.
        if not sys_path.exists():
            sys.exit(f"{sys_path.name} is missing although the curve "
                     f"is being resumed.\nRefusing to continue and "
                     f"leave an incomplete per-system record.")
        prev_sys = pd.read_parquet(sys_path)
        need_sys = {"system_id", "feature_set", "layer", "quadrant",
                    "auprc", "positive_rate"}
        if not need_sys.issubset(prev_sys.columns):
            sys.exit(f"{sys_path.name} lacks "
                     f"{sorted(need_sys - set(prev_sys.columns))}")
        sys_cells = set(zip(prev_sys.feature_set,
                            prev_sys.layer.astype(int)))
        if sys_cells != done_cells:
            only_agg = sorted(done_cells - sys_cells)[:5]
            only_sys = sorted(sys_cells - done_cells)[:5]
            sys.exit(f"the curve and its per-system sidecar disagree "
                     f"about which cells are done.\n  in curve only: "
                     f"{only_agg}\n  in sidecar only: {only_sys}\n"
                     f"Refusing to resume from an inconsistent pair.")
        sys_rows = prev_sys.to_dict("records")
        print(f"  resuming: {len(done_cells)} (feature set, layer) "
              f"pairs already scored\n", flush=True)

    def flush_rows():
        if not rows:
            return
        tmp = Path(str(out) + ".tmp")
        pd.DataFrame(rows).to_parquet(tmp)
        tmp.replace(out)
        if sys_rows:
            tmp2 = Path(str(sys_path) + ".tmp")
            pd.DataFrame(sys_rows).to_parquet(tmp2)
            tmp2.replace(sys_path)

    t0 = time.time()
    for which in args.features:
        for li, layer in enumerate(layers):
            if (which, layer) in done_cells:
                continue
            # Assemble this layer only. Holding every layer at once would
            # need tens of gigabytes; one layer is a few hundred MB.
            Xtr, ytr = [], []
            for r in train.itertuples():
                Xtr.append(load_layer(Path(r.path), layer, which))
                ytr.append(labels[r.system_id])
            Xtr = np.concatenate(Xtr).astype(np.float32)
            ytr = np.concatenate(ytr).astype(np.int8)

            # Fitted on training residues only, applied unchanged to test.
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0)
            sd[sd < 1e-6] = 1.0
            Xtr = (Xtr - mu) / sd

            clf = LogisticRegression(
                C=args.c, max_iter=args.max_iter,
                class_weight=CLASS_WEIGHT, random_state=args.seed)
            # A probe that did not converge produces a point on the depth
            # curve that reflects where the optimiser stopped rather than
            # what the representation supports. sklearn only warns, so
            # one bad layer would sit in the curve unnoticed and the
            # onset rule would treat it as data.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(Xtr, ytr)
            if any(issubclass(w.category, ConvergenceWarning)
                   for w in caught):
                raise RuntimeError(
                    f"probe did not converge at feature={which}, "
                    f"layer={layer}, C={args.c}.\nRun --sweep-c to find "
                    f"a value that converges at every layer. Do not "
                    f"raise\nC for some layers only: the same capacity "
                    f"must apply at every depth.")
            del Xtr, ytr

            # Score each test protein separately, then aggregate per
            # quadrant. A protein with no positives or no negatives has
            # no defined AUPRC and is skipped rather than scored as zero.
            per_sys = []
            for r in test.itertuples():
                X = load_layer(Path(r.path), layer, which)
                y = labels[r.system_id]
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                p = clf.decision_function((X - mu) / sd)
                a = float(average_precision_score(y, p))
                per_sys.append((r.quadrant, r.system_id, a,
                                float(y.mean())))
                # Kept alongside the aggregates. The median is what the
                # curve reports, but a per-protein regression -- does
                # length or pocket size explain this score? -- needs the
                # individual values, and recovering them later means
                # refitting.
                sys_rows.append({"system_id": r.system_id,
                                 "feature_set": which,
                                 "layer": int(layer),
                                 "quadrant": r.quadrant, "auprc": a,
                                 "positive_rate": float(y.mean())})

            for q in QUADRANTS:
                vals = [a for (qq, _, a, _) in per_sys if qq == q]
                base = [b for (qq, _, _, b) in per_sys if qq == q]
                if not vals:
                    continue
                rows.append({
                    "feature_set": which, "layer": int(layer),
                    "quadrant": q, "n_systems": len(vals),
                    "auprc_median": float(np.median(vals)),
                    "auprc_mean": float(np.mean(vals)),
                    "auprc_q25": float(np.quantile(vals, 0.25)),
                    "auprc_q75": float(np.quantile(vals, 0.75)),
                    # The positive rate is the AUPRC a random ranking
                    # would reach, so a curve must be read against it
                    # rather than against zero.
                    "baseline_median": float(np.median(base)),
                })

            flush_rows()
            if True:
                el = time.time() - t0
                done = (args.features.index(which) * len(layers)) + li + 1
                tot = len(args.features) * len(layers)
                print(f"  [{done}/{tot}] {which} layer {layer:>2} "
                      f"| {el/60:.1f} min elapsed", flush=True)

    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("no results; check that test systems have both classes")
    flush_rows()

    print(f"\n{'feat':7}{'cell':6}{'first':>8}{'last':>8}{'peak':>8}"
          f"{'base':>8}")
    for which in args.features:
        for q in QUADRANTS:
            sub = res[(res.feature_set == which)
                      & (res.quadrant == q)].sort_values("layer")
            if sub.empty:
                continue
            print(f"{which:7}{q:6}{sub.auprc_median.iloc[0]:>8.3f}"
                  f"{sub.auprc_median.iloc[-1]:>8.3f}"
                  f"{sub.auprc_median.max():>8.3f}"
                  f"{sub.baseline_median.iloc[0]:>8.3f}")

    Path(str(out) + ".meta.json").write_text(json.dumps({
        "C": args.c, "max_iter": args.max_iter,
        "class_weight": CLASS_WEIGHT, "seed": args.seed,
        "layers": [int(x) for x in layers],
        "feature_sets": list(args.features),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "aggregation": "median AUPRC across test proteins",
        "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))
    print(f"\nWrote {out}")
    print(f"\n  'base' is the median positive rate: the AUPRC a random\n"
          f"  ranking achieves. A curve is only above chance where it\n"
          f"  exceeds that, and the null calibration makes that precise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
