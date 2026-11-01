#!/usr/bin/env python3
"""
Measure the false-positive rate of the onset rule on this study's own
null, and apply the rule to the observed curves.

Two jobs, kept in one place because they must use the same rule.

1. AUDIT. How often does a trajectory containing no pocket information
   produce an onset? The synthetic calibration showed a three-stage rule
   firing nineteen layers early under transformer-like depth
   correlation, which is why five stages were pre-registered. This
   measures the surviving error rate on the real representations rather
   than inferring it from a simulation.

2. APPLY. The onset for each observed curve, with "unresolved" reported
   rather than defaulted when no qualifying run exists.

WHY LEAVE ONE OUT
-----------------
tau at each layer is computed from all B null replicates. Asking how
often replicate b exceeds a threshold that b itself helped construct is
an in-sample question, and it understates the error rate: b's own
contribution pushes tau up at exactly the layers where b is high.

For the audit, each replicate is therefore tested against a threshold
built from the other B - 1:

    tau_l^(-b) = the k-th smallest of the other replicates' values
    k = ceil((1 - alpha) ((B - 1) + 1))      = 95 for B = 100

This needs no additional probe fits -- every replicate's per-layer AUPRC
is already saved -- and it is the number to report as the rule's
empirical false-positive rate.

The observed curves are still tested against the full-B threshold. The
leave-one-out construction exists only to keep the audit honest; using
it on real data would discard a replicate for no reason.

WHAT THE RULE IS
----------------
The onset is the first layer beginning PERSIST consecutive layers whose
AUPRC exceeds tau at their own layer:

    l* = min { l : M_j > tau_j for all j in [l, l + PERSIST - 1] }

PERSIST = 5, pre-registered. The audit also reports the rate at other
values so the choice can be seen rather than asserted; those are
diagnostics, not alternative endpoints.

ADJACENT STAGES, NOT ADJACENT MEASUREMENTS
------------------------------------------
The pre-registered rule requires five adjacent REPRESENTATION STAGES:
layers l, l+1, l+2, l+3, l+4. On a sparse grid such as 0, 4, 8, 12, 16
those are five consecutive measurements spanning seventeen stages, and
treating them as a persistence window would silently weaken the rule
that the synthetic calibration exists to justify -- persistence 5 was
chosen because shorter runs fire early under depth correlation, and
five widely spaced samples are a shorter run.

Sparse-layer nulls are therefore allowed for pilot calibration
summaries but NOT for the primary onset or its false-positive audit.

Unit spacing alone is not enough. Layers 5..47 are unit-spaced and so
are 0..40, yet either would evaluate the endpoint on part of the trunk
while reporting it as the trunk's onset -- and the observed-curve path
merges curve against tau on their intersection, so a truncation at
either end would pass unnoticed. The primary audit therefore requires
the COMPLETE layer set, 0..n_layers-1, and refuses anything else.

--n-layers is explicit rather than fixed at 48 because a second trunk
will have a different depth; it is required so the expected set is
always stated rather than inferred from whatever was supplied.

USAGE
-----
    python3 audit_onset_null.py \\
        --replicates null_tau_replicates.parquet \\
        --tau null_tau.parquet \\
        --curves depth_curves.parquet \\
        --out onset_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05
PERSIST = 5          # pre-registered
DIAGNOSTIC_PERSIST = (1, 2, 3, 4, 5, 6, 7)


def order_tau(vals: np.ndarray, alpha: float = ALPHA) -> float:
    """k-th smallest with k = ceil((1-alpha)(B+1)); see calibrate_null."""
    v = np.sort(vals)
    k = min(int(np.ceil((1 - alpha) * (v.size + 1))), v.size)
    return float(v[k - 1])


def first_run(exceeds: np.ndarray, layers: np.ndarray,
              persist: int) -> int | None:
    """First layer starting `persist` consecutive exceedances."""
    run = 0
    for i, e in enumerate(exceeds):
        run = run + 1 if e else 0
        if run >= persist:
            return int(layers[i - persist + 1])
    return None


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicates", required=True)
    ap.add_argument("--tau", required=True)
    ap.add_argument("--curves", help="observed curves; omit to audit only")
    ap.add_argument("--n-layers", type=int, required=True,
                    help="the trunk's depth. The primary onset is "
                         "defined on the whole trunk, so layers 0.."
                         "n-1 must all be present; stating it here "
                         "prevents a truncated calibration being "
                         "reported as the trunk's onset.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="permit a layer subset. For pilot summaries "
                         "only: the result is NOT the pre-registered "
                         "onset and is labelled as such.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    reps = pd.read_parquet(args.replicates)
    tau = pd.read_parquet(args.tau)
    for c in ("feature_set", "layer", "quadrant", "perm_id",
              "auprc_median"):
        if c not in reps.columns:
            sys.exit(f"replicate file lacks '{c}'")

    expected = np.arange(args.n_layers)

    def require_full_layers(layers, where):
        lay = np.asarray(layers, dtype=int)
        if np.array_equal(lay, expected):
            return
        gaps = np.diff(lay)
        why = ("not unit-spaced (gaps "
               f"{sorted(set(gaps.tolist()))})"
               if gaps.size and not np.all(gaps == 1)
               else "unit-spaced but incomplete")
        missing = sorted(set(expected.tolist()) - set(lay.tolist()))
        extra = sorted(set(lay.tolist()) - set(expected.tolist()))
        msg = (f"{where}: the pre-registered onset is defined on the "
               f"whole trunk, layers 0..{args.n_layers - 1}, and "
               f"requires {PERSIST} ADJACENT stages.\n  supplied: "
               f"{why}\n  missing: {missing[:12]}"
               f"{' ...' if len(missing) > 12 else ''}\n  extra: "
               f"{extra[:12]}{' ...' if len(extra) > 12 else ''}")
        if not args.allow_partial:
            sys.exit(msg + "\n\nPass --allow-partial for a pilot "
                           "summary, which is not the pre-registered "
                           "onset.")
        print(f"  WARNING, partial layer set: {where}. The reported "
              f"onset is NOT the pre-registered endpoint.")

    for c in ("feature_set", "layer", "quadrant", "tau"):
        if c not in tau.columns:
            sys.exit(f"tau file lacks '{c}'")

    # The key sets must match in BOTH directions. The reconstruction
    # below iterates over the replicate groups, so a tau file carrying
    # an extra (feature set, regime, layer) -- a leftover from another
    # run -- would never be visited and would pass unexamined.
    def _keys(df):
        return set(map(tuple, df[["feature_set", "quadrant", "layer"]]
                       .drop_duplicates()
                       .itertuples(index=False, name=None)))

    rk, tk = _keys(reps), _keys(tau)
    if rk != tk:
        missing = sorted(rk - tk)
        extra = sorted(tk - rk)
        sys.exit("the tau and replicate files describe different "
                 "cells:\n"
                 f"  in replicates, absent from tau: {missing[:8]}\n"
                 f"  in tau, absent from replicates: {extra[:8]}")

    # tau is supplied as its own file, but it must be the tau these
    # replicates produced. Reconstructing it costs nothing and prevents
    # the silent pairing of one run's trajectories with another run's
    # threshold, which would otherwise yield a perfectly ordinary
    # looking result.
    for (feature, quadrant, layer), g in reps.groupby(
            ["feature_set", "quadrant", "layer"]):
        row = tau[(tau.feature_set == feature)
                  & (tau.quadrant == quadrant) & (tau.layer == layer)]
        if len(row) != 1:
            sys.exit(f"{feature}/{quadrant} layer {layer}: tau row "
                     f"missing or duplicated ({len(row)} rows)")
        want = order_tau(g.auprc_median.to_numpy())
        if not np.isclose(float(row.tau.iloc[0]), want,
                          rtol=0, atol=1e-12):
            sys.exit(
                f"{feature}/{quadrant} layer {layer}: tau "
                f"{float(row.tau.iloc[0]):.6f} does not match the "
                f"{want:.6f} these replicates produce. The two files "
                f"are from different runs.")
    print(f"tau matches the supplied replicates at all "
          f"{len(reps.groupby(['feature_set','quadrant','layer']))} "
          f"(feature, regime, layer) combinations\n")

    results = {"persist": PERSIST, "alpha": ALPHA,
               "n_layers_expected": int(args.n_layers),
               "partial_allowed": bool(args.allow_partial),
               "audit": {}, "observed": {}}

    # --- 1. audit -------------------------------------------------------
    print("Empirical false-positive rate of the onset rule, "
          "leave-one-replicate-out.\n")
    print(f"{'curve':16}{'B':>5}{'layers':>8}{'spacing':>9}"
          + "".join(f"{f'p={p}':>8}" for p in DIAGNOSTIC_PERSIST))
    for (feature, quadrant), sub in reps.groupby(["feature_set",
                                                  "quadrant"]):
        # pivot, not pivot_table: the latter silently averages
        # duplicate (perm_id, layer) rows, which in a null audit would
        # quietly merge two trajectories into one.
        if sub.duplicated(["perm_id", "layer"]).any():
            sys.exit(f"{feature}/{quadrant}: duplicate perm_id/layer "
                     f"rows in the replicate file")
        wide = sub.pivot(index="perm_id", columns="layer",
                         values="auprc_median")
        if wide.isna().any().any():
            n_missing = int(wide.isna().sum().sum())
            sys.exit(f"{feature}/{quadrant}: {n_missing} missing "
                     f"entries -- some replicates lack some layers, so "
                     f"the trajectories are incomplete and a run of "
                     f"exceedances cannot be evaluated across them")
        layers = np.asarray(sorted(wide.columns))
        M = wide[layers].to_numpy()          # (B, n_layers)
        B = M.shape[0]
        if B < 20:
            print(f"  skipping {feature}/{quadrant}: only {B} replicates")
            continue
        require_full_layers(layers,
                            f"null replicates {feature}/{quadrant}")
        # Recorded, not assumed: in --allow-partial mode the grid need
        # not be unit-spaced, and writing "1" there would make the
        # output claim something false about how the run was done.
        gaps = sorted(set(np.diff(layers).tolist()))
        spacing = ("1" if gaps == [1] and len(layers) == args.n_layers
                   else f"partial:{gaps}")

        rates = {}
        for persist in DIAGNOSTIC_PERSIST:
            fired = 0
            for b in range(B):
                others = np.delete(M, b, axis=0)
                # Threshold from the other B-1 replicates only, so the
                # trajectory under test did not help set its own bar.
                t = np.array([order_tau(others[:, j])
                              for j in range(len(layers))])
                if first_run(M[b] > t, layers, persist) is not None:
                    fired += 1
            rates[persist] = fired / B

        key = f"{feature}/{quadrant}"
        results["audit"][key] = {
            "n_replicates": int(B), "n_layers": int(len(layers)),
            "layer_spacing": spacing,
            "layers": [int(x) for x in layers],
            "fpr_by_persistence": {str(k): v for k, v in rates.items()},
            "fpr_at_preregistered_persistence": rates[PERSIST],
        }
        print(f"{key:16}{B:>5}{len(layers):>8}{spacing:>9}"
              + "".join(f"{rates[p]:>8.2f}" for p in DIAGNOSTIC_PERSIST))

    print(f"\n  Each column is the fraction of NULL trajectories that\n"
          f"  produce an onset at that persistence. p={PERSIST} is the\n"
          f"  pre-registered rule; the others show how the error rate\n"
          f"  falls with persistence and are diagnostics, not\n"
          f"  alternative endpoints.")

    # --- 2. apply -------------------------------------------------------
    if args.curves:
        curves = pd.read_parquet(args.curves)
        print(f"\n\nOnset on the observed curves, against the full-B "
              f"threshold.\n")
        print(f"{'curve':16}{'onset':>8}{'first>tau':>11}"
              f"{'n above':>9}")
        for (feature, quadrant), sub in curves.groupby(["feature_set",
                                                        "quadrant"]):
            t = tau[(tau.feature_set == feature)
                    & (tau.quadrant == quadrant)]
            if t.empty:
                print(f"  {feature}/{quadrant}: no tau; skipped")
                continue
            m = sub[["layer", "auprc_median"]].merge(
                t[["layer", "tau"]], on="layer", how="inner"
            ).sort_values("layer")
            if m.empty:
                print(f"  {feature}/{quadrant}: no layers in common "
                      f"between curve and tau; skipped")
                continue
            layers = m.layer.to_numpy()
            require_full_layers(layers, f"observed curve {feature}/"
                                        f"{quadrant}")
            exceeds = (m.auprc_median > m.tau).to_numpy()
            onset = first_run(exceeds, layers, PERSIST)
            first_over = int(layers[exceeds.argmax()]) if exceeds.any() \
                else None
            key = f"{feature}/{quadrant}"
            results["observed"][key] = {
                "onset": onset,
                "onset_resolved": onset is not None,
                "first_layer_above_tau": first_over,
                "n_layers_above_tau": int(exceeds.sum()),
                "n_layers_compared": int(len(layers)),
            }
            print(f"{key:16}"
                  f"{(onset if onset is not None else 'unresolved'):>8}"
                  f"{(first_over if first_over is not None else '--'):>11}"
                  f"{int(exceeds.sum()):>9}")
        print(f"\n  'unresolved' means no {PERSIST} adjacent\n"
              f"  representation layers stayed above tau. It is reported\n"
              f"  as such rather\n"
              f"  than defaulted to a layer, because 'the signal never\n"
              f"  became reliably decodable' and 'it became decodable at\n"
              f"  the last layer' are different findings.")

    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
