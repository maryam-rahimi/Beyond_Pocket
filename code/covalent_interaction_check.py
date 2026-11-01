#!/usr/bin/env python3
"""
Does the covalent association differ between the four regimes, or is
one coefficient enough?

WHY THIS FOLLOWS THE POOLED FIT
-------------------------------
The pooled model found covalent systems lower by 0.221 in mean AUPRC,
interval -0.279 to -0.160, barely moved by adjusting for regime. But
that model contains no covalent-by-regime term, so it fits one
coefficient and assumes the association is common rather than showing
it is. Its own per-regime medians do not all agree:

    FF   covalent 0.113   other 0.579
    FN   covalent 0.116   other 0.318
    NF   covalent 0.650   other 0.596
    NN   covalent 0.044   other 0.529

Three regimes point one way and NF points the other, on fifteen
systems. Whether that reversal is real decides what can be said. If
the association is common, "covalent systems score lower" stands as
written. If NF genuinely differs, the claim is narrower and NF needs
an account.

WHAT IS FITTED
--------------
Two least-squares models on the same test systems:

    additive      auprc ~ covalent + regime
    interaction   auprc ~ covalent * regime

and the comparison is the drop in residual sum of squares from adding
the three interaction terms, alongside the per-regime covalent slopes
the interaction model implies.

THE NULL HAS TO BE SIMULATED, NOT RESAMPLED
-------------------------------------------
The drop cannot be negative: the interaction model contains the
additive one, so three more parameters never raise the training RSS.
A percentile interval from resampling the observed data therefore sits
above zero whether or not any interaction exists -- it inherits
whatever apparent heterogeneity the sample happens to carry, and
fitting three extra parameters to noise produces a positive drop on
its own. An interval excluding zero would mean nothing.

So the null is constructed rather than resampled, by a wild cluster
bootstrap. The additive model is fitted, its residuals kept, and each
replicate rebuilds the outcome as the additive fit plus those
residuals with a random sign drawn once per receptor-family cluster.
That generates data in which the additive model is true while the
cluster-correlated residual structure survives. Refitting both models
to each replicate gives the drop expected under no interaction, and
the observed drop is read against that.

A wild bootstrap rather than an F test for the reason the rest of this
study resamples: proteins within a family share sequence and fold, and
an F test would take the residual degrees of freedom at face value.

WHAT THE ANSWER MEANS
---------------------
An observed drop beyond what the simulated null produces means one
coefficient is not enough: the covalent association differs by regime
and the pooled number describes none of them exactly. A drop inside
the null's range means the additive model is not refuted on this
sample, which is weaker than saying the association is common -- NF's
half holds fifteen systems and that is thin whichever way it points.

Exploratory, like the pooled fit it follows.

USAGE
-----
    python3 covalent_interaction_check.py \\
        --index probe_index.parquet \\
        --per-system per_system_auprc.parquet \\
        --plinder-dir ~/.local/share/plinder/2026-09 \\
        --feature s --layer 18 \\
        --out covalent_interaction.json
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
MIN_HALF = 15          # as in the split check, and for the same reason


def design(frame, interaction):
    """Design matrix for the additive or the interaction model."""
    cov = frame.covalent.to_numpy(float)
    cols = [np.ones(len(frame)), cov]
    for q in QUADRANTS[1:]:                  # FF is the reference
        ind = (frame.quadrant == q).to_numpy(float)
        cols.append(ind)
        if interaction:
            cols.append(ind * cov)
    return np.column_stack(cols)


def fit_rss(X, y):
    """RSS, or nan if the design is rank deficient.

    lstsq solves a rank-deficient system silently, which in a
    replicate that has lost a subgroup would return an RSS from a
    model that is not the one named. Checked rather than trusted.
    """
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return float("nan"), None
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), None
    return float(np.sum((y - X @ beta) ** 2)), beta


def rss(frame, interaction, y=None):
    """Residual sum of squares, or nan where the fit is not defined."""
    yy = frame.auprc.to_numpy(float) if y is None else y
    if len(np.unique(frame.covalent.to_numpy(float))) < 2:
        return float("nan")
    return fit_rss(design(frame, interaction), yy)[0]


def per_regime_slopes(frame):
    """The covalent slope the interaction model implies in each
    regime: the base coefficient, plus that regime's interaction."""
    y = frame.auprc.to_numpy(float)
    _, beta = fit_rss(design(frame, True), y)
    if beta is None:
        return {q: float("nan") for q in QUADRANTS}
    base = beta[1]
    out = {QUADRANTS[0]: float(base)}
    for i, q in enumerate(QUADRANTS[1:]):
        out[q] = float(base + beta[3 + 2 * i])
    return out


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-system", required=True)
    ap.add_argument("--plinder-dir", required=True)
    ap.add_argument("--feature", default="s")
    ap.add_argument("--layer", type=int)
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
        sys.exit("some test systems have no family_cluster; the "
                 "cluster bootstrap is undefined for them.")
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
                 f"no covalent flag.")
    d["covalent"] = d.covalent.astype(bool)
    print(f"test systems {len(d):,}: {int(d.covalent.sum()):,} "
          f"covalent")

    slopes = per_regime_slopes(d)
    print(f"\n{'regime':8}{'covalent n':>12}{'slope':>10}")
    sizes = {}
    for q in QUADRANTS:
        s = d[d.quadrant == q]
        n_c = int(s.covalent.sum())
        sizes[q] = {"n_covalent": n_c,
                    "n_other": int(len(s) - n_c),
                    "below_floor": bool(n_c < MIN_HALF
                                        or len(s) - n_c < MIN_HALF)}
        print(f"{q:8}{n_c:>12,}{slopes[q]:>+10.3f}"
              + ("  (small)" if sizes[q]["below_floor"] else ""))

    r_add, r_int = rss(d, False), rss(d, True)
    drop = r_add - r_int
    print(f"\nresidual sum of squares")
    print(f"  additive     {r_add:.3f}")
    print(f"  interaction  {r_int:.3f}")
    print(f"  drop         {drop:.3f}")

    # The null: the additive fit, plus its own residuals with a sign
    # flipped per cluster. Rademacher weights, one per receptor
    # family, so the cluster-correlated residual structure survives
    # while any interaction in the data is removed.
    rng = np.random.default_rng(args.seed)
    X_add = design(d, False)
    y_obs = d.auprc.to_numpy(float)
    _, beta0 = fit_rss(X_add, y_obs)
    if beta0 is None:
        sys.exit("the additive model is rank deficient on the "
                 "observed data; the null cannot be built from it.")
    fitted0 = X_add @ beta0
    resid0 = y_obs - fitted0
    codes = pd.Categorical(d.family_cluster).codes
    n_cl = int(codes.max()) + 1

    null = np.empty(args.n_boot)
    for i in range(args.n_boot):
        w = rng.choice((-1.0, 1.0), size=n_cl)[codes]
        y_star = fitted0 + w * resid0
        a, b = rss(d, False, y_star), rss(d, True, y_star)
        null[i] = (a - b) if not (np.isnan(a) or np.isnan(b)) \
            else float("nan")
    null = null[~np.isnan(null)]
    if len(null) < args.n_boot * 0.5:
        print(f"\n  {args.n_boot - len(null):,} replicates were rank "
              f"deficient and dropped.")
    q95 = float(np.percentile(null, 95))
    p = (1 + int(np.sum(null >= drop))) / (len(null) + 1)
    excl = bool(drop > q95)
    print(f"\nnull drop, wild cluster bootstrap under the additive "
          f"model")
    print(f"  median       {np.median(null):.3f}")
    print(f"  95th pct     {q95:.3f}")
    print(f"  observed     {drop:.3f}")
    print(f"  p            {p:.4f}")

    small = [q for q in QUADRANTS if sizes[q]["below_floor"]]
    if excl:
        verdict = ("the observed drop exceeds the 95th percentile of "
                   "the simulated null: one\n  coefficient does not "
                   "describe all four regimes, and the pooled "
                   "estimate\n  describes none of them exactly")
    else:
        verdict = ("the observed drop falls inside the null's range: "
                   "the additive model is\n  not refuted on this "
                   "sample. That is weaker than showing the "
                   "association\n  is common")
    if small:
        verdict += (f". {', '.join(small)} "
                    f"{'has' if len(small) == 1 else 'have'} a half "
                    f"below {MIN_HALF} systems, so the "
                    f"{'slope' if len(small) == 1 else 'slopes'} "
                    f"above {'is' if len(small) == 1 else 'are'} thin "
                    f"whichever way\n  they point")
    print(f"\n  {verdict}.")

    res = {"feature": args.feature, "layer": selected_layer,
           "flag": FLAG, "exploratory": True,
           "models": {"additive": "auprc ~ covalent + regime",
                      "interaction": "auprc ~ covalent * regime"},
           "rss_additive": round(r_add, 4),
           "rss_interaction": round(r_int, 4),
           "rss_drop": round(drop, 4),
           "null": {"method": "wild cluster bootstrap under the "
                              "additive model, Rademacher signs per "
                              "receptor family",
                    "median": round(float(np.median(null)), 4),
                    "pct95": round(q95, 4),
                    "p": round(p, 4),
                    "n_replicates_used": int(len(null))},
           "drop_exceeds_null": excl,
           "per_regime_slope": {q: round(slopes[q], 4)
                                for q in QUADRANTS},
           "sizes": sizes, "min_half": MIN_HALF,
           "n_boot": args.n_boot, "seed": args.seed,
           "resampled": "family_cluster",
           "plinder_dir": str(Path(args.plinder_dir).expanduser())}
    out.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out}")
    print("\n  A slope per regime is not a mechanism. It says whether "
          "one number can\n  stand for four, which is a question "
          "about how to report the pooled\n  association rather than "
          "about why covalent systems score as they do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
