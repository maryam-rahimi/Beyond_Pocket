#!/usr/bin/env python3
"""
Fit pocket decodability on both familiarity axes continuously,
without dichotomising either for this analysis.

WHY, AFTER THE CELLS
--------------------
Every result so far rests on four cells drawn at a chosen value of
pocket coverage. The threshold sensitivity showed the hard regime
lowest wherever that line is drawn, which is reassuring, but a
continuous model removes the line rather than varying it. If pocket
familiarity predicts decodability as a slope, the finding no longer
depends on the coverage dichotomisation used to build the four
regimes. It still depends on the sequence clustering behind the
family axis, which is a threshold of a different kind and is not
removed here.

The family axis has a second reason. Its binary form is effectively
zero relatives against some: every novel system has a count of zero,
while the familiar group runs from 7 at its tenth percentile to 643 at
its ninetieth and 4,560 at its maximum. MIRAGE measures its own
familiarity gap between support 1 and support 301 or more, and both of
those fall inside this study's single familiar cell. So a null on the
binary axis is consistent with two things -- family support does not
matter, or a gradient within one cell was never resolved -- and only
the continuous form separates them.

THE MODEL
---------
    AUPRC_i = b0 + b_f log10(1 + S_f,i) + b_p S_p,i
                 + b_fp log10(1 + S_f,i) S_p,i + e_i

fitted over the test proteins, with intervals from resampling
receptor-family clusters as elsewhere. Both axes are standardised
before fitting so the coefficients are comparable in size; the
unstandardised fit is reported alongside, since a slope per unit of
coverage is the interpretable quantity and a slope per standard
deviation is the comparable one.

BOTH FITS ARE CENTRED
---------------------
In a model with an interaction, a main effect is the slope of one
axis where the other is zero. Uncentred, the raw pocket coefficient
would be the pocket slope for a protein with no pre-cutoff relatives
at all, and the raw family coefficient the family slope at zero
pocket coverage -- neither of which is the quantity the text would
claim. Centring both axes makes each main effect the slope at the
other axis's mean, which is what a reader takes a main effect to
mean. The raw fit is centred but not scaled, so its units stay
interpretable.

S_f enters as log10(1 + count). The counts run from 0 to several
thousand with a long tail, so a linear term would be set almost
entirely by the largest families.

WHAT IS DROPPED, AND WHY IT IS REPORTED
---------------------------------------
Targets whose receptor is only partly resolved carry a lower bound on
support rather than a count, and are excluded. Their number is
reported rather than absorbed: a regression fitted on the systems
whose support happens to be exactly known is not quite a regression on
the test set.

TWO LIMITS, STATED BEFORE THE NUMBERS
-------------------------------------
AUPRC is bounded in [0, 1] and a linear model does not respect that,
so the fit is an approximation that will be worst where predicted
values approach either end.

The pocket metric has heavy ceiling mass -- many targets sit at or
near complete coverage -- which compresses the top of its range and
limits what any continuous model can resolve there. That is a known
property of the measure, recorded when it was chosen, and it is the
reason the cells were retained alongside this fit rather than replaced
by it.

USAGE
-----
    python3 continuous_two_axis.py \\
        --index probe_index.parquet \\
        --per-system per_system_auprc.parquet \\
        --targets targets_with_sf_count.parquet \\
        --feature s --layer 18 \\
        --out continuous_two_axis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TERMS = ("intercept", "S_f", "S_p", "S_f x S_p")


def build(frame, standardise, centre=None, scale=None):
    """Design matrix for the two-axis model.

    Both forms are CENTRED; only the standardised one is scaled. An
    uncentred interaction model gives main effects at the other
    axis's zero, which here would mean a protein with no pre-cutoff
    relatives or a pocket with no coverage at all.

    Returns the centring and scaling used, so a bootstrap replicate
    can be put on the same footing as the observed fit rather than
    recentred to itself.
    """
    f = frame.S_f_log10.to_numpy(float)
    p = frame.S_p.to_numpy(float)
    if centre is None:
        centre = (f.mean(), p.mean())
        scale = (f.std() or 1.0, p.std() or 1.0)
    f = f - centre[0]
    p = p - centre[1]
    if standardise:
        f = f / scale[0]
        p = p / scale[1]
    return (np.column_stack([np.ones(len(frame)), f, p, f * p]),
            centre, scale)


def fit(frame, standardise, centre=None, scale=None):
    """Coefficients, or nan where the design is rank deficient."""
    X, c, s = build(frame, standardise, centre, scale)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return np.full(X.shape[1], np.nan), c, s
    try:
        beta, *_ = np.linalg.lstsq(X, frame.auprc.to_numpy(float),
                                   rcond=None)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), c, s
    return beta, c, s


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-system", required=True)
    ap.add_argument("--targets", required=True,
                    help="the table carrying S_f_count, S_f_log10, "
                         "S_f_status and S_p, from "
                         "build_continuous_sf.py")
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
    tg = pd.read_parquet(args.targets)
    for c in ("system_id", "split", "family_cluster"):
        if c not in idx.columns:
            sys.exit(f"--index has no '{c}'")
    for c in ("system_id", "S_f_count", "S_f_log10", "S_f_status",
              "S_p"):
        if c not in tg.columns:
            sys.exit(f"--targets has no '{c}'. Run "
                     f"build_continuous_sf.py first.")
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

    test = idx[idx.split == "test"][["system_id", "family_cluster"]]
    if test.family_cluster.isna().any():
        sys.exit("some test systems have no family_cluster; the "
                 "cluster bootstrap is undefined for them.")
    d = test.merge(scores[["system_id", "auprc"]], on="system_id",
                   how="inner", validate="one_to_one")
    if len(d) != len(test):
        sys.exit(f"only {len(d):,} of {len(test):,} test systems have "
                 f"an AUPRC score.")
    d = d.merge(tg[["system_id", "S_f_log10", "S_f_count",
                    "S_f_status", "S_p"]], on="system_id",
                how="left", validate="one_to_one")
    if d.S_f_status.isna().any():
        sys.exit(f"{int(d.S_f_status.isna().sum())} test systems are "
                 f"absent from --targets.")

    n_all = len(d)
    dropped = d[d.S_f_status != "resolved"]
    d = d[d.S_f_status == "resolved"].reset_index(drop=True)
    print(f"test systems {n_all:,}; {len(dropped):,} dropped for a "
          f"partly resolved receptor")
    # Resolved guarantees an exact family count. It does not
    # guarantee the other analysis fields are finite.
    need = ["auprc", "S_f_count", "S_f_log10", "S_p"]
    bad = ~np.isfinite(d[need].to_numpy(float)).all(axis=1)
    if bad.any():
        sys.exit(f"{int(bad.sum())} of the resolved systems carry a "
                 f"non-finite value among {need}.")
    print(f"fitted on {len(d):,}")
    print(f"  S_f count   median {d.S_f_count.median():.0f}, "
          f"range {d.S_f_count.min():.0f} to {d.S_f_count.max():.0f}")
    print(f"  S_p         median {d.S_p.median():.1f}, "
          f"range {d.S_p.min():.0f} to {d.S_p.max():.0f}")
    at_ceiling = float((d.S_p >= 99).mean())
    print(f"  S_p at or above 99: {at_ceiling:.1%}")

    beta_s, centre, scale = fit(d, True)
    beta_r, _, _ = fit(d, False, centre, scale)
    if np.isnan(beta_s).any():
        sys.exit("the model is rank deficient on the observed data.")

    rng = np.random.default_rng(args.seed)
    clusters = d.family_cluster.unique()
    by_cl = {c: g for c, g in d.groupby("family_cluster")}
    bs = np.empty((args.n_boot, 4))
    br = np.empty((args.n_boot, 4))
    for i in range(args.n_boot):
        rep = pd.concat([by_cl[c] for c in
                         rng.choice(clusters, len(clusters),
                                    replace=True)], ignore_index=True)
        # The observed centring and scaling, so each replicate's
        # coefficients are on the same footing as the fit they are
        # bracketing rather than on their own.
        bs[i], _, _ = fit(rep, True, centre, scale)
        br[i], _, _ = fit(rep, False, centre, scale)
    keep = ~np.isnan(bs).any(axis=1)
    bs, br = bs[keep], br[keep]
    if len(bs) < args.n_boot * 0.5:
        print(f"\n  {args.n_boot - len(bs):,} replicates were rank "
              f"deficient and dropped.")

    print(f"\n{'term':12}{'standardised':>24}{'raw':>22}")
    results = {}
    for j, name in enumerate(TERMS):
        lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
        rlo, rhi = np.percentile(br[:, j], [2.5, 97.5])
        excl = bool(lo > 0 or hi < 0)
        results[name] = {
            "standardised": {"estimate": round(float(beta_s[j]), 4),
                             "lo": round(float(lo), 4),
                             "hi": round(float(hi), 4),
                             "excludes_zero": excl},
            "raw": {"estimate": round(float(beta_r[j]), 6),
                    "lo": round(float(rlo), 6),
                    "hi": round(float(rhi), 6)}}
        print(f"{name:12}{beta_s[j]:>+9.3f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>14}"
              f"{beta_r[j]:>+10.4f} "
              f"{f'[{rlo:+.4f}, {rhi:+.4f}]':>11}"
              + ("  *" if excl else ""))

    sep = [t for t in TERMS[1:] if results[t]["standardised"]
           ["excludes_zero"]]
    # The interaction comes first. With a non-zero interaction the
    # main effects are slopes at the other axis's mean rather than
    # independent effects, and a branch keyed on the main effects
    # would turn a moderation result into an independence story.
    if "S_f x S_p" in sep:
        verdict = ("the two axes interact: how strongly one predicts "
                   "decodability depends\n  on the level of the "
                   "other. The main effects below are slopes at the "
                   "other\n  axis's mean, not independent effects")
    elif not sep:
        verdict = ("neither axis predicts decodability as a slope on "
                   "this sample")
    elif sep == ["S_p"]:
        verdict = ("pocket familiarity predicts decodability as a "
                   "slope and family support\n  does not, which is "
                   "the cell result without a threshold")
    elif "S_f" in sep and "S_p" in sep:
        verdict = ("both axes predict decodability. The family axis "
                   "reads null when split\n  in two and not when "
                   "left continuous, which is what a gradient inside "
                   "one\n  cell looks like")
    elif sep == ["S_f"]:
        verdict = ("family support predicts decodability as a slope "
                   "and pocket familiarity\n  does not, which "
                   "reverses the cell result")
    else:
        verdict = f"the terms that separate are {', '.join(sep)}"
    print(f"\n  {verdict}.")

    results["_meta"] = {
        "feature": args.feature, "layer": selected_layer,
        "model": "auprc ~ log10(1+S_f) * S_p, both axes centred; "
                 "the standardised fit is also scaled",
        "n_fitted": int(len(d)), "n_dropped": int(len(dropped)),
        "dropped_reason": "receptor only partly resolved, so support "
                          "is a lower bound rather than a count",
        "S_p_at_or_above_99": round(at_ceiling, 4),
        "n_boot": args.n_boot, "seed": args.seed,
        "n_replicates_used": int(len(bs)),
        "resampled": "family_cluster"}
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print(f"\n  AUPRC is bounded in [0, 1] and this model is not, so "
          f"the fit is an\n  approximation. {at_ceiling:.0%} of "
          f"targets sit at or above 99 on the pocket\n  measure, "
          f"which compresses the top of its range and limits what any"
          f"\n  continuous model can resolve there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
