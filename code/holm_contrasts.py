#!/usr/bin/env python3
"""
Attach Holm-corrected p-values to the eighteen regime contrasts.

WHY THIS AND WHY NOW
--------------------
The paper reports eighteen contrasts with percentile intervals and
says plainly that no multiplicity correction is applied, so an
interval excluding zero is an interval rather than a test at a stated
level. That disclosure is honest but it leaves the central claim --
that FN sits below each of the other three regimes -- resting on
uncorrected intervals, which is the first thing a reviewer will press
on.

Correcting them turns the claim into one that survives a stated
family-wise error rate, or shows that it does not.

THREE CHOICES, FIXED BEFORE THE RESULT
--------------------------------------
THE FAMILY IS ALL EIGHTEEN. The six pairwise regime comparisons, each
with its direction fixed in advance, in each of three feature sets. Not the three FN contrasts, and not six within each
feature set separately: the manuscript presents and interprets all
eighteen together, so narrowing the family after seeing which regime
is interesting would be selection by another name.

HOLM, NOT BENJAMINI-HOCHBERG. The claim is confirmatory rather than a
screen, and Holm controls the family-wise error rate without
assumptions about dependence among contrasts that share cells and are
therefore strongly correlated.

NULL-CENTRED BOOTSTRAP P-VALUES. The obvious quantity -- twice the
fraction of replicates on the far side of zero -- inverts a confidence
interval rather than testing a null, and it cannot go below 1/B no
matter how large the effect. The p-value here re-centres each
bootstrap distribution on the observed estimate, so the replicates
describe a world where the contrast is zero, and asks how often that
world produces something as extreme as what was seen:

    d_b = theta*_b - theta_hat
    p   = (1 + #{ |d_b| >= |theta_hat| }) / (B + 1)

REPRODUCING THE PUBLISHED RESAMPLING
------------------------------------
The published intervals came from one RNG stream consumed across
feature sets in pandas groupby order, so the draws for the second
feature set depend on the first having consumed its 2000 first. This
script replays that loop in the same order rather than resampling
afresh, and then CHECKS its own intervals against the published ones.
If they disagree the script stops: a p-value computed from a different
resampling than the interval beside it would be two analyses presented
as one.

USAGE
-----
    python3 holm_contrasts.py \\
        --index probe_index.parquet \\
        --per-system per_system_auprc.parquet \\
        --published bootstrap_contrasts.json \\
        --out holm_contrasts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUADRANTS = ("FF", "FN", "NF", "NN")
# Exactly as in bootstrap_contrasts.py, in the same order. The sign of
# each contrast was fixed in advance there and is not re-chosen here.
CONTRASTS = (("FF", "FN"), ("NF", "NN"), ("FF", "NN"), ("NF", "FF"),
             ("NN", "FN"), ("NF", "FN"))
TOL = 5e-4          # the published values are rounded to 4 decimals


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved.

    Sorted ascending, each multiplied by the number of hypotheses
    remaining, then made monotone so a later p-value is never smaller
    than an earlier one.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * pvals[i]
        running = max(running, v)
        adj[i] = min(running, 1.0)
    return adj


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-system", required=True)
    ap.add_argument("--published", required=True,
                    help="bootstrap_contrasts.json, whose intervals "
                         "this run must reproduce")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    pub = json.loads(Path(args.published).read_text())
    for k in ("n_boot", "seed", "alpha", "by_feature"):
        if k not in pub:
            sys.exit(f"{Path(args.published).name} has no '{k}'")
    n_boot, seed = int(pub["n_boot"]), int(pub["seed"])
    pub_alpha = float(pub["alpha"])
    # Survival is declared at the same level the intervals were built
    # at, so a run cannot pair 95% intervals with a 10% decision.
    if abs(args.alpha - pub_alpha) > 1e-12:
        sys.exit(f"--alpha is {args.alpha} but the published "
                 f"intervals were built at {pub_alpha}.")
    print(f"replaying the published resampling: {n_boot} replicates, "
          f"seed {seed}, alpha {pub_alpha}")

    idx = pd.read_parquet(args.index)
    per = pd.read_parquet(args.per_system)
    # The same guard the published run used: a duplicated score row
    # would weight one protein twice in every replicate.
    dup = per.duplicated(["system_id", "feature_set", "layer"],
                         keep=False)
    if dup.any():
        ex = per.loc[dup, ["system_id", "feature_set",
                           "layer"]].head(3).to_dict("records")
        sys.exit(f"duplicate per-system score rows, e.g. {ex}")
    for c in ("system_id", "quadrant", "family_cluster", "split"):
        if c not in idx.columns:
            sys.exit(f"--index has no '{c}'")

    # Checked separately from the merge, and before it. Replaying the
    # published JSON proves fidelity to the published run; it does not
    # prove that run used only held-out proteins. The merge below is
    # left unfiltered so the resampling reproduces exactly, so this is
    # the only place that assumption is tested.
    score_ids = set(per.system_id.unique())
    seen = idx[idx.system_id.isin(score_ids)][
        ["system_id", "split"]].drop_duplicates()
    missing = score_ids - set(seen.system_id)
    if missing:
        sys.exit(f"{len(missing)} scored systems are absent from "
                 f"--index, e.g. {sorted(missing)[:3]}. Refusing to "
                 f"reproduce an incompletely joined analysis.")
    non_test = seen[seen.split != "test"]
    if not non_test.empty:
        sys.exit(f"{len(non_test)} scored systems are not in the "
                 f"test split, e.g. "
                 f"{non_test.head(3).to_dict('records')}. The "
                 f"published contrasts are over held-out proteins.")
    print(f"all {len(score_ids):,} scored systems are test systems")
    # Column selection exactly as the published run did it, and
    # unfiltered for the same reason: filtering to the test split
    # here would change which clusters exist and so which are drawn,
    # and the replay would no longer reproduce the published
    # intervals. The check above is what makes the unfiltered merge
    # safe. The score file carries its own quadrant column, so
    # merging the whole frame would suffix both and leave no plain
    # "quadrant" to group on.
    df = per[["system_id", "feature_set", "layer", "auprc"]].merge(
        idx[["system_id", "quadrant", "family_cluster"]],
        on="system_id", how="inner", validate="many_to_one")
    if df.empty:
        sys.exit("no rows after joining scores to the index")
    if df.family_cluster.isna().any():
        bad = df.loc[df.family_cluster.isna(),
                     "system_id"].unique()[:5].tolist()
        sys.exit(f"family_cluster is missing for scored systems, "
                 f"e.g. {bad}.\nThe cluster bootstrap is undefined "
                 f"for them.")

    # One stream, consumed in groupby order, exactly as published.
    rng = np.random.default_rng(seed)
    lo_q, hi_q = 100 * pub_alpha / 2, 100 * (1 - pub_alpha / 2)
    rows, mismatches = [], []

    for feature, sub in df.groupby("feature_set"):
        layers = sorted(sub.layer.unique())
        if len(layers) != 1:
            sys.exit(f"{feature} carries several scored layers "
                     f"{layers}.")
        layer = int(layers[0])
        clusters = sub.family_cluster.unique()
        by_cluster = {c: g for c, g in sub.groupby("family_cluster")}
        n_clusters = len(clusters)

        boot_med = {q: np.empty(n_boot) for q in QUADRANTS}
        for b in range(n_boot):
            drawn = rng.choice(clusters, size=n_clusters,
                               replace=True)
            rep = pd.concat([by_cluster[c] for c in drawn],
                            ignore_index=True)
            g = rep.groupby("quadrant").auprc.median()
            for q in QUADRANTS:
                boot_med[q][b] = g.get(q, np.nan)

        # Matching numbers is strong evidence of the same analysis;
        # matching the counts behind them makes it fail closed. A
        # replay on different proteins that happened to land within
        # tolerance would otherwise pass.
        pub_f = pub["by_feature"].get(feature)
        if pub_f is None:
            sys.exit(f"{feature} is absent from the published file")
        if layer != pub_f["layer"]:
            sys.exit(f"{feature}: replaying layer {layer}, published "
                     f"layer {pub_f['layer']}")
        if n_clusters != pub_f["n_clusters"]:
            sys.exit(f"{feature}: {n_clusters} clusters now, "
                     f"{pub_f['n_clusters']} published")
        for q in QUADRANTS:
            n_now = int((sub.quadrant == q).sum())
            n_pub = pub_f["cells"].get(q, {}).get("n")
            if n_pub is not None and n_now != n_pub:
                sys.exit(f"{feature}/{q}: {n_now} proteins now, "
                         f"{n_pub} published")

        observed = {q: float(sub[sub.quadrant == q].auprc.median())
                    for q in QUADRANTS
                    if not sub[sub.quadrant == q].empty}
        pub_c = pub_f.get("contrasts", {})

        for a, b_ in CONTRASTS:
            if a not in observed or b_ not in observed:
                continue
            name = f"{a}-{b_}"
            est = observed[a] - observed[b_]
            d = boot_med[a] - boot_med[b_]
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [lo_q, hi_q])

            # The replayed interval must match the published one, or
            # the p-value below belongs to a different resampling
            # than the interval it will sit beside.
            p_ref = pub_c.get(name)
            if p_ref is None:
                mismatches.append(f"{feature} {name}: absent from "
                                  f"the published file")
            else:
                for label, mine, theirs in (("estimate", est,
                                             p_ref["estimate"]),
                                            ("lo", lo, p_ref["lo"]),
                                            ("hi", hi, p_ref["hi"])):
                    if abs(mine - theirs) > TOL:
                        mismatches.append(
                            f"{feature} {name} {label}: replayed "
                            f"{mine:.4f}, published {theirs:.4f}")

            # Null-centred two-sided p-value.
            centred = d - est
            p = (1 + int(np.sum(np.abs(centred) >= abs(est)))) \
                / (len(d) + 1)
            rows.append({"feature": feature, "layer": layer,
                         "contrast": name, "estimate": est,
                         "lo": float(lo), "hi": float(hi),
                         "p_raw": p, "n_draws": int(len(d))})

    if mismatches:
        sys.exit("the replayed resampling does not reproduce the "
                 "published intervals:\n  "
                 + "\n  ".join(mismatches[:8])
                 + f"\n\n{len(mismatches)} discrepancies. A p-value "
                   f"from a different resampling than\nits interval "
                   f"would be two analyses presented as one.")
    print(f"replay matches the published intervals for all "
          f"{len(rows)} contrasts")

    # Fatal, not a warning. The family was fixed at eighteen before
    # the result; applying Holm to seventeen would be a different
    # test from the pre-specified one, and the difference would not
    # be visible in the output.
    if len(rows) != 18:
        have = sorted(f"{r['feature']}/{r['contrast']}" for r in rows)
        sys.exit(f"the pre-specified multiplicity family is eighteen "
                 f"contrasts; {len(rows)} are present.\nRefusing to "
                 f"apply Holm to a different family. Found: "
                 f"{have}")

    p_adj = holm(np.array([r["p_raw"] for r in rows]))
    for r, pa in zip(rows, p_adj):
        r["p_holm"] = float(pa)
        r["survives"] = bool(pa <= args.alpha)

    print(f"\n{'feature':8}{'contrast':10}{'estimate':>10}"
          f"{'p raw':>9}{'p Holm':>9}  survives")
    for r in rows:
        print(f"{r['feature']:8}{r['contrast']:10}"
              f"{r['estimate']:>+10.3f}{r['p_raw']:>9.4f}"
              f"{r['p_holm']:>9.4f}"
              f"{'  yes' if r['survives'] else '  no'}")

    fn = [r for r in rows if "FN" in r["contrast"]]
    fn_survive = [r for r in fn if r["survives"]]
    others = [r for r in rows if "FN" not in r["contrast"]]
    print(f"\n{len(fn_survive)} of {len(fn)} contrasts involving FN "
          f"survive Holm at {args.alpha}")
    print(f"{sum(r['survives'] for r in others)} of {len(others)} "
          f"others survive")

    # The floor a bootstrap p-value cannot go below, which bounds what
    # any correction can achieve.
    floor = 1 / (n_boot + 1)
    at_floor = [r["contrast"] for r in rows
                if abs(r["p_raw"] - floor) < 1e-12]
    if at_floor:
        print(f"\n  {len(at_floor)} contrasts sit at the bootstrap "
              f"floor of {floor:.2e}, which is\n  the smallest "
              f"p-value {n_boot} replicates can produce. Their true "
              f"p-values\n  may be smaller; the correction is "
              f"conservative for them.")

    res = {"family_size": len(rows),
           "family": "all pairwise regime contrasts across all "
                     "feature sets; not narrowed after seeing which "
                     "regime separates",
           "correction": "Holm-Bonferroni",
           "p_value": "null-centred two-sided cluster bootstrap, "
                      "p = (1 + #{|theta*_b - theta_hat| >= "
                      "|theta_hat|}) / (B + 1)",
           "alpha": args.alpha, "n_boot": n_boot, "seed": seed,
           "bootstrap_p_floor": floor,
           "replay_verified_against": str(Path(args.published).name),
           "n_fn_contrasts": len(fn),
           "n_fn_surviving": len(fn_survive),
           "contrasts": rows}
    out.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out}")
    print("\n  Holm over all eighteen, with the family fixed before "
          "the result. A p-value\n  here tests a null; the intervals "
          "beside it remain percentile intervals on\n  the same "
          "resampling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
