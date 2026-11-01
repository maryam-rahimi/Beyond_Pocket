#!/usr/bin/env python3
"""
Ask how much of the per-protein AUPRC is explained by design factors
that carry no pocket information.

WHY
---
AUPRC on a per-residue label is sensitive to the shape of the
evaluation problem, not only to what the representation encodes. A
protein with a large pocket relative to its length is easier to score
well on than one with a small pocket in a long chain, whatever the
features contain. If the regimes differ in those quantities, a gap
between them can be a property of the evaluation geometry wearing a
novelty costume.

This is the check the CRISPRi study of Shoeibi and Yousefi (arXiv
2608.00152) applies to its own endpoint, where a sample-count-only
model reached R^2 = 0.43 against 0.26 for the four magnitude scalars,
and the scalars added 0.0017 on top. Most of the apparent signal
overlapped with a design factor.

WHAT IS TESTED
--------------
Three nested models of per-protein AUPRC at a chosen layer:

  design   length, pocket size, positive rate, number of pockets
  regime   the four familiarity cells as indicators
  both     design + regime

and the comparison that matters:

  does regime explain variance BEYOND the design factors?

If adding the regime indicators to the design model improves R^2 only
slightly, the regime gap is largely a restatement of differences in
protein length or pocket size. If the improvement is substantial, the
regimes differ for reasons the design factors do not capture.

The reverse is also reported: how much the design factors add beyond
regime. A design model that explains most of the variance on its own
is a warning regardless of what regime adds.

WHAT THIS DOES NOT ESTABLISH
----------------------------
A regime effect surviving these four covariates is not thereby shown to
be about familiarity. Other properties of a pocket -- depth, polarity,
solvent exposure, whether it is an enzyme active site -- are
uncontrolled and could track S_p. This narrows the set of alternative
explanations; it does not empty it.

USAGE
-----
    python3 design_factor_check.py \\
        --index probe_index.parquet \\
        --per-system per_system_auprc.parquet \\
        --out design_factors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUADRANTS = ("FF", "FN", "NF", "NN")
DESIGN_COLS = ("length", "n_pocket", "positive_rate")


def r2_ols(X: np.ndarray, y: np.ndarray) -> float:
    """In-sample R^2 of an ordinary least squares fit with intercept."""
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-system", required=True,
                    help="per-protein AUPRC, one row per (system, "
                         "feature set, layer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int,
                    help="layer to test; default is each feature set's "
                         "own peak layer")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    idx = pd.read_parquet(args.index)
    per = pd.read_parquet(args.per_system)
    need = {"system_id", "feature_set", "layer", "auprc"}
    if not need.issubset(per.columns):
        sys.exit(f"--per-system lacks {sorted(need - set(per.columns))}")

    # The index is authoritative for design metadata. The per-system
    # file already carries quadrant, and the one written by
    # run_depth_probes.py also carries positive_rate, so merging both
    # sides would produce quadrant_x / quadrant_y and the references
    # below would fail. Take only the score columns from the left.
    per = per[["system_id", "feature_set", "layer", "auprc"]].copy()
    idx_need = {"system_id", "quadrant", *DESIGN_COLS}
    missing = idx_need - set(idx.columns)
    if missing:
        sys.exit(f"--index lacks {sorted(missing)}")
    df = per.merge(idx[["system_id", "quadrant", *DESIGN_COLS]],
                   on="system_id", how="inner", validate="many_to_one")
    if df.empty:
        sys.exit("no rows after joining per-system scores to the index")

    results = {}
    print(f"{'feature':8}{'layer':>7}{'n':>6}{'design':>9}{'regime':>9}"
          f"{'both':>8}{'regime adds':>13}{'design adds':>13}")
    for feature, sub_all in df.groupby("feature_set"):
        if args.layer is not None:
            layer = args.layer
        else:
            # Each feature set's own peak, since that is where the
            # reported contrast lives.
            layer = int(sub_all.groupby("layer").auprc.median().idxmax())
        sub = sub_all[sub_all.layer == layer]
        if len(sub) < 50:
            print(f"  skipping {feature} layer {layer}: only {len(sub)} "
                  f"systems")
            continue

        y = sub.auprc.to_numpy(dtype=float)
        design = sub[list(DESIGN_COLS)].to_numpy(dtype=float)
        # Indicators for three of the four cells; the fourth is the
        # intercept, so the design matrix stays full rank.
        regime = np.column_stack([(sub.quadrant == q).to_numpy(float)
                                  for q in QUADRANTS[1:]])

        r_design = r2_ols(design, y)
        r_regime = r2_ols(regime, y)
        r_both = r2_ols(np.column_stack([design, regime]), y)
        print(f"{feature:8}{layer:>7}{len(sub):>6}{r_design:>9.3f}"
              f"{r_regime:>9.3f}{r_both:>8.3f}"
              f"{r_both - r_design:>13.3f}{r_both - r_regime:>13.3f}")

        # Per-regime means of each design factor, so a difference that
        # drives the gap is visible rather than inferred.
        by_q = {}
        for q in QUADRANTS:
            g = sub[sub.quadrant == q]
            if g.empty:
                continue
            by_q[q] = {"n": int(len(g)),
                       "auprc_median": round(float(g.auprc.median()), 4),
                       **{c: round(float(g[c].median()), 4)
                          for c in DESIGN_COLS}}

        results[feature] = {
            "layer": layer, "n_systems": int(len(sub)),
            "r2_design_only": round(r_design, 4),
            "r2_regime_only": round(r_regime, 4),
            "r2_both": round(r_both, 4),
            "regime_adds_over_design": round(r_both - r_design, 4),
            "design_adds_over_regime": round(r_both - r_regime, 4),
            "by_quadrant": by_q,
        }

    if not results:
        sys.exit("nothing tested")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print("\n  'regime adds' is the quantity of interest: how much the\n"
          "  familiarity cells explain BEYOND length, pocket size and\n"
          "  positive rate. A small value means the regime gap is\n"
          "  largely a restatement of those.\n")
    print("  A large 'design only' R^2 is a warning on its own, even\n"
          "  when regime adds a lot: it means most of the variation in\n"
          "  per-protein AUPRC is evaluation geometry rather than\n"
          "  anything the representation encodes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
