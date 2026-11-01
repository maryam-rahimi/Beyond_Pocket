#!/usr/bin/env python3
"""
Ask whether alignment depth accounts for the gap between familiarity
regimes.

WHY THIS QUESTION
-----------------
The regime gap is present at layer 0 and the 48 Pairformer blocks
roughly preserve it rather than creating it. Layer 0's input is not raw
sequence: Boltz-1 runs its MSA module first, so what enters the trunk
already carries processed evolutionary information. A disparity visible
at the first block therefore originated before the trunk, in the
alignment-derived features or the input embedding.

That makes alignment quality the natural suspect. If pockets with low
S_p sit in proteins with weaker coevolutionary signal, the gap would
appear exactly where it does, and the explanation would be about the
alignments rather than about pocket novelty.

The comparison that carries the weight is FN against NF. Those two
cells are closely matched on length, pocket size and positive rate and
differ by a factor of 2.6 in median AUPRC. If they also differ in
alignment depth, that is a candidate mechanism. If they do not, then
median msa_depth_min does not explain the gap -- which is narrower
than saying the alignments do not, since depth is one measure of them
and not the one the literature says matters most.

DEPTH IS A POOR PROXY, AND THE CEILING MAKES IT WORSE
-----------------------------------------------------
Two limits, both reported rather than worked around.

Feldman and Skolnick find AlphaFold 3 depends on phylogenetic
DIVERSITY rather than on alignment depth: a few divergent sequences
restore accuracy where many near-identical ones do not. Depth is
therefore a weak proxy for the quantity that matters, and a null result
here does not clear the alignments.

The depths were also recorded with a cap of 8192. A system at the
ceiling has an unknown true depth, and treating 8192 as a number
understates differences among well-aligned systems. This script reports the ceiling fraction per regime and gives the
correlation twice: over all systems, which is descriptive and
attenuated by the censoring, and over the uncapped systems, which is
the interpretable one. Read the second.

USAGE
-----
    python3 msa_covariate_check.py \\
        --index probe_index.parquet \\
        --per-system per_system_auprc.parquet \\
        --extraction-index reps_v3/index.csv \\
        --out msa_covariate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUADRANTS = ("FF", "FN", "NF", "NN")
CEILING = 8192


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, without a scipy dependency."""
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-system", required=True)
    ap.add_argument("--extraction-index", required=True,
                    help="index.csv from the extraction run, carrying "
                         "the per-system MSA depths")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    idx = pd.read_parquet(args.index)
    per = pd.read_parquet(args.per_system)
    dup = per.duplicated(["system_id", "feature_set", "layer"],
                         keep=False)
    if dup.any():
        ex = per.loc[dup, ["system_id", "feature_set",
                           "layer"]].head(3).to_dict("records")
        sys.exit(f"duplicate per-system score rows, e.g. {ex}")
    ext = pd.read_csv(args.extraction_index)
    for c in ("system_id", "msa_depth_min", "msa_depth_median",
              "msa_files"):
        if c not in ext.columns:
            sys.exit(f"{Path(args.extraction_index).name} has no "
                     f"'{c}'")

    df = (per[["system_id", "feature_set", "layer", "auprc"]]
          .merge(idx[["system_id", "quadrant"]], on="system_id",
                 how="inner", validate="many_to_one")
          .merge(ext[["system_id", "msa_depth_min",
                      "msa_depth_median", "msa_files"]],
                 on="system_id", how="inner", validate="many_to_one"))
    if df.empty:
        sys.exit("no rows after joining scores to the extraction index")

    results = {"ceiling": CEILING, "by_feature": {}}
    for feature, sub_all in df.groupby("feature_set"):
        # Several layers under one feature set would be reduced to
        # whichever happened to be first in row order, and the rest
        # silently dropped.
        layers = sorted(sub_all["layer"].unique())
        if len(layers) != 1:
            sys.exit(f"{feature} carries several scored layers "
                     f"{layers}; this check needs exactly one per "
                     f"feature set.")
        layer = int(layers[0])
        sub = sub_all
        at_ceiling = sub.msa_depth_min >= CEILING
        print(f"\n{feature} (layer {layer}), {len(sub)} proteins, "
              f"{int(at_ceiling.sum())} at the {CEILING} ceiling")
        print(f"{'cell':6}{'n':>5}{'at ceil':>9}{'min depth':>11}"
              f"{'med depth':>11}{'AUPRC':>8}")

        cells = {}
        for q in QUADRANTS:
            g = sub[sub.quadrant == q]
            if g.empty:
                continue
            ceil_frac = float((g.msa_depth_min >= CEILING).mean())
            cells[q] = {
                "n": int(len(g)),
                "frac_at_ceiling": round(ceil_frac, 3),
                "median_depth_min": float(g.msa_depth_min.median()),
                "median_depth_median":
                    float(g.msa_depth_median.median()),
                "median_auprc": round(float(g.auprc.median()), 4),
            }
            print(f"{q:6}{len(g):>5}{ceil_frac:>9.2f}"
                  f"{g.msa_depth_min.median():>11.0f}"
                  f"{g.msa_depth_median.median():>11.0f}"
                  f"{g.auprc.median():>8.3f}")

        # Correlation on the uncapped systems only. A system at the
        # ceiling has an unknown true depth, so including it as 8192
        # would compress the top of the range and bias the estimate
        # toward zero.
        free = sub[~at_ceiling]
        rho_all = spearman(sub.msa_depth_min.to_numpy(),
                           sub.auprc.to_numpy())
        rho_free = spearman(free.msa_depth_min.to_numpy(),
                            free.auprc.to_numpy())
        print(f"\n  Spearman(depth_min, AUPRC): {rho_all:+.3f} over "
              f"all {len(sub)}, {rho_free:+.3f} over the "
              f"{len(free)} uncapped")

        # The matched comparison. FN and NF agree on length, pocket
        # size and positive rate; if they also agree on depth, then
        # median msa_depth_min does not explain their gap. Other
        # properties of the alignments -- effective sequence count,
        # phylogenetic diversity, coverage heterogeneity -- are
        # untested here.
        fn, nf = sub[sub.quadrant == "FN"], sub[sub.quadrant == "NF"]
        matched = None
        if len(fn) and len(nf):
            matched = {
                "fn_median_depth_min": float(fn.msa_depth_min.median()),
                "nf_median_depth_min": float(nf.msa_depth_min.median()),
                "fn_median_auprc": round(float(fn.auprc.median()), 4),
                "nf_median_auprc": round(float(nf.auprc.median()), 4),
                "fn_frac_at_ceiling":
                    round(float((fn.msa_depth_min >= CEILING).mean()), 3),
                "nf_frac_at_ceiling":
                    round(float((nf.msa_depth_min >= CEILING).mean()), 3),
            }
            print(f"\n  FN vs NF, matched on geometry:")
            print(f"    depth_min  {matched['fn_median_depth_min']:.0f} "
                  f"vs {matched['nf_median_depth_min']:.0f}")
            print(f"    AUPRC      {matched['fn_median_auprc']:.3f} "
                  f"vs {matched['nf_median_auprc']:.3f}")

        results["by_feature"][feature] = {
            "layer": layer, "n": int(len(sub)),
            "n_at_ceiling": int(at_ceiling.sum()),
            "spearman_all": round(rho_all, 4),
            "spearman_uncapped": round(rho_free, 4),
            "n_uncapped": int(len(free)),
            "cells": cells, "fn_vs_nf": matched,
        }

    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print("\n  Depth is a weak proxy. Feldman and Skolnick report "
          "that AlphaFold 3\n  depends on phylogenetic diversity "
          "rather than depth, so a null result\n  here rules out "
          "median msa_depth_min as the explanation, not the "
          "alignments.")
    print(f"\n  The {CEILING} cap also censors the top of the range, "
          f"so differences among\n  well-aligned systems are "
          f"understated. The uncapped correlation is the\n  one to "
          f"read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
