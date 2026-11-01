#!/usr/bin/env python3
"""
Plot the depth curves, one panel per feature set, four regimes
overlaid.

WHY THIS LAYOUT
---------------
The comparison the paper makes is between regimes, so the regimes are
overlaid within a panel and the feature sets are separated across
panels. The alternative -- a panel per regime with feature sets
overlaid -- would put the paper's own contrast across panels, where a
reader has to carry numbers between axes to see it.

`s` comes first because it is the pre-registered primary feature set.

WHAT IS AND IS NOT DRAWN
------------------------
Drawn: the median AUPRC across test proteins at each layer, and the
random-ranking baseline as a dashed line. Shaded bands are the
interquartile range across proteins, which is wide -- a factor of
nearly three at the plateau -- and should not be mistaken for a
confidence interval on the median. The caption says so.

NOT drawn: the null threshold and the onset. Neither exists yet. A
figure with an unlabelled horizontal line where tau belongs invites
the reader to assume a calibration that has not been done, so the
space is left empty until it has.

USAGE
-----
    python3 plot_depth_curves.py \\
        --curves depth_curves.parquet \\
        --out figures/depth_curves.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"needs matplotlib: {e}")

QUADRANTS = ("FF", "FN", "NF", "NN")
LABELS = {"FF": "FF  familiar family, familiar pocket",
          "FN": "FN  familiar family, novel pocket",
          "NF": "NF  novel family, familiar pocket",
          "NN": "NN  novel family, novel pocket"}
# Colour by POCKET familiarity, which is the axis that separates the
# cells: the two familiar-pocket regimes share a hue and the two novel
# ones share another, so the finding is visible before the legend is
# read. Line style distinguishes the family axis within each pair.
STYLE = {"FF": ("#1D9E75", "-"), "NF": ("#0F6E56", "--"),
         "FN": ("#D85A30", "-"), "NN": ("#993C1D", "--")}
ORDER = ("s", "zpool", "both")
TITLES = {"s": "per-residue track (s)",
          "zpool": "pooled pair track (z)",
          "both": "concatenated"}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curves", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-iqr", action="store_true",
                    help="omit the interquartile bands")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.curves)
    need = {"feature_set", "layer", "quadrant", "auprc_median",
            "auprc_q25", "auprc_q75", "baseline_median"}
    if not need.issubset(df.columns):
        sys.exit(f"curve file lacks {sorted(need - set(df.columns))}")

    features = [f for f in ORDER if f in set(df.feature_set)]
    if not features:
        sys.exit("no recognised feature sets in the curve file")

    fig, axes = plt.subplots(1, len(features),
                             figsize=(4.6 * len(features), 4.0),
                             sharey=True)
    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        sub = df[df.feature_set == feature]
        for q in QUADRANTS:
            g = sub[sub.quadrant == q].sort_values("layer")
            if g.empty:
                continue
            colour, dash = STYLE[q]
            if not args.no_iqr:
                ax.fill_between(g.layer, g.auprc_q25, g.auprc_q75,
                                color=colour, alpha=0.10, linewidth=0)
            ax.plot(g.layer, g.auprc_median, color=colour,
                    linestyle=dash, linewidth=1.8, label=LABELS[q])
        # The random-ranking baseline is each regime's positive rate,
        # and those differ: roughly 0.079 for FF against 0.065 for NN,
        # a fifth of their own size. One line at the median would set a
        # single bar that is too high for the low-prevalence regimes
        # and too low for FF, flattering the cells the paper compares.
        bases = (sub.groupby("quadrant")["baseline_median"]
                 .median().dropna())
        lo, hi = float(bases.min()), float(bases.max())
        if hi - lo > 1e-6:
            # A band, not four lines: the point is that there is no one
            # baseline, and four dotted lines would clutter the panel
            # without being individually readable.
            ax.axhspan(lo, hi, color="#5F5E5A", alpha=0.12, linewidth=0)
            ax.annotate(f"random ranking ({lo:.3f}-{hi:.3f})",
                        xy=(1, hi), xytext=(1, hi + 0.015),
                        fontsize=7.5, color="#5F5E5A")
        else:
            ax.axhline(lo, color="#5F5E5A", linestyle=":",
                       linewidth=1.0)
            ax.annotate(f"random ranking ({lo:.3f})",
                        xy=(1, lo), xytext=(1, lo + 0.015),
                        fontsize=7.5, color="#5F5E5A")
        ax.set_title(TITLES.get(feature, feature), fontsize=10)
        ax.set_xlabel("Pairformer block")
        ax.set_xlim(0, int(sub.layer.max()))
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("AUPRC, median across test proteins")
    axes[0].set_ylim(0, None)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")

    first = df[df.feature_set == features[0]]
    per_layer = first.groupby("layer")["n_systems"].sum()
    if per_layer.nunique() != 1:
        print(f"\n  WARNING: the number of test proteins varies across "
              f"layers\n  (min {per_layer.min()}, max "
              f"{per_layer.max()}); the curve is not scoring one fixed "
              f"set.")
    n = int(per_layer.median())
    print(f"\n  {len(features)} panels, {n} test proteins per layer.")
    print("  Bands are the interquartile range ACROSS PROTEINS, not a\n"
          "  confidence interval on the median. Say so in the caption:\n"
          "  the spread is wide and a reader will otherwise read them\n"
          "  as uncertainty on the curve.")
    print("\n  No null threshold and no onset are drawn, because "
          "neither has\n  been computed. Add them only once the "
          "calibration exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
