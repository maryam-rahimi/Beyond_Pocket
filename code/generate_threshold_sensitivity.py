#!/usr/bin/env python3
"""
Plot and tabulate how the four regimes move as the pocket-familiarity
threshold changes, with the metric held fixed.

WHY
---
The cells exist because a line was drawn at a particular value of
pocket coverage. A finding that appears only near that value is a
finding about the line. Holding the metric fixed and moving only the
threshold separates the two, which an alternative metric at an
alternative threshold cannot: that varies both at once.

WHAT IT SHOWS
-------------
One panel per feature set, one line per regime, median AUPRC against
threshold. A regime that stays lowest across the range is lowest
wherever the line is drawn; an ordering that appears at one value and
not at others belongs to that value.

Cell sizes vary enormously across the range -- a strict threshold
leaves few familiar-pocket systems and a permissive one leaves few
novel-pocket ones -- so each point is annotated with its n and points
below a floor are drawn hollow and excluded from the reported
ordering. A median over twenty systems is not evidence of the same
kind as a median over two hundred.

WHAT IT CANNOT SHOW
-------------------
These are regroupings of scores from a probe fitted once, on a split
built at the primary threshold. The sample was balanced for that
threshold, so cells away from it are unbalanced by construction. This
is a sensitivity analysis of the assignment, not a re-run of the study
at each threshold.

USAGE
-----
    python3 generate_threshold_sensitivity.py \\
        --primary alternative_sp_qcov50.json \\
        --primary-threshold 80 \\
        --others alternative_sp_qcov50.json \\
                 alternative_sp_qcov100.json \\
        --outdir threshold_tables/

The same file can appear as --primary and among --others: its
original columns give the stored assignment, its new columns the
regrouping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUADRANTS = ("FF", "FN", "NF", "NN")
FEATURE_ORDER = ("s", "zpool", "both")
FEATURE_TEX = {"s": r"\texttt{s}", "zpool": r"\texttt{z}",
               "both": "concat."}
FEATURE_TITLE = {"s": "per-residue track", "zpool": "pooled pair track",
                 "both": "concatenated"}
# Below this a cell median is reported but not used to establish an
# ordering. Fixed here rather than passed in.
MIN_CELL = 30
STYLE = {"FF": ("#1D9E75", "-"), "NF": ("#0F6E56", "--"),
         "FN": ("#D85A30", "-"), "NN": ("#993C1D", "--")}


def load(path: Path, primary: bool):
    """One threshold's cells, from either side of the alt-sp output."""
    d = json.loads(path.read_text())
    for k in ("threshold", "metric", "by_feature"):
        if k not in d:
            sys.exit(f"{path.name} has no '{k}'")
    out = {}
    for f, blk in d["by_feature"].items():
        cells = {}
        for q, c in blk["cells"].items():
            # The primary file's own threshold is described by the
            # "original" columns; the others by the "new" ones.
            n = c["n_original"] if primary else c["n_new"]
            m = c["median_original"] if primary else c["median_new"]
            cells[q] = {"n": n, "median": m}
        out[f] = cells
    return float(d["threshold"]), d["metric"], out


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primary", required=True,
                    help="any alt-sp run; its 'original' columns hold "
                         "the stored assignment and supply the "
                         "primary point. Which file is used does not "
                         "matter, since every run carries the same "
                         "original columns.")
    ap.add_argument("--primary-threshold", type=float, required=True,
                    help="the threshold the STORED assignment was "
                         "built at. Required and not inferred: a "
                         "file's own threshold field describes its "
                         "alternative regrouping, not its original "
                         "columns, so reading it would plot the "
                         "stored cells at the wrong place on the "
                         "axis -- wrong, and entirely plausible "
                         "looking.")
    ap.add_argument("--others", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    # The returned threshold is deliberately discarded: it is the
    # file's ALTERNATIVE threshold, while the cells taken here are its
    # original ones.
    _, metric0, cells0 = load(Path(args.primary), primary=True)
    thr0 = args.primary_threshold
    points = {thr0: cells0}
    for p in args.others:
        t, m, c = load(Path(p), primary=False)
        if m != metric0:
            sys.exit(f"{Path(p).name} uses metric '{m}', the primary "
                     f"uses '{metric0}'.\nMoving the metric as well as "
                     f"the threshold is a different comparison.")
        if t == thr0:
            sys.exit(f"{Path(p).name} regroups at {t:g}, which is the "
                     f"primary threshold.\nIts 'new' columns would "
                     f"duplicate the primary point rather than add to "
                     f"it.")
        if t in points:
            sys.exit(f"threshold {t:g} appears twice")
        points[t] = c
    thrs = sorted(points)
    feats = [f for f in FEATURE_ORDER if f in cells0]
    print(f"metric {metric0}, thresholds {[f'{t:g}' for t in thrs]}")

    # Which regime is lowest at each threshold, ignoring cells too
    # small to support the claim.
    print(f"\nLOWEST REGIME, cells under {MIN_CELL} systems excluded")
    print(f"{'features':10}" + "".join(f"{t:>12g}" for t in thrs))
    lowest = {}
    for f in feats:
        row, cells_row = [], []
        for t in thrs:
            usable = {q: v["median"] for q, v in points[t][f].items()
                      if v["n"] >= MIN_CELL and v["median"] is not None}
            if not usable:
                row.append("--")
                cells_row.append(None)
                continue
            lo = min(usable, key=usable.get)
            row.append(f"{lo} {usable[lo]:.3f}")
            cells_row.append(lo)
        lowest[f] = cells_row
        print(f"{f:10}" + "".join(f"{x:>12}" for x in row))

    # Two conditions, and the first does not imply the second: every
    # feature set could be internally stable while disagreeing with
    # the others about which regime is lowest. Taking feats[0]'s
    # answer after testing only the first would state a cross-feature
    # claim on within-feature evidence.
    stable_regime = {}
    for f, vals in lowest.items():
        present = [v for v in vals if v is not None]
        stable_regime[f] = (present[0]
                            if len(present) == len(vals)
                            and len(set(present)) == 1
                            else None)
    regimes = [stable_regime[f] for f in feats]
    all_stable = (bool(regimes)
                  and all(r is not None for r in regimes)
                  and len(set(regimes)) == 1)
    print()
    if all_stable:
        q = regimes[0]
        print(f"  {q} is lowest at every threshold in every feature "
              f"set.")
    else:
        for f in feats:
            print(f"  {f}: {' -> '.join(str(x) for x in lowest[f])}")
        within = [f for f in feats if stable_regime[f] is not None]
        if len(within) == len(feats):
            print(f"  Each feature set is internally stable, but they "
                  f"do not agree on which\n  regime is lowest: "
                  f"{sorted(set(regimes))}.")
        else:
            print(f"  The lowest regime is not the same throughout, "
                  f"so the ordering depends\n  on where the line is "
                  f"drawn.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- the table ----------------------------------------------------
    # Three outcomes, not two. "The ordering depends on the
    # threshold" is a far stronger claim than "the feature sets
    # disagree", and a caption that reports the second as the first
    # would overstate the instability. The terminal output already
    # draws this distinction; the caption must use the same one.
    within_stable = {f: (len(set(v)) == 1 and v[0] is not None)
                     for f, v in lowest.items()}
    if all_stable:
        verdict = (rf"The same regime is lowest at every threshold in "
                   rf"every feature set.")
    elif all(within_stable.values()):
        names = ", ".join(f"{FEATURE_TEX[f]}: {lowest[f][0]}"
                          for f in feats)
        verdict = (rf"The identity of the lowest regime is stable "
                   rf"across thresholds within each feature set, but "
                   rf"differs between them ({names}).")
    else:
        unstable = [f for f in feats if not within_stable[f]]
        verdict = (rf"The identity of the lowest regime changes across "
                   rf"thresholds in "
                   rf"{'both' if len(unstable) == 2 else 'at least one'} "
                   rf"feature "
                   rf"{'sets' if len(unstable) > 1 else 'set'}, so the "
                   rf"ordering is sensitive to where the boundary is "
                   rf"placed.")
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Median AUPRC by regime as the pocket-familiarity "
        rf"threshold moves, with the metric held at "
        rf"\texttt{{{metric0.replace('_', chr(92) + '_')}}} and the "
        rf"aggregation unchanged. Cell sizes vary with the threshold "
        rf"by construction: a strict boundary leaves few "
        rf"familiar-pocket systems and a permissive one few "
        rf"novel-pocket ones. Medians over fewer than {MIN_CELL} "
        rf"systems are shown in parentheses and are not used to "
        rf"establish an ordering. {verdict} These are regroupings of "
        rf"one probe's scores, not a re-run of the study at each "
        rf"threshold.}}",
        r"\label{tab:threshold-sensitivity}",
        r"\begin{tabular}{ll" + "r" * len(thrs) + "}", r"\toprule",
        r"Features & Cell & " +
        " & ".join(rf"\(\tau={t:g}\)" for t in thrs) + r" \\",
        r"\midrule",
    ]
    for i, f in enumerate(feats):
        for j, q in enumerate(QUADRANTS):
            head = FEATURE_TEX[f] if j == 0 else ""
            vals = []
            for t in thrs:
                c = points[t][f].get(q)
                if c is None or c["median"] is None:
                    vals.append("--")
                elif c["n"] < MIN_CELL:
                    vals.append(f"({c['median']:.3f})")
                else:
                    vals.append(f"{c['median']:.3f}")
            lines.append(f"{head} & {q} & " + " & ".join(vals) +
                         r" \\")
        if i < len(feats) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (outdir / "threshold_sensitivity.tex").write_text(
        "\n".join(lines) + "\n")

    # --- the figure ---------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (no matplotlib; the figure is skipped)")
    else:
        fig, axes = plt.subplots(1, len(feats),
                                 figsize=(4.4 * len(feats), 3.9),
                                 sharey=True)
        if len(feats) == 1:
            axes = [axes]
        for ax, f in zip(axes, feats):
            for q in QUADRANTS:
                xs, ys, small = [], [], []
                for t in thrs:
                    c = points[t][f].get(q)
                    if c is None or c["median"] is None:
                        continue
                    xs.append(t)
                    ys.append(c["median"])
                    small.append(c["n"] < MIN_CELL)
                if not xs:
                    continue
                col, dash = STYLE[q]
                ax.plot(xs, ys, color=col, linestyle=dash,
                        linewidth=1.8, label=q, zorder=2)
                # Hollow markers where the cell is too small to carry
                # the claim, so the eye does not read them as equal
                # evidence.
                for x, y, s in zip(xs, ys, small):
                    ax.plot([x], [y], "o", color=col, markersize=5,
                            markerfacecolor=("none" if s else col),
                            zorder=3)
            ax.set_title(FEATURE_TITLE.get(f, f), fontsize=10)
            ax.set_xlabel("pocket-familiarity threshold")
            ax.set_xticks(thrs)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("median AUPRC")
        axes[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "threshold_sensitivity.pdf", dpi=300,
                    bbox_inches="tight")
        print(f"\nWrote {outdir / 'threshold_sensitivity.pdf'}")

    md = ["## Threshold sensitivity", "",
          f"Metric `{metric0}`, thresholds "
          f"{', '.join(f'{t:g}' for t in thrs)}.", "",
          "| features | cell | " +
          " | ".join(f"t={t:g}" for t in thrs) + " |",
          "|---|---|" + "---|" * len(thrs)]
    for f in feats:
        for q in QUADRANTS:
            vals = []
            for t in thrs:
                c = points[t][f].get(q)
                vals.append("--" if c is None or c["median"] is None
                            else f"{c['median']:.3f}"
                            f"{' *' if c['n'] < MIN_CELL else ''}")
            md.append(f"| {f} | {q} | " + " | ".join(vals) + " |")
    md += ["", f"`*` fewer than {MIN_CELL} systems."]
    (outdir / "threshold_sensitivity.md").write_text(
        "\n".join(md) + "\n")
    print(f"Wrote files to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
