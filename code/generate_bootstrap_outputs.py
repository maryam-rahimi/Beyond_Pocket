#!/usr/bin/env python3
"""
Table and figure for the bootstrap contrasts.

  bootstrap_cells.tex      per-regime medians with intervals
  bootstrap_contrasts.tex  the six contrasts, with which intervals
                           exclude zero
  bootstrap_contrasts.pdf  a forest plot of the same
  bootstrap_summary.md     both tables, for progress notes

WHAT THE CAPTIONS SAY, AND WHY THEY ARE DERIVED
-----------------------------------------------
Which contrasts exclude zero is read from the data, not written in
advance. A caption asserting that a particular comparison holds would
quietly become false if the analysis were re-run on other data, and
that failure is harder to notice than a wrong number because prose is
not checked against a table.

The captions also state what an interval here is not. Eighteen
contrasts are computed with no multiplicity correction, so an interval
excluding zero is an interval and not a test at a stated level. A
reader who takes the marked rows as significant findings at five per
cent would be reading more than the analysis provides.

THE PRE-REGISTERED CONTRAST IS MARKED
-------------------------------------
FF versus NN is the pre-registered primary comparison and is marked as
such wherever it appears, including when its interval contains zero. A
table that presented it as one row among six would let a null primary
endpoint pass unnoticed among intermediate contrasts that did
separate.

USAGE
-----
    python3 generate_bootstrap_outputs.py \\
        --bootstrap bootstrap_contrasts.json \\
        --outdir bootstrap_tables/
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
PRIMARY = "FF-NN"


def verdict(b: dict) -> dict:
    """What the intervals show, read from them."""
    feats = [f for f in FEATURE_ORDER if f in b["by_feature"]]
    if not feats:
        sys.exit(f"no recognised feature sets; by_feature holds "
                 f"{sorted(b['by_feature'])} and this expects "
                 f"{list(FEATURE_ORDER)}")

    # The primary endpoint is required explicitly, before anything
    # derived from the contrasts that happen to be present. names is
    # built from the union of what exists, so a primary contrast absent
    # from every feature set would never enter it and the completeness
    # check below would never see it -- in a script whose stated
    # purpose is keeping a null primary endpoint visible.
    for f in feats:
        if PRIMARY not in b["by_feature"][f].get("contrasts", {}):
            sys.exit(f"{f}: the pre-registered primary contrast "
                     f"{PRIMARY} is absent.\nRefusing to generate a "
                     f"paper-facing table without the primary "
                     f"endpoint.")

    names = set()
    for f in feats:
        names |= set(b["by_feature"][f]["contrasts"])

    # Every contrast must appear in every feature set before the
    # caption can say "in every feature set". Otherwise a contrast
    # absent from one and separating in the other two would be
    # described as holding everywhere, where in fact one feature set
    # is silent rather than supportive -- a weaker claim wearing the
    # words of a stronger one.
    for c in sorted(names):
        absent = [f for f in feats
                  if c not in b["by_feature"][f]["contrasts"]]
        if absent:
            sys.exit(f"contrast {c} is absent from {absent}, so "
                     f"consistency across feature sets\ncannot be "
                     f"summarised. Regenerate the bootstrap with the "
                     f"same contrasts throughout.")

    always, never = [], []
    for c in sorted(names):
        seen = [b["by_feature"][f]["contrasts"][c]["excludes_zero"]
                for f in feats]
        if all(seen):
            always.append(c)
        elif not any(seen):
            never.append(c)
    prim = [b["by_feature"][f]["contrasts"][PRIMARY]["excludes_zero"]
            for f in feats if PRIMARY in b["by_feature"][f]["contrasts"]]
    return {"feats": feats, "always": always, "never": never,
            "primary_excludes": (all(prim) if prim else None),
            "primary_any": (any(prim) if prim else None)}


def cells_table(b: dict, v: dict) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Median AUPRC per familiarity regime at each "
        rf"feature set's peak layer, with percentile intervals from "
        rf"{b['n_boot']} bootstrap replicates. The resampling unit is "
        rf"the receptor-family cluster, not the protein: proteins "
        rf"within a family share sequence and fold, and resampling "
        rf"them individually would count near-duplicates as "
        rf"independent and give intervals that are too narrow.}}",
        r"\label{tab:bootstrap-cells}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Features & Cell & $n$ & Median & "
        rf"{int(100 * (1 - b['alpha']))}\% interval \\", r"\midrule",
    ]
    for i, f in enumerate(v["feats"]):
        d = b["by_feature"][f]
        for j, q in enumerate(QUADRANTS):
            c = d["cells"].get(q)
            if c is None:
                continue
            head = FEATURE_TEX[f] if j == 0 else ""
            lines.append(f"{head} & {q} & {c['n']} & {c['median']:.3f} "
                         f"& [{c['lo']:.3f}, {c['hi']:.3f}] \\\\")
        if i < len(v["feats"]) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def contrasts_table(b: dict, v: dict) -> str:
    n_tot = sum(len(b["by_feature"][f]["contrasts"]) for f in v["feats"])
    bits = []
    if v["always"]:
        bits.append("Intervals exclude zero in every feature set for "
                    + ", ".join(c.replace("-", "--")
                                for c in v["always"]) + ".")
    if v["never"]:
        bits.append("They contain zero in every feature set for "
                    + ", ".join(c.replace("-", "--")
                                for c in v["never"]) + ".")
    if v["primary_excludes"] is False and not v["primary_any"]:
        bits.append(rf"The pre-registered primary contrast "
                    rf"{PRIMARY.replace('-', '--')} is among the "
                    rf"latter.")
    elif v["primary_excludes"]:
        bits.append(rf"The pre-registered primary contrast "
                    rf"{PRIMARY.replace('-', '--')} is among the "
                    rf"former.")

    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Bootstrap contrasts between familiarity regimes, "
        rf"each computed within the replicate rather than from two "
        rf"independently resampled estimates. "
        rf"{' '.join(bits)} {n_tot} contrasts are reported with no "
        rf"multiplicity correction, so an interval excluding zero is "
        rf"an interval and not a test at a stated level. $\dagger$ "
        rf"marks the pre-registered primary contrast.}}",
        r"\label{tab:bootstrap-contrasts}",
        r"\begin{tabular}{llrrc}", r"\toprule",
        r"Features & Contrast & Estimate & "
        rf"{int(100 * (1 - b['alpha']))}\% interval & Excludes 0 \\",
        r"\midrule",
    ]
    for i, f in enumerate(v["feats"]):
        d = b["by_feature"][f]["contrasts"]
        for j, (c, r) in enumerate(d.items()):
            head = FEATURE_TEX[f] if j == 0 else ""
            mark = r"$\dagger$" if c == PRIMARY else ""
            lines.append(
                f"{head} & {c.replace('-', '--')}{mark} & "
                f"{r['estimate']:+.3f} & "
                f"[{r['lo']:+.3f}, {r['hi']:+.3f}] & "
                f"{'yes' if r['excludes_zero'] else 'no'} \\\\")
        if i < len(v["feats"]) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def forest(b: dict, v: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (no matplotlib; skipping the forest plot)")
        return

    rows = []
    for f in v["feats"]:
        for c, r in b["by_feature"][f]["contrasts"].items():
            rows.append((f, c, r))
    fig, ax = plt.subplots(figsize=(6.2, 0.30 * len(rows) + 1.2))
    ax.axvline(0, color="#5F5E5A", linewidth=1.0, zorder=1)
    labels = []
    for i, (f, c, r) in enumerate(rows):
        y = len(rows) - i
        # Colour by whether the interval excludes zero, so the reader
        # sees which contrasts separate before reading a single number.
        colour = "#1D9E75" if r["excludes_zero"] else "#8A8782"
        ax.plot([r["lo"], r["hi"]], [y, y], color=colour, linewidth=2,
                solid_capstyle="round", zorder=2)
        ax.plot([r["estimate"]], [y], "o", color=colour, markersize=4.5,
                zorder=3)
        mark = " †" if c == PRIMARY else ""
        labels.append((y, f"{f}  {c}{mark}"))
    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([t for _, t in labels], fontsize=8)
    ax.set_ylim(0.3, len(rows) + 0.7)
    ax.set_xlabel("difference in median AUPRC")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    b = json.loads(Path(args.bootstrap).read_text())
    for k in ("by_feature", "n_boot", "alpha"):
        if k not in b:
            sys.exit(f"{Path(args.bootstrap).name} has no '{k}'")
    if not b["by_feature"]:
        sys.exit("the by_feature block is empty")

    v = verdict(b)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "bootstrap_cells.tex").write_text(cells_table(b, v) + "\n")
    (outdir / "bootstrap_contrasts.tex").write_text(
        contrasts_table(b, v) + "\n")
    forest(b, v, outdir / "bootstrap_contrasts.pdf")

    md = ["## Bootstrap intervals", "",
          "| features | cell | n | median | interval |",
          "|---|---|---|---|---|"]
    for f in v["feats"]:
        for q in QUADRANTS:
            c = b["by_feature"][f]["cells"].get(q)
            if c:
                md.append(f"| {f} | {q} | {c['n']} | {c['median']:.3f} "
                          f"| [{c['lo']:.3f}, {c['hi']:.3f}] |")
    md += ["", "## Contrasts", "",
           "| features | contrast | estimate | interval | excludes 0 |",
           "|---|---|---|---|---|"]
    for f in v["feats"]:
        for c, r in b["by_feature"][f]["contrasts"].items():
            md.append(f"| {f} | {c} | {r['estimate']:+.3f} | "
                      f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                      f"{'yes' if r['excludes_zero'] else 'no'} |")
    (outdir / "bootstrap_summary.md").write_text("\n".join(md) + "\n")

    print(f"\nexcludes zero in every feature set: "
          f"{v['always'] if v['always'] else 'none'}")
    print(f"contains zero in every feature set : "
          f"{v['never'] if v['never'] else 'none'}")
    if v["primary_excludes"] is False and not v["primary_any"]:
        print(f"\n  The pre-registered primary contrast {PRIMARY} "
              f"contains zero in every\n  feature set. Report it as "
              f"such; the intermediate contrasts are where the\n  "
              f"separation is, which is what they were included to "
              f"determine.")
    print(f"\nWrote files to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
