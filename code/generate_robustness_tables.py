#!/usr/bin/env python3
"""
Generate the robustness tables from the two check outputs.

Both tables qualify the main result rather than supporting it, and
both are written from the JSON those checks produced rather than
typed, so re-running a check updates the table.

  robustness_design.tex    how much the familiarity cells explain
                           beyond protein length, pocket size and
                           positive rate
  robustness_altsp.tex     cell medians under the primary and an
                           alternative operationalisation of pocket
                           familiarity
  robustness_summary.md    both, for progress notes

CAPTIONS ARE DERIVED, NOT ASSERTED
----------------------------------
A caption stating what the numbers show is a transcribed claim in
prose. Re-run a check on other data and the table updates while the
sentence above it quietly becomes false, which is the same failure as
typing a number by hand and harder to notice.

So the alternative-operationalisation caption reports what the data
says -- in how many estimable comparisons the contrast keeps its
direction, and whether its magnitude is smaller or larger -- computed
from the JSON. The design-factor caption states what the columns are
and leaves the interpretation to the text, where the ratios can be
described as observed rather than assumed.

USAGE
-----
    python3 generate_robustness_tables.py \\
        --design design_factors.json \\
        --alt-sp alternative_sp_pli70.json \\
        --outdir robustness_tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUADRANTS = ("FF", "FN", "NF", "NN")
PAIRS = (("FF", "FN"), ("NF", "NN"))
DESIGN_REQUIRED = {"layer", "n_systems", "r2_design_only",
                   "r2_regime_only", "r2_both",
                   "regime_adds_over_design",
                   "design_adds_over_regime"}


def altsp_verdict(a: dict) -> dict:
    """What the alternative actually shows, read from the data.

    A caption that states a conclusion the generator was written to
    expect is a transcribed claim in prose: re-run the check on other
    data and the table updates while the sentence above it quietly
    becomes false. The sentence is therefore derived here.
    """
    gaps = []
    for f in FEATURE_ORDER:
        if f not in a["by_feature"]:
            continue
        c = a["by_feature"][f]["cells"]
        for fam, nov in PAIRS:
            if fam not in c or nov not in c:
                continue
            o1, o2 = c[fam]["median_original"], c[nov]["median_original"]
            n1, n2 = c[fam]["median_new"], c[nov]["median_new"]
            if None in (o1, o2, n1, n2):
                continue
            gaps.append((f, fam, nov, o1 - o2, n1 - n2))
    if not gaps:
        return {"n": 0, "sentence": "No comparison had enough systems "
                                    "in both cells to estimate."}
    # An exact sign, not a truth test: (o > 0) == (n > 0) counts a gap
    # of exactly zero against a negative one as agreeing, since both
    # are False. Not reachable with these data, but the comparison
    # should mean what it says.
    def _sign(x):
        return 1 if x > 0 else (-1 if x < 0 else 0)

    signs = sum(1 for _, _, _, o, n in gaps if _sign(o) == _sign(n))
    shrunk = sum(1 for _, _, _, o, n in gaps if abs(n) < abs(o))
    grown = sum(1 for _, _, _, o, n in gaps if abs(n) > abs(o))
    n = len(gaps)
    if signs == n:
        sent = (f"The direction of the familiar-versus-novel pocket "
                f"contrast is preserved in all {n} estimable "
                f"comparisons.")
    elif signs == 0:
        sent = (f"The direction of the contrast reverses in all {n} "
                f"estimable comparisons.")
    else:
        sent = (f"The direction of the contrast is preserved in "
                f"{signs} of {n} estimable comparisons.")
    if shrunk == n:
        sent += " Its magnitude is smaller under the alternative."
    elif grown == n:
        sent += " Its magnitude is larger under the alternative."
    elif shrunk or grown:
        sent += (f" Its magnitude is smaller in {shrunk} and larger in "
                 f"{grown}.")
    return {"n": n, "signs_hold": signs, "shrunk": shrunk,
            "grown": grown, "gaps": gaps, "sentence": sent}
FEATURE_ORDER = ("s", "zpool", "both")
FEATURE_TEX = {"s": r"\texttt{s}", "zpool": r"\texttt{z}",
               "both": "concat."}


def design_table(d: dict) -> str:
    feats = [f for f in FEATURE_ORDER if f in d]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-protein AUPRC against design factors and "
        r"familiarity regime, at each feature set's peak layer. "
        r"\emph{design} is an ordinary least squares fit on protein "
        r"length, pocket size and positive rate; \emph{regime} on "
        r"indicators for the four familiarity cells. The last column "
        r"is the variance the familiarity cells explain beyond the "
        r"design covariates. The interpretation is in the text.}",
        r"\label{tab:robustness-design}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Features & Layer & $n$ & $R^2_{\mathrm{design}}$ & "
        r"$R^2_{\mathrm{regime}}$ & $R^2_{\mathrm{both}}$ & "
        r"$\Delta R^2_{\mathrm{regime}\mid\mathrm{design}}$ \\",
        r"\midrule",
    ]
    for f in feats:
        r = d[f]
        lines.append(
            f"{FEATURE_TEX[f]} & {r['layer']} & {r['n_systems']} & "
            f"{r['r2_design_only']:.3f} & {r['r2_regime_only']:.3f} & "
            f"{r['r2_both']:.3f} & "
            f"{r['regime_adds_over_design']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def altsp_table(a: dict) -> str:
    feats = [f for f in FEATURE_ORDER if f in a["by_feature"]]
    pct = 100 * a["n_reassigned"] / a["n_test"]
    verdict = altsp_verdict(a)
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Cell medians under the primary definition of "
        rf"pocket familiarity and under an alternative. The "
        rf"alternative uses \texttt{{{a['metric']}}} at threshold "
        rf"{a['threshold']:g}, aggregated by {a['agg']} over a "
        rf"system's ligands, and reassigns {a['n_reassigned']} of "
        rf"{a['n_test']} test systems ({pct:.0f}\%). The probe is "
        rf"unchanged: it was trained on pooled training systems and "
        rf"never saw a cell label, so only the evaluation grouping "
        rf"differs. {verdict['sentence']} Cells with fewer than 20 "
        rf"systems are not reported.}}",
        r"\label{tab:robustness-altsp}",
        r"\begin{tabular}{ll rr rr}", r"\toprule",
        r" & & \multicolumn{2}{c}{primary} & "
        r"\multicolumn{2}{c}{alternative} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Features & Cell & $n$ & median & $n$ & median \\",
        r"\midrule",
    ]
    for i, f in enumerate(feats):
        cells = a["by_feature"][f]["cells"]
        for j, q in enumerate(QUADRANTS):
            c = cells.get(q)
            if c is None:
                continue
            mo = (f"{c['median_original']:.3f}"
                  if c["median_original"] is not None else "--")
            mn = (f"{c['median_new']:.3f}"
                  if c["median_new"] is not None else "--")
            head = FEATURE_TEX[f] if j == 0 else ""
            lines.append(f"{head} & {q} & {c['n_original']} & {mo} & "
                         f"{c['n_new']} & {mn} \\\\")
        if i < len(feats) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def summary_md(d: dict, a: dict) -> str:
    out = ["## Robustness checks", "",
           "### Design factors", "",
           "| features | layer | design | regime | both | regime adds |",
           "|---|---|---|---|---|---|"]
    for f in (x for x in FEATURE_ORDER if x in d):
        r = d[f]
        out.append(f"| {f} | {r['layer']} | {r['r2_design_only']:.3f} | "
                   f"{r['r2_regime_only']:.3f} | {r['r2_both']:.3f} | "
                   f"{r['regime_adds_over_design']:.3f} |")
    pct = 100 * a["n_reassigned"] / a["n_test"]
    out += ["", f"### Alternative operationalisation "
                f"(`{a['metric']}` at {a['threshold']:g}, "
                f"{a['agg']} over ligands)", "",
            f"{a['n_reassigned']} of {a['n_test']} test systems "
            f"reassigned ({pct:.0f}%).", "",
            "| features | cell | n primary | median | n alt | median |",
            "|---|---|---|---|---|---|"]
    for f in (x for x in FEATURE_ORDER if x in a["by_feature"]):
        for q in QUADRANTS:
            c = a["by_feature"][f]["cells"].get(q)
            if c is None:
                continue
            mo = ("--" if c["median_original"] is None
                  else f"{c['median_original']:.3f}")
            mn = ("--" if c["median_new"] is None
                  else f"{c['median_new']:.3f}")
            out.append(f"| {f} | {q} | {c['n_original']} | {mo} | "
                       f"{c['n_new']} | {mn} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--alt-sp", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    d = json.loads(Path(args.design).read_text())
    a = json.loads(Path(args.alt_sp).read_text())
    for f in FEATURE_ORDER:
        if f not in d:
            continue
        absent = DESIGN_REQUIRED - set(d[f])
        if absent:
            sys.exit(f"{Path(args.design).name}: feature '{f}' lacks "
                     f"{sorted(absent)}")
    for k in ("by_feature", "n_test", "n_reassigned", "metric",
              "threshold", "agg"):
        if k not in a:
            sys.exit(f"{Path(args.alt_sp).name} has no '{k}'")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "robustness_design.tex").write_text(design_table(d) + "\n")
    (outdir / "robustness_altsp.tex").write_text(altsp_table(a) + "\n")
    (outdir / "robustness_summary.md").write_text(summary_md(d, a))

    # The two numbers a reader should take away, computed rather than
    # asserted, so a changed check changes what is printed.
    print("design factors:")
    for f in (x for x in FEATURE_ORDER if x in d):
        r = d[f]
        ratio = (r["design_adds_over_regime"]
                 / r["regime_adds_over_design"]
                 if r["regime_adds_over_design"] else float("nan"))
        print(f"  {f:6} regime adds {r['regime_adds_over_design']:.3f}, "
              f"design adds {r['design_adds_over_regime']:.3f} "
              f"({ratio:.1f}x more)")

    print(f"\nalternative operationalisation "
          f"({a['metric']} at {a['threshold']:g}):")
    for f in (x for x in FEATURE_ORDER if x in a["by_feature"]):
        c = a["by_feature"][f]["cells"]
        pairs = []
        for fam, nov in (("FF", "FN"), ("NF", "NN")):
            o1, o2 = c[fam]["median_original"], c[nov]["median_original"]
            n1, n2 = c[fam]["median_new"], c[nov]["median_new"]
            if None in (o1, o2, n1, n2):
                continue
            pairs.append(f"{fam}-{nov} {o1 - o2:+.3f} -> {n1 - n2:+.3f}")
        print(f"  {f:6} " + ",  ".join(pairs))

    print(f"\nWrote 3 files to {outdir}")
    print(f"\n  {altsp_verdict(a)['sentence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
