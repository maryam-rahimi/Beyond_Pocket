#!/usr/bin/env python3
"""
Generate the continuous two-axis table, its caption and the macros the
prose uses.

EVERY FIGURE IN THE SECTION COMES FROM HERE
-------------------------------------------
The table, the caption and the numbers in the surrounding paragraphs
are written from the same two files. The prose uses the macros rather
than repeating the values, so a re-run moves the text with the table
instead of leaving them to drift apart.

That matters more in this section than in most. The family-support
slope is only interpretable alongside the range it was estimated
over, and the range is a property of how the test set was sampled
rather than of the model. A paragraph that cites the slope from the
regression and the range from memory would eventually cite a slope
from one run and a range from another.

WHAT THE SUPPORT RANGE IS DOING HERE
------------------------------------
The stratified test set is concentrated at low family support, because
two of the four cells are novel-family by construction and every
system in them has no pre-cutoff relative. So the family slope is
estimated mostly between no support and modest support. The macros
carry the median, the share at zero and the ninetieth percentile for
exactly that reason: the claim is bounded by them and the prose has
to say so.

WHAT IS CHECKED
---------------
That the regression and the support table describe the same systems,
and that the counts behind the range are the counts behind the fit.
Two files written by two scripts can disagree, and a section whose
slope comes from one and whose range comes from the other would be
wrong in a way nothing downstream would catch.

USAGE
-----
    python3 generate_continuous_table.py \\
        --fit continuous_two_axis.json \\
        --targets targets_with_sf_count.parquet \\
        --index probe_index.parquet \\
        --outdir continuous_tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Fixed here so the table's shape does not follow dict order.
ROW_ORDER = ("S_f", "S_p", "S_f x S_p")
LABEL = {"S_f": r"family support, \(\log_{10}(1+S_f)\)",
         "S_p": r"pocket familiarity, \(S_p\)",
         "S_f x S_p": "interaction"}


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    fit = json.loads(Path(args.fit).read_text())
    meta = fit.get("_meta")
    if meta is None:
        sys.exit(f"{Path(args.fit).name} has no _meta block")
    rows = [r for r in ROW_ORDER if r in fit]
    extra = [k for k in fit
             if k not in ROW_ORDER and k not in ("_meta", "intercept")]
    if extra:
        sys.exit(f"{Path(args.fit).name} carries terms not in this "
                 f"table's fixed order: {extra}.")
    if len(rows) != len(ROW_ORDER):
        sys.exit(f"expected {len(ROW_ORDER)} terms, found {len(rows)}")

    tg = pd.read_parquet(args.targets)
    idx = pd.read_parquet(args.index)
    test_idx = idx[idx.split == "test"]
    if test_idx.system_id.duplicated().any():
        sys.exit("duplicate system_id in the test split")
    if tg.system_id.duplicated().any():
        sys.exit("duplicate system_id in --targets")
    test = set(test_idx.system_id)
    res = tg[tg.S_f_status == "resolved"]
    sub = res[res.system_id.isin(test)]

    # Three checks, not one. Matching the fitted count alone cannot
    # tell 534 of 534 resolved from 534 of 550: both give the same
    # number of resolved test systems while meaning different things,
    # and only the second would mean systems were lost from the
    # regression.
    test_tg = tg[tg.system_id.isin(test)]
    n_unresolved_test = int((test_tg.S_f_status != "resolved").sum())
    if len(sub) != meta["n_fitted"]:
        sys.exit(f"the fit used {meta['n_fitted']} systems and the "
                 f"support table has {len(sub)} resolved test "
                 f"systems. They are not the same analysis.")
    if n_unresolved_test != meta["n_dropped"]:
        sys.exit(f"the fit reports {meta['n_dropped']} dropped, but "
                 f"the supplied tables imply {n_unresolved_test} "
                 f"unresolved test systems.")
    if meta["n_fitted"] + meta["n_dropped"] != len(test_idx):
        sys.exit(f"{meta['n_fitted']} fitted plus "
                 f"{meta['n_dropped']} dropped is not the test-set "
                 f"size of {len(test_idx)}.")

    med_test = float(sub.S_f_count.median())
    med_all = float(res.S_f_count.median())
    zero_frac = float((sub.S_f_count == 0).mean())
    p90 = float(sub.S_f_count.quantile(0.90))
    n_unresolved_all = int((tg.S_f_status != "resolved").sum())

    sep = [r for r in rows if fit[r]["standardised"]["excludes_zero"]]
    inter_sep = "S_f x S_p" in sep
    sf, sp = fit["S_f"]["standardised"], fit["S_p"]["standardised"]

    # Which axis is larger in standardised units, and by how much.
    ratio = (abs(sp["estimate"]) / abs(sf["estimate"])
             if sf["estimate"] else float("nan"))

    main_note = ("" if inter_sep else
                 " The interaction does not separate from zero, so "
                 "each slope is read at the other axis's mean.")

    caption = (
        rf"\caption{{Both familiarity axes fitted continuously, "
        rf"without the coverage threshold that defines the four "
        rf"regimes. Standardised main-effect coefficients describe a "
        rf"one-standard-deviation change in one axis at the mean of "
        rf"the other; the interaction is on the product of the two "
        rf"standardised axes. Raw coefficients use the corresponding "
        rf"centred natural units. Both axes are centred, so each main "
        rf"effect is the slope at the other axis's mean. "
        rf"Fitted on {meta['n_fitted']} test systems at layer "
        rf"{meta['layer']}, feature set "
        rf"\\texttt{{{meta['feature']}}}, with intervals from "
        rf"{meta['n_boot']} resamples of receptor-family clusters."
        rf"{main_note} The family slope is estimated over a test set "
        rf"concentrated at low support: {100 * zero_frac:.0f} per "
        rf"cent of its systems have no pre-cutoff relative and its "
        rf"median count is {med_test:.0f}, against {med_all:.0f} "
        rf"across all resolved targets.}}")

    lines = [r"\begin{table}[t]", r"\centering", caption,
             r"\label{tab:continuous}",
             r"\begin{tabular}{l rl rl}", r"\toprule",
             r"Term & \multicolumn{2}{c}{standardised} & "
             r"\multicolumn{2}{c}{raw} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"\midrule"]
    for r in rows:
        st, rw = fit[r]["standardised"], fit[r]["raw"]
        mark = r"$^{\star}$" if st["excludes_zero"] else ""
        lines.append(
            rf"{LABEL[r]}{mark} & {st['estimate']:+.3f} & "
            rf"[{st['lo']:+.3f}, {st['hi']:+.3f}] & "
            rf"{rw['estimate']:+.4f} & "
            rf"[{rw['lo']:+.4f}, {rw['hi']:+.4f}] \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "continuous.tex").write_text("\n".join(lines) + "\n")

    # The macros the prose uses, so no figure is typed twice.
    def ci(d):
        return f"[{d['lo']:+.3f}, {d['hi']:+.3f}]"

    m = [r"% Generated. Do not edit; re-run the generator.",
         rf"\newcommand{{\ConSf}}{{{sf['estimate']:+.3f}}}",
         rf"\newcommand{{\ConSfCI}}{{{ci(sf)}}}",
         rf"\newcommand{{\ConSp}}{{{sp['estimate']:+.3f}}}",
         rf"\newcommand{{\ConSpCI}}{{{ci(sp)}}}",
         rf"\newcommand{{\ConSfDir}}{{"
         rf"{'negatively' if sf['estimate'] < 0 else 'positively'}}}",
         rf"\newcommand{{\ConSpDir}}{{"
         rf"{'positively' if sp['estimate'] > 0 else 'negatively'}}}",
         rf"\newcommand{{\ConRatio}}{{{ratio:.1f}}}",
         rf"\newcommand{{\ConNFit}}{{{meta['n_fitted']}}}",
         rf"\newcommand{{\ConZeroPct}}{{{100 * zero_frac:.0f}}}",
         rf"\newcommand{{\ConMedTest}}{{{med_test:.0f}}}",
         rf"\newcommand{{\ConMedAll}}{{{med_all:.0f}}}",
         rf"\newcommand{{\ConPninety}}{{{p90:.0f}}}",
         rf"\newcommand{{\ConUnresolvedAll}}{{{n_unresolved_all}}}",
         rf"\newcommand{{\ConUnresolvedTest}}{{"
         rf"{n_unresolved_test}}}",
         rf"\newcommand{{\ConCeiling}}{{"
         rf"{100 * meta['S_p_at_or_above_99']:.0f}}}",
         rf"\newcommand{{\ConInterNote}}{{"
         rf"{'The interaction also separates from zero, so neither main effect is an independent slope' if inter_sep else 'The interaction does not separate from zero, so each slope is read at the other axis mean'}}}"]
    (outdir / "continuous_macros.tex").write_text("\n".join(m) + "\n")

    md = ["## Continuous two-axis fit", "",
          f"Fitted on {meta['n_fitted']} test systems, layer "
          f"{meta['layer']}, feature {meta['feature']}.", "",
          "| term | standardised | raw |", "|---|---|---|"]
    for r in rows:
        st, rw = fit[r]["standardised"], fit[r]["raw"]
        md.append(f"| {r} | {st['estimate']:+.3f} {ci(st)} | "
                  f"{rw['estimate']:+.4f} |")
    md += ["",
           f"Support in the fitted test set: median {med_test:.0f}, "
           f"p90 {p90:.0f}, {100 * zero_frac:.0f}% at zero.",
           f"Across all resolved targets: median {med_all:.0f}.",
           f"{n_unresolved_all} targets in the full table lack an "
           f"exact count. {n_unresolved_test} of them fall in the "
           f"test split, so {'none were' if n_unresolved_test == 0 else 'some were'} "
           f"lost from this regression."]
    (outdir / "continuous_summary.md").write_text("\n".join(md) + "\n")

    print(f"fitted on {meta['n_fitted']}, layer {meta['layer']}")
    print(f"separating: {sep}")
    print(f"S_f {sf['estimate']:+.3f} {ci(sf)}   "
          f"S_p {sp['estimate']:+.3f} {ci(sp)}")
    print(f"support in the fitted set: median {med_test:.0f}, "
          f"p90 {p90:.0f}, {100 * zero_frac:.0f}% at zero")
    print(f"\nWrote 3 files to {outdir}")
    print("\n  The support range is in the macros because the family "
          "slope is only\n  interpretable alongside it, and a "
          "paragraph citing one from a file and\n  the other from "
          "memory would eventually cite them from different runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
