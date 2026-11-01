#!/usr/bin/env python3
"""
Generate the Holm-corrected contrast table and the macros its
paragraph uses.

WHAT THE TABLE HAS TO CARRY
---------------------------
Eighteen rows, each with its estimate, its percentile interval, its
raw p-value and its Holm-adjusted one, and whether it survives. The
interval and the p-value come from the same resampling, which the
producing script verified, and both are shown because they answer
different questions: the interval says how much a different sample of
clusters would move the estimate, the p-value says how often a world
with no difference would produce something this large.

The caption states the family, because a corrected p-value means
nothing without one.

WHAT THE SCRIPT WORKS OUT RATHER THAN BEING TOLD
------------------------------------------------
Which contrasts survive, how many of them involve the regime of
interest, and whether that regime accounts for all of them. The
headline sentence in the paper turns on exactly that, and a generator
that is told the answer cannot catch it changing.

It also finds how many contrasts sit at the bootstrap floor. That is
the smallest p-value the replicate count can produce, so a contrast
there has an unknown true p-value and the correction is conservative
for it. A table that showed the floor as though it were an estimate
would overstate precision in the direction of caution, which is the
less harmful direction but still worth saying.

USAGE
-----
    python3 generate_holm_table.py \\
        --holm holm_contrasts.json \\
        --outdir holm_tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The order the table prints in, fixed here so it does not follow
# whatever order the producing script happened to emit.
FEATURES = ("s", "zpool", "both")
PAIRS = (("FF", "FN"), ("NF", "FN"), ("NN", "FN"),
         ("NF", "NN"), ("FF", "NN"), ("NF", "FF"))
FOCUS = "FN"


def fmt_p(p, floor):
    """The computed value, never a bound.

    With p = (1 + #extreme) / (B + 1), zero extreme draws gives
    exactly 1/(B+1) rather than something smaller. Printing it as
    "< 1/(B+1)" would present a computed number as an upper bound and
    read as more significant than what was calculated. The caption
    says instead how many values sit at that resolution limit.
    """
    return f"{p:.4f}"


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holm", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    d = json.loads(Path(args.holm).read_text())
    for k in ("contrasts", "family_size", "alpha", "n_boot",
              "correction", "bootstrap_p_floor"):
        if k not in d:
            sys.exit(f"{Path(args.holm).name} has no '{k}'")
    rows = d["contrasts"]
    if len(rows) != d["family_size"]:
        sys.exit(f"the file reports a family of {d['family_size']} "
                 f"but carries {len(rows)} contrasts")
    # The family was fixed at eighteen before the result. A file
    # carrying a different number is a different analysis, however
    # self-consistent it is.
    if d["family_size"] != 18:
        sys.exit(f"the pre-specified family is eighteen contrasts; "
                 f"this file reports {d['family_size']}.")
    alpha, floor = d["alpha"], d["bootstrap_p_floor"]

    # Index by feature and contrast so the printed order is the fixed
    # one above rather than the file's.
    by = {(r["feature"], r["contrast"]): r for r in rows}
    want = [(f, f"{a}-{b}") for f in FEATURES for a, b in PAIRS]
    missing = [k for k in want if k not in by]
    if missing:
        sys.exit(f"the file lacks {len(missing)} of the expected "
                 f"rows, e.g. {missing[:3]}. Either the feature sets "
                 f"or the contrast set differ from this table's "
                 f"fixed order.")
    extra = [k for k in by if k not in want]
    if extra:
        sys.exit(f"the file carries rows outside this table's fixed "
                 f"order: {extra[:3]}")

    surv = [r for r in rows if r["survives"]]
    focus_all = [r for r in rows if FOCUS in r["contrast"]]
    focus_surv = [r for r in focus_all if r["survives"]]
    # The strongest sentence says FN is BELOW the others, which is a
    # claim about direction and not only about survival. The contrast
    # directions were fixed in advance as FF-FN, NF-FN and NN-FN, so
    # FN is lower exactly when the estimate is positive. Checked
    # rather than assumed: a contrast running the other way would
    # otherwise be described as supporting the claim it contradicts.
    focus_below = [r for r in focus_surv if r["estimate"] > 0]
    at_floor = [r for r in rows if abs(r["p_raw"] - floor) < 1e-12]

    # The claim the paper will make, derived rather than asserted.
    if not surv:
        verdict = (rf"No contrast survives correction at "
                   rf"\(\alpha={alpha}\).")
    elif (len(focus_below) == len(focus_all)
          and len(focus_surv) == len(surv)):
        verdict = (rf"All {len(focus_all)} contrasts placing "
                   rf"\({FOCUS}\) below another regime survive, in "
                   rf"every feature set, and no other contrast does.")
    elif len(focus_surv) == len(surv):
        verdict = (rf"{len(focus_surv)} of the {len(focus_all)} "
                   rf"contrasts involving \({FOCUS}\) survive, of "
                   rf"which {len(focus_below)} place it below the "
                   rf"other regime, and no other contrast survives.")
    else:
        others = len(surv) - len(focus_surv)
        verdict = (rf"{len(focus_surv)} of {len(focus_all)} "
                   rf"contrasts involving \({FOCUS}\) survive, of "
                   rf"which {len(focus_below)} place it below the "
                   rf"other regime, alongside {others} that do not "
                   rf"involve it.")

    floor_note = ("" if not at_floor else
                  rf" {len(at_floor)} raw p-values equal "
                  rf"\(1/({d['n_boot']}+1)\), the smallest value "
                  rf"this many replicates can produce; their true "
                  rf"values may be smaller, so the correction is "
                  rf"conservative for them.")

    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{The eighteen regime contrasts with "
        rf"{d['correction']} correction. The multiplicity family is "
        rf"all {d['family_size']} contrasts across all three feature "
        rf"sets, fixed before the result rather than narrowed to the "
        rf"contrasts that separate. Intervals are percentile "
        rf"intervals from {d['n_boot']} resamples of receptor-family "
        rf"clusters; p-values are two-sided and null-centred on the "
        rf"same resamples. {verdict}{floor_note}}}",
        r"\label{tab:holm}",
        r"\begin{tabular}{l l r l r r c}", r"\toprule",
        r"Features & Contrast & Estimate & 95\% interval & "
        r"$p$ & $p_{\mathrm{Holm}}$ & Survives \\",
        r"\midrule",
    ]
    last = None
    for f, name in want:
        r = by[(f, name)]
        if last is not None and f != last:
            lines.append(r"\midrule")
        last = f
        a, b = name.split("-")
        lines.append(
            rf"\texttt{{{f}}} & \({a}\)--\({b}\) & "
            rf"{r['estimate']:+.3f} & "
            rf"[{r['lo']:+.3f}, {r['hi']:+.3f}] & "
            rf"{fmt_p(r['p_raw'], floor)} & {r['p_holm']:.4f} & "
            rf"{'yes' if r['survives'] else 'no'} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "holm_contrasts.tex").write_text("\n".join(lines) + "\n")

    # The largest Holm-adjusted p-value among the surviving focus
    # contrasts: the paper can say "all below X" from one number.
    worst = max((r["p_holm"] for r in focus_surv), default=None)

    m = [r"% Generated. Do not edit; re-run the generator.",
         rf"\newcommand{{\HolmFamily}}{{{d['family_size']}}}",
         rf"\newcommand{{\HolmAlpha}}{{{alpha}}}",
         rf"\newcommand{{\HolmNBoot}}{{{d['n_boot']:,}}}",
         rf"\newcommand{{\HolmNSurvive}}{{{len(surv)}}}",
         rf"\newcommand{{\HolmNFocus}}{{{len(focus_all)}}}",
         rf"\newcommand{{\HolmNFocusSurvive}}{{{len(focus_surv)}}}",
         rf"\newcommand{{\HolmNFocusBelow}}{{{len(focus_below)}}}",
         rf"\newcommand{{\HolmNOther}}{{"
         rf"{len(rows) - len(focus_all)}}}",
         rf"\newcommand{{\HolmNOtherSurvive}}{{"
         rf"{len(surv) - len(focus_surv)}}}",
         rf"\newcommand{{\HolmNAtFloor}}{{{len(at_floor)}}}",
         rf"\newcommand{{\HolmFloor}}{{{floor:.4f}}}"]
    if worst is not None:
        m.append(rf"\newcommand{{\HolmWorstFocus}}{{{worst:.3f}}}")
    (outdir / "holm_macros.tex").write_text("\n".join(m) + "\n")

    md = [f"## Holm-corrected contrasts (family of "
          f"{d['family_size']})", "",
          "| features | contrast | estimate | interval | p | p Holm | "
          "survives |", "|---|---|---:|---:|---:|---:|---|"]
    for f, name in want:
        r = by[(f, name)]
        md.append(
            f"| {f} | {name} | {r['estimate']:+.3f} | "
            f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p_raw']:.4f} | "
            f"{r['p_holm']:.4f} | "
            f"{'yes' if r['survives'] else 'no'} |")
    md += ["",
           f"{len(focus_surv)} of {len(focus_all)} contrasts "
           f"involving {FOCUS} survive; "
           f"{len(surv) - len(focus_surv)} of "
           f"{len(rows) - len(focus_all)} others do.",
           f"{len(at_floor)} raw p-values at the bootstrap floor of "
           f"{floor:.2e}."]
    (outdir / "holm_summary.md").write_text("\n".join(md) + "\n")

    print(f"family {d['family_size']}, alpha {alpha}")
    print(f"surviving: {len(surv)}  "
          f"({len(focus_surv)} involve {FOCUS}, "
          f"{len(surv) - len(focus_surv)} do not)")
    print(f"of those involving {FOCUS}, {len(focus_below)} place it "
          f"below the other regime")
    if worst is not None:
        print(f"largest adjusted p among surviving {FOCUS} "
              f"contrasts: {worst:.4f}")
    print(f"at the bootstrap floor: {len(at_floor)}")
    print(f"\nWrote 3 files to {outdir}")
    print("\n  The caption states the family, because a corrected "
          "p-value without one\n  is not interpretable, and the "
          "verdict sentence is derived from the rows\n  rather than "
          "written alongside them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
