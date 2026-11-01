#!/usr/bin/env python3
"""
Generate the covalent-analysis table from the three runs that produced
it.

WHAT THE TABLE HAS TO CARRY
---------------------------
This is an explanation tested and set aside, which is harder to
tabulate than one that holds. Three results have to sit together or
the reader cannot see why it was set aside:

  the pooled association, which is real and sizeable;
  that it is not common across the regimes;
  and the per-regime slopes, which show the regime that prompted the
  explanation has nearly the weakest association with it.

Any one alone misleads. The pooled number alone reads as support. The
slopes alone read as noise without the test that says they differ.

The caption is built from the files for the same reason as elsewhere:
a caption stating a p-value in prose is a transcribed number, and this
one is the hinge of the argument.

CHECKED BEFORE WRITING
----------------------
That the three files describe the same analysis -- the same feature
set, the same layer, the same release. They are separate runs, and a
table pairing one run's slopes with another's test would be wrong in a
way nothing downstream would catch.

USAGE
-----
    python3 generate_covalent_table.py \\
        --pooled covalent_pooled.json \\
        --interaction covalent_interaction.json \\
        --split covalent_split.json \\
        --outdir covalent_tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUADRANTS = ("FF", "FN", "NF", "NN")


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--interaction", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    pool = json.loads(Path(args.pooled).read_text())
    inter = json.loads(Path(args.interaction).read_text())
    split = json.loads(Path(args.split).read_text())
    sm = split.get("_meta")
    if sm is None:
        sys.exit(f"{Path(args.split).name} has no _meta block")

    # The three must describe the same analysis.
    for key, a, b in (("feature", pool.get("feature"),
                       inter.get("feature")),
                      ("layer", pool.get("layer"), inter.get("layer")),
                      ("plinder_dir", pool.get("plinder_dir"),
                       inter.get("plinder_dir"))):
        if a != b:
            sys.exit(f"pooled and interaction disagree on {key}: "
                     f"{a!r} against {b!r}. They are not the same "
                     f"analysis.")
    if sm.get("feature") != pool.get("feature"):
        sys.exit(f"the split used feature {sm.get('feature')!r} and "
                 f"the pooled fit {pool.get('feature')!r}.")
    if sm.get("layer") != pool.get("layer"):
        sys.exit(f"the split used layer {sm.get('layer')!r} and the "
                 f"pooled fit {pool.get('layer')!r}.")
    if sm.get("plinder_dir") != pool.get("plinder_dir"):
        sys.exit(f"the split read {sm.get('plinder_dir')!r} and the "
                 f"pooled fit {pool.get('plinder_dir')!r}.")

    cells = pool["per_cell"]
    slopes = inter["per_regime_slope"]
    sizes = inter["sizes"]
    # Matching metadata does not guarantee the same systems were
    # analysed. The table takes medians from one file and slopes from
    # another, so the subgroups behind them must agree.
    for q in QUADRANTS:
        pc, iz = cells[q], sizes[q]
        if (pc["n_covalent"] != iz["n_covalent"]
                or pc["n_other"] != iz["n_other"]):
            sys.exit(f"{q}: the pooled fit saw "
                     f"{pc['n_covalent']}/{pc['n_other']} and the "
                     f"interaction fit {iz['n_covalent']}/"
                     f"{iz['n_other']}. Different systems.")
    adj, un = pool["adjusted"], pool["unadjusted"]
    p = inter["null"]["p"]
    fired = inter["drop_exceeds_null"]

    # Which regime carries the strongest association, and where the
    # regime that prompted the question sits in that order.
    # Ordered from most negative to most positive, which is what
    # matters here. "Strongest association" would be ambiguous when
    # one slope points the other way, as NF's does.
    order = sorted(QUADRANTS, key=lambda q: slopes[q])
    fn_rank = order.index("FN") + 1
    ORDINAL = {1: "most negative", 2: "second most negative",
               3: "third most negative", 4: "least negative"}

    # Which regime has the lowest non-covalent baseline, for the
    # floor-compression caution. Derived, not assumed to be FN.
    lowest_base = min(
        (q for q in QUADRANTS if cells[q]["median_other"] is not None),
        key=lambda q: cells[q]["median_other"])

    est = adj["estimate"]
    ival = rf"\([{adj['lo']:+.3f}, {adj['hi']:+.3f}]\)"
    if est < 0:
        pooled_phrase = (rf"covalent systems have mean AUPRC lower by "
                         rf"\({abs(est):.3f}\) {ival}")
    elif est > 0:
        pooled_phrase = (rf"covalent systems have mean AUPRC higher "
                         rf"by \({est:.3f}\) {ival}")
    else:
        pooled_phrase = "the adjusted association is zero"

    verdict = (
        rf"The association is not common across regimes: the "
        rf"interaction terms reduce the residual sum of squares by "
        rf"more than a wild cluster bootstrap under the additive "
        rf"model produces (\(p={p:.3f}\)), so one coefficient "
        rf"describes none of them exactly."
        if fired else
        rf"The additive model is not refuted (\(p={p:.3f}\)), which "
        rf"is weaker than showing the association is common."
    )
    placing = (rf" \(FN\), whose composition first suggested the "
               rf"explanation, has the {ORDINAL[fn_rank]} slope of "
               rf"the four.")

    small = [q for q in QUADRANTS if sizes[q]["below_floor"]]
    # Built outside the f-string: Python before 3.12 refuses a
    # backslash inside an f-string expression, and \( is a backslash.
    small_names = " and ".join("\\(" + q + "\\)" for q in small)
    small_note = ""
    if small:
        small_note = (" " + small_names + " "
                      + ("has" if len(small) == 1 else "have")
                      + " at least one subgroup below the declared "
                      + "minimum of " + str(inter["min_half"])
                      + " systems.")

    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Covalent ligands and pocket decodability, by "
        rf"regime. Pooled across the four and adjusting for regime, "
        rf"{pooled_phrase}, against \({un['estimate']:+.3f}\) "
        rf"unadjusted. {verdict}{placing}"
        rf"{small_note} Slopes come from a model with a "
        rf"covalent-by-regime term; medians are descriptive. "
        rf"Exploratory: these comparisons were decided after the "
        rf"composition result.}}",
        r"\label{tab:covalent}",
        r"\begin{tabular}{l rr rr r}", r"\toprule",
        r"Regime & \multicolumn{2}{c}{covalent} & "
        r"\multicolumn{2}{c}{other} & slope \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & \(n\) & median & \(n\) & median & \\", r"\midrule",
    ]
    for q in QUADRANTS:
        c = cells[q]
        mark = r"$^{\dagger}$" if sizes[q]["below_floor"] else ""
        mc = ("--" if c["median_covalent"] is None
              else f"{c['median_covalent']:.3f}")
        mo = ("--" if c["median_other"] is None
              else f"{c['median_other']:.3f}")
        lines.append(
            rf"\({q}\){mark} & {c['n_covalent']} & {mc} & "
            rf"{c['n_other']} & {mo} & {slopes[q]:+.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "covalent.tex").write_text("\n".join(lines) + "\n")

    # The numbers the prose cites, as macros. The subsection \input{}s
    # this and then uses \CovAdj and the rest, so every figure in the
    # text has the same source as the table beside it and cannot drift
    # from it on a re-run.
    pm = [r"% Generated. Do not edit; re-run the generator.",
          rf"\newcommand{{\CovAdj}}{{{abs(est):.3f}}}",
          rf"\newcommand{{\CovAdjCI}}{{[{adj['lo']:+.3f}, "
          rf"{adj['hi']:+.3f}]}}",
          rf"\newcommand{{\CovUnadj}}{{{un['estimate']:+.3f}}}",
          rf"\newcommand{{\CovP}}{{{p:.3f}}}",
          rf"\newcommand{{\CovFNPlace}}{{{ORDINAL[fn_rank]}}}",
          rf"\newcommand{{\CovMostNeg}}{{\({order[0]}\) and "
          rf"\({order[1]}\)}}",
          rf"\newcommand{{\CovLeastNeg}}{{\({order[-2]}\) and "
          rf"\({order[-1]}\)}}",
          rf"\newcommand{{\CovSplitEst}}{{{sm['estimate']:+.3f}}}",
          rf"\newcommand{{\CovSplitCI}}{{[{sm['lo']:+.3f}, "
          rf"{sm['hi']:+.3f}]}}",
          rf"\newcommand{{\CovLowestBase}}{{\({lowest_base}\)}}"]
    (outdir / "covalent_macros.tex").write_text("\n".join(pm) + "\n")

    md = ["## Covalent ligands and decodability", "",
          f"Pooled, adjusted for regime: {adj['estimate']:+.3f} "
          f"[{adj['lo']:+.3f}, {adj['hi']:+.3f}]  "
          f"(unadjusted {un['estimate']:+.3f})", "",
          f"Interaction null test: p = {p:.4f}, "
          f"{'fires' if fired else 'does not fire'}", "",
          "| regime | covalent n | median | other n | median | slope |",
          "|---|---:|---:|---:|---:|---:|"]
    for q in QUADRANTS:
        c = cells[q]
        mc = ("--" if c["median_covalent"] is None
              else f"{c['median_covalent']:.3f}")
        mo = ("--" if c["median_other"] is None
              else f"{c['median_other']:.3f}")
        md.append(f"| {q} | {c['n_covalent']} | {mc} | "
                  f"{c['n_other']} | {mo} | {slopes[q]:+.3f} |")
    md += ["",
           f"Within-cell contrast (FN gap minus NF gap): "
           f"{sm['estimate']:+.3f} [{sm['lo']:+.3f}, {sm['hi']:+.3f}], "
           f"comparison "
           f"{'made' if sm.get('comparison_made') else 'NOT made'}.",
           "",
           f"FN ranks {fn_rank} of 4 from most negative to most "
           f"positive covalent slope."]
    (outdir / "covalent_summary.md").write_text("\n".join(md) + "\n")

    print(f"pooled adjusted   {adj['estimate']:+.3f} "
          f"[{adj['lo']:+.3f}, {adj['hi']:+.3f}]")
    print(f"interaction p     {p:.4f}  "
          f"({'fires' if fired else 'does not fire'})")
    print(f"slopes            " + "  ".join(
        f"{q} {slopes[q]:+.3f}" for q in order))
    print(f"FN has the {ORDINAL[fn_rank]} slope of the four")
    print(f"lowest non-covalent baseline: {lowest_base}")
    print(f"\nWrote 3 files to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
