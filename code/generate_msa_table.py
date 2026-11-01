#!/usr/bin/env python3
"""
Generate the alignment-depth table and the macros its paragraph uses.

WHY THE MACROS AND NOT JUST THE TABLE
-------------------------------------
This result is a null, and a null is only as good as the caveats
attached to it. Three numbers carry those caveats: how many systems
sit at the recording ceiling, what the correlation is among the rest,
and how close the two matched cells are on depth. A paragraph that
cites them from memory while the table is regenerated would
eventually cite one run's caveats beside another run's numbers, and
the mismatch would be invisible.

So the paragraph's figures are emitted as macros from the same file
that fills the table.

WHAT THE SCRIPT DECIDES RATHER THAN BEING TOLD
----------------------------------------------
Which feature set is primary: the one whose layer matches the layer
the depth curves peak at is not recoverable from this file alone, so
the primary set is named on the command line and checked to exist.

Whether the matched pair is close on depth: the ratio is computed, not
asserted, and the caption says which direction it runs.

Whether any regime contradicts the depth account outright: a regime
with markedly lower depth and higher AUPRC than FN is evidence against
depth explaining the gap, and the script finds it rather than the
author remembering it.

USAGE
-----
    python3 generate_msa_table.py \\
        --msa msa_covariate.json \\
        --primary s \\
        --outdir msa_tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUADRANTS = ("FF", "FN", "NF", "NN")
PAIR = ("FN", "NF")


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msa", required=True)
    ap.add_argument("--primary", default="s",
                    help="the feature set the prose treats as "
                         "primary; its numbers fill the macros")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    d = json.loads(Path(args.msa).read_text())
    if "by_feature" not in d or "ceiling" not in d:
        sys.exit(f"{Path(args.msa).name} is not an msa_covariate "
                 f"output")
    feats = sorted(d["by_feature"])
    if args.primary not in feats:
        sys.exit(f"--primary {args.primary!r} is not in the file; it "
                 f"holds {feats}")
    ceiling = d["ceiling"]
    pf = d["by_feature"][args.primary]
    cells = pf["cells"]

    for q in QUADRANTS:
        if q not in cells:
            sys.exit(f"the primary feature set has no {q} cell")
    for f in feats:
        o = d["by_feature"][f]
        if o["n"] != pf["n"] or o["n_at_ceiling"] != pf["n_at_ceiling"]:
            sys.exit(f"feature set {f} covers {o['n']} systems with "
                     f"{o['n_at_ceiling']} at the ceiling, against "
                     f"{pf['n']} and {pf['n_at_ceiling']} for "
                     f"{args.primary}. The same proteins should "
                     f"underlie every feature set.")

    if sum(cells[q]["n"] for q in QUADRANTS) != pf["n"]:
        sys.exit(f"the cell counts sum to "
                 f"{sum(cells[q]['n'] for q in QUADRANTS)} but the "
                 f"file reports {pf['n']} systems.")

    fn, nf = cells[PAIR[0]], cells[PAIR[1]]
    depth_ratio = (fn["median_depth_min"] / nf["median_depth_min"]
                   if nf["median_depth_min"] else float("nan"))
    auprc_ratio = (nf["median_auprc"] / fn["median_auprc"]
                   if fn["median_auprc"] else float("nan"))

    # Which way the depth difference runs, rather than its size
    # alone. A factor stated without a direction leaves the reader to
    # guess which cell is better aligned, and the whole point is that
    # the better-aligned one is not the better-scoring one.
    if fn["median_depth_min"] > nf["median_depth_min"]:
        depth_phrase = (rf"\({PAIR[0]}\) has {depth_ratio:.2f}"
                        rf"\(\times\) the median alignment depth "
                        rf"of \({PAIR[1]}\)")
    elif fn["median_depth_min"] < nf["median_depth_min"]:
        depth_phrase = (rf"\({PAIR[1]}\) has {1 / depth_ratio:.2f}"
                        rf"\(\times\) the median alignment depth "
                        rf"of \({PAIR[0]}\)")
    else:
        depth_phrase = (rf"\({PAIR[0]}\) and \({PAIR[1]}\) have "
                        rf"equal median alignment depth")

    # A regime with clearly lower depth than FN and clearly higher
    # AUPRC is direct evidence against the depth account. Found here
    # rather than asserted in the prose.
    shallower = [q for q in QUADRANTS
                 if q != PAIR[0]
                 and cells[q]["median_depth_min"]
                 < fn["median_depth_min"] * 0.9
                 and cells[q]["median_auprc"] > fn["median_auprc"]]

    lines = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{Alignment depth by familiarity regime, with "
        rf"pocket decodability for comparison. Depth is the minimum "
        rf"over a system's chains of its alignment's sequence count, "
        rf"recorded with a cap of {ceiling:,}; the ceiling column "
        rf"gives the fraction of each regime's systems at that cap, "
        rf"whose true depth is unknown. AUPRC is the median at the "
        rf"peak layer of the \texttt{{{args.primary}}} feature set "
        rf"(layer {pf['layer']}). Of the pair similar on protein "
        rf"length, the system-level pocket measure and positive "
        rf"rate, {depth_phrase}, while their median AUPRC differs by "
        rf"a factor of {auprc_ratio:.1f}.}}",
        r"\label{tab:msa-depth}",
        r"\begin{tabular}{l r r r r}", r"\toprule",
        r"Regime & \(n\) & median depth & at ceiling & median AUPRC \\",
        r"\midrule",
    ]
    for q in QUADRANTS:
        c = cells[q]
        lines.append(
            rf"\({q}\) & {c['n']} & {c['median_depth_min']:,.0f} & "
            rf"{c['frac_at_ceiling']:.2f} & {c['median_auprc']:.3f} "
            rf"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "msa_depth.tex").write_text("\n".join(lines) + "\n")

    # The correlations across feature sets, for the prose.
    others = [f for f in feats if f != args.primary]
    # Single backslashes: in an rf-string "\\t" is two characters,
    # and LaTeX would see \\texttt rather than \texttt.
    corr_other = "; ".join(
        f"{d['by_feature'][f]['spearman_all']:.3f} and "
        f"{d['by_feature'][f]['spearman_uncapped']:.3f} for "
        + chr(92) + "texttt{" + f + "}" for f in others)

    m = [r"% Generated. Do not edit; re-run the generator.",
         rf"\newcommand{{\MsaN}}{{{pf['n']}}}",
         rf"\newcommand{{\MsaCeiling}}{{{ceiling:,}}}",
         rf"\newcommand{{\MsaNCeiling}}{{{pf['n_at_ceiling']}}}",
         rf"\newcommand{{\MsaNUncapped}}{{{pf['n_uncapped']}}}",
         rf"\newcommand{{\MsaRhoAll}}{{{pf['spearman_all']:.3f}}}",
         rf"\newcommand{{\MsaRhoUncapped}}{{"
         rf"{pf['spearman_uncapped']:.3f}}}",
         rf"\newcommand{{\MsaRhoOthers}}{{{corr_other}}}",
         rf"\newcommand{{\MsaFnDepth}}{{"
         rf"{fn['median_depth_min']:,.0f}}}",
         rf"\newcommand{{\MsaNfDepth}}{{"
         rf"{nf['median_depth_min']:,.0f}}}",
         rf"\newcommand{{\MsaFnCeil}}{{{fn['frac_at_ceiling']:.2f}}}",
         rf"\newcommand{{\MsaNfCeil}}{{{nf['frac_at_ceiling']:.2f}}}",
         rf"\newcommand{{\MsaAuprcRatio}}{{{auprc_ratio:.1f}}}",
         rf"\newcommand{{\MsaLayer}}{{{pf['layer']}}}"]
    if shallower:
        q = min(shallower, key=lambda x: cells[x]["median_depth_min"])
        bs = chr(92)
        m += [f"{bs}newcommand{{{bs}MsaShallowRegime}}"
              f"{{{bs}({q}{bs})}}",
              rf"\newcommand{{\MsaShallowDepth}}{{"
              rf"{cells[q]['median_depth_min']:,.0f}}}",
              rf"\newcommand{{\MsaShallowAuprc}}{{"
              rf"{cells[q]['median_auprc']:.3f}}}"]
    (outdir / "msa_macros.tex").write_text("\n".join(m) + "\n")

    print(f"primary {args.primary}, layer {pf['layer']}, "
          f"{pf['n']} systems")
    print(f"at ceiling {pf['n_at_ceiling']}, uncapped "
          f"{pf['n_uncapped']}")
    print(f"rho all {pf['spearman_all']:+.3f}, uncapped "
          f"{pf['spearman_uncapped']:+.3f}")
    print(f"{PAIR[0]} depth {fn['median_depth_min']:,.0f} vs "
          f"{PAIR[1]} {nf['median_depth_min']:,.0f}  "
          f"(ratio {max(depth_ratio, 1 / depth_ratio):.2f}), "
          f"AUPRC ratio {auprc_ratio:.1f}")
    if shallower:
        print(f"regimes shallower than {PAIR[0]} yet scoring higher: "
              f"{shallower}")
    else:
        print(f"no regime is both shallower than {PAIR[0]} and "
              f"higher-scoring")
    print(f"\nWrote 2 files to {outdir}")
    print("\n  Depth is one summary of an alignment. A null here "
          "excludes median depth\n  as the explanation, not the "
          "alignments, and the macros carry the ceiling\n  "
          "fraction so the paragraph cannot state the result without "
          "its limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
