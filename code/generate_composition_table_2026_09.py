#!/usr/bin/env python3
"""
Generate the FN-versus-NF composition table from the corrected 2026-09
composition analysis.

This generator intentionally does NOT read the old index_coverage.json.
The corrected composition analysis is fail-closed on release provenance and
requires every frozen test system to be present in the supplied study release,
so there is no complete-case coverage caveat to propagate.

USAGE
-----
    python3 generate_composition_table_2026_09.py \
        --composition fn_composition_2026_09.json \
        --outdir composition_tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUADRANTS = ("FF", "FN", "NF", "NN")

# Primary corrected set: three continuous quantities + four ligand flags.
# ligand_is_oligo is deliberately absent because the 2026-09 schema has
# three distinct oligo-related flags rather than the preregistered single flag.
ROW_ORDER = (
    "biounit_num_ligands",
    "ligand_num_atoms_with_crystal_contacts",
    "ligand_num_pocket_residues",
    "ligand_is_cofactor",
    "ligand_is_ion",
    "ligand_is_fragment",
    "ligand_is_covalent",
)


def tex(s: str) -> str:
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        s = s.replace(a, b)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--composition", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    comp = json.loads(Path(args.composition).read_text())
    meta = comp.get("_meta")
    if meta is None:
        sys.exit(f"{Path(args.composition).name} has no _meta block")

    dropped = meta.get("n_systems_dropped")
    if dropped not in (0, None):
        sys.exit(
            f"{Path(args.composition).name} reports {dropped} dropped systems. "
            "The corrected 2026-09 analysis must have complete frozen-test-set "
            "coverage; refusing to generate a complete-case table."
        )

    rows = [r for r in ROW_ORDER if r in comp]
    unexpected = [r for r in comp if r != "_meta" and r not in ROW_ORDER]
    if unexpected:
        sys.exit(
            f"{Path(args.composition).name} carries comparisons not in this "
            f"table's fixed row order: {unexpected}."
        )

    if len(rows) != meta.get("n_comparisons"):
        sys.exit(
            f"the file reports {meta.get('n_comparisons')} comparisons but "
            f"{len(rows)} primary rows are present"
        )
    if len(rows) != 7:
        sys.exit(f"expected 7 corrected primary comparisons, found {len(rows)}")

    omitted = meta.get("originally_planned_but_unavailable", [])
    if "ligand_is_oligo" not in omitted:
        sys.exit(
            "corrected metadata does not record ligand_is_oligo as originally "
            "planned but unavailable"
        )

    degenerate = [
        r for r in rows
        if len(set(comp[r]["cells"].values())) == 1
    ]
    live = [r for r in rows if r not in degenerate]
    sep = [r for r in live if comp[r]["excludes_zero"]]

    deg_note = ""
    if degenerate:
        names = ", ".join(tex(comp[r]["label"]) for r in degenerate)
        deg_note = (
            rf" {names.capitalize()} takes the same value in every regime and "
            rf"is reported but not counted among non-degenerate comparisons."
        )

    verdict = (
        rf"{len(sep)} of {len(live)} non-degenerate comparisons have "
        rf"cluster-bootstrap intervals excluding zero.{deg_note} "
        rf"No multiplicity correction is applied; these differences are "
        rf"candidate compositional confounders, not causal mechanisms."
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Composition of the four familiarity regimes using four "
        rf"ligand-type flags and three structural quantities fixed before "
        rf"inspection of the corrected 2026-09 results. The primary contrast "
        rf"is \(FN-NF\); intervals come from {meta['n_boot']} resamples of "
        rf"receptor-family clusters. The earlier single oligo flag is omitted "
        rf"because that variable does not exist in the study-release schema "
        rf"and was not replaced post hoc. FN and NF are similar on the "
        rf"system-level design-factor pocket measure, but the per-ligand pocket-"
        rf"residue quantity tested here is a different estimand. {verdict}}}",
        r"\label{tab:fn-composition}",
        r"\begin{tabular}{l rrrr r l}",
        r"\toprule",
        r"Quantity & \(FF\) & \(FN\) & \(NF\) & \(NN\) & "
        r"\(FN-NF\) & 95\% interval \\",
        r"\midrule",
    ]

    for r in rows:
        d = comp[r]
        c = d["cells"]
        mark = r"$^{\star}$" if d["excludes_zero"] else ""
        lines.append(
            f"{tex(d['label'])}{mark} & "
            + " & ".join(f"{c[q]:.3f}" for q in QUADRANTS)
            + f" & {d['fn_minus_nf']:+.3f} & "
            + f"[{d['lo']:+.3f}, {d['hi']:+.3f}] \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tex_path = outdir / "fn_composition_2026_09.tex"
    tex_path.write_text("\n".join(lines) + "\n")

    md = [
        "## Composition of the four regimes (2026-09 rerun)",
        "",
        "| quantity | FF | FN | NF | NN | FN-NF | interval |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        d = comp[r]
        c = d["cells"]
        md.append(
            f"| {d['label']} | "
            + " | ".join(f"{c[q]:.3f}" for q in QUADRANTS)
            + f" | {d['fn_minus_nf']:+.3f} | "
            + f"[{d['lo']:+.3f}, {d['hi']:+.3f}] |"
        )

    md += [
        "",
        f"Separating non-degenerate comparisons: "
        f"{', '.join(comp[r]['label'] for r in sep) if sep else 'none'}.",
        "",
        "The preregistered single `ligand_is_oligo` flag is omitted because "
        "it is unavailable in the 2026-09 study-release schema and was not "
        "redefined post hoc.",
    ]
    md_path = outdir / "fn_composition_summary_2026_09.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"primary comparisons: {len(rows)}")
    print(f"non-degenerate: {len(live)}")
    print(f"interval excludes zero: {[comp[r]['label'] for r in sep]}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
