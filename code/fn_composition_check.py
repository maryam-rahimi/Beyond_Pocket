#!/usr/bin/env python3
"""
Ask whether the FN and NF cells differ on four pre-specified
properties of their structures.

WHY THESE FOUR, AND WHY NOW
---------------------------
FN and NF are matched on protein length, pocket size, positive rate
and alignment depth, and differ 2.6-fold in median AUPRC. Each of
those four has been tested and none explains the gap. The four below
probe qualitatively different aspects of the binding-site problem
rather than further variants of the same covariates.

  LIGAND COUNT. zPocket reports that chains holding more than one
  scored pocket are harder, their best head falling from 0.751 to
  0.599. If FN structures hold more ligands, that is a mechanism with
  precedent rather than a new guess.

  LIGAND TYPE. A cofactor site, an ion site and a drug-like site are
  different problems. Enrichment for one kind would mean "novel
  pocket" partly stands for "different kind of pocket".

  CRYSTAL CONTACTS. A ligand touching neighbouring molecules in the
  lattice may be sitting in a crystallisation artefact rather than a
  binding site, which would make the label itself suspect.

  POCKET RESIDUES PER LIGAND. The system-level median is matched, but
  the per-ligand distribution need not be.

PRE-SPECIFICATION
-----------------
The comparisons, the flags and the aggregations are fixed in this file
rather than passed in, so they cannot be chosen after seeing a result.
LIGAND_FLAGS is one comparison family that expands into four numbers,
and that is stated rather than counted as one. All of them are
reported whatever they show.

AGGREGATION, DECLARED IN ADVANCE
--------------------------------
  ligand count      already per system
  ligand flags      fraction of SYSTEMS having any ligand of that
                    type, not a fraction of ligand rows, so a
                    multi-ligand system carries no extra weight
  crystal contacts  MAXIMUM across a system's ligands, since the
                    question is whether ANY scored site may be
                    compromised
  pocket residues   MEDIAN across ligands, matching how the system
                    label is formed elsewhere

UNCERTAINTY
-----------
Intervals come from resampling receptor-family clusters, as in the
main bootstrap: proteins within a family share sequence and fold, so
resampling them individually would give intervals that are too narrow.
The estimate is a difference of medians for continuous quantities and
of proportions for the flags.

WHAT A DIFFERENCE WOULD AND WOULD NOT MEAN
------------------------------------------
That the covariate is a candidate explanation or confounder for the
performance gap. Not that it caused the 2.6-fold difference. Seven
numbers are reported without multiplicity correction and are
exploratory; an interval excluding zero is an interval.

Two further quantities are not tested here. Whether the ligand is
flagged an artifact was considered after the comparison set was
fixed. The oligo flags are three columns where one was declared. Both
would turn a pre-specified analysis into a search.

USAGE
-----
    python3 fn_composition_check.py \\
        --index probe_index.parquet \\
        --out fn_composition.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUADRANTS = ("FF", "FN", "NF", "NN")

# Fixed here, not passed in. The flags are one comparison family that
# expands into four numbers; naming them in the file is what makes that
# auditable.
# Exactly the five declared before any result was seen. A sixth,
# ligand_is_artifact, was briefly added while writing this file
# because it seemed to bear on the crystal-contact question; it is
# removed, because a variable added after the plan is not
# pre-specified however reasonable it looks. It can be tested
# separately and labelled exploratory.
# ligand_is_oligo is ABSENT. It was declared with the other four, but
# the study's release carries no such column: it has
# ligand_is_oligosaccharide, ligand_is_oligonucleotide and
# ligand_is_oligopeptide as three separate flags. Choosing one of them
# now would be a post-hoc choice, and collapsing all three into "any
# oligo" would be a new operationalisation decided after seeing the
# schema. Either would turn a pre-specified comparison into one
# selected with the data in view, so the variable is dropped rather
# than substituted.
#
# An earlier run of this script reported a significant oligo
# difference. That run read the 2024-06 release, whose schema does
# carry the column; it is superseded, not merely recomputed.
#
# The three oligo flags can be examined separately and labelled
# exploratory. They do not belong in this set.
LIGAND_FLAGS = ("ligand_is_cofactor", "ligand_is_ion",
                "ligand_is_fragment", "ligand_is_covalent")
CONTINUOUS = {
    "biounit_num_ligands": ("first", "ligands in the structure"),
    "ligand_num_atoms_with_crystal_contacts":
        ("max", "ligand atoms with crystal contacts"),
    "ligand_num_pocket_residues": ("median", "pocket residues"),
}
PRIMARY = ("FN", "NF")


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--plinder-dir", required=True,
                    help="the release directory, e.g. "
                         "~/.local/share/plinder/2026-09. Read as "
                         "files: the installed package serves only "
                         "2024-06 and would substitute it silently, "
                         "which is how an earlier run of this script "
                         "reported 43%% coverage of a test set that "
                         "is wholly present in the study's own "
                         "release.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.n_boot <= 0:
        sys.exit("--n-boot must be positive")

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    idx = pd.read_parquet(args.index)
    for c in ("system_id", "quadrant", "split", "family_cluster"):
        if c not in idx.columns:
            sys.exit(f"--index has no '{c}'")
    test = idx[idx.split == "test"][
        ["system_id", "quadrant", "family_cluster"]]
    if test.family_cluster.isna().any():
        sys.exit("some test systems have no family_cluster; the "
                 "cluster bootstrap is undefined for them")
    print(f"test systems {len(test):,}, clusters "
          f"{test.family_cluster.nunique():,}")

    ann_p = (Path(args.plinder_dir).expanduser() / "index"
             / "annotation_table.parquet")
    if not ann_p.exists():
        sys.exit(f"{ann_p} does not exist. Check --plinder-dir names "
                 f"a release directory.")
    cols = ["system_id", *LIGAND_FLAGS, *CONTINUOUS]
    print(f"reading {ann_p} ...")
    pl = pd.read_parquet(ann_p, columns=cols)
    # Presence is decided by the merge, not by whether one field is
    # null. A system that matched PLINDER but has no ligand count is
    # present, and dropping it would discard its ligand flags and its
    # pocket residues too, while reporting it as "absent from the
    # index" -- two different facts told as one. Field-level nulls are
    # left in place; stat() drops them per variable.
    lig = test.merge(pl, on="system_id", how="left", indicator=True)
    absent_ids = lig.loc[lig["_merge"] == "left_only",
                         "system_id"].unique()
    missing = len(absent_ids)
    if missing:
        # Fatal, not a caveat. This analysis exists because an earlier
        # run read a release missing 57% of the test systems and
        # reported the result as a complete-case analysis. The whole
        # point of naming the release is that every frozen test system
        # is present in it; any absence means the wrong directory, not
        # a smaller sample.
        sys.exit(f"{missing:,} of {len(test):,} frozen test systems "
                 f"are absent from the supplied release, e.g. "
                 f"{list(absent_ids[:3])}.\nThis is a provenance "
                 f"failure. Refusing to continue as a complete-case "
                 f"analysis.")
    lig = lig.drop(columns="_merge")
    print(f"  all {len(test):,} test systems present")
    print(f"  {len(lig):,} ligand rows over "
          f"{lig.system_id.nunique():,} systems")

    # "first" is only meaningful if the field really is constant
    # within a system. Checked rather than assumed: if it varied, the
    # missingness test above could also retain part of a system and
    # change the median and maximum below.
    nuniq = lig.groupby("system_id")["biounit_num_ligands"].nunique(
        dropna=True)
    varying = nuniq[nuniq > 1]
    if len(varying):
        sys.exit(f"biounit_num_ligands varies within "
                 f"{len(varying)} systems, e.g. "
                 f"{varying.head(3).to_dict()}.\n'first' has no "
                 f"meaning for them; the aggregation must be "
                 f"re-declared.")

    # System-level values, by the aggregations declared above.
    g = lig.groupby(["system_id", "quadrant", "family_cluster"])
    per_sys = pd.DataFrame(index=g.size().index)
    for col, (how, _) in CONTINUOUS.items():
        per_sys[col] = (g[col].first() if how == "first"
                        else g[col].max() if how == "max"
                        else g[col].median())
    for f in LIGAND_FLAGS:
        # ANY ligand of that type, so a multi-ligand system is one vote
        per_sys[f] = g[f].max().astype(float)
    per_sys = per_sys.reset_index()

    rng = np.random.default_rng(args.seed)
    clusters = per_sys.family_cluster.unique()
    by_cl = {c: d for c, d in per_sys.groupby("family_cluster")}

    def stat(d, col, kind):
        s = d[col].dropna()
        if not len(s):
            return float("nan")
        return float(s.mean() if kind == "prop" else s.median())

    boot = {col: np.empty(args.n_boot)
            for col in list(CONTINUOUS) + list(LIGAND_FLAGS)}
    for b in range(args.n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        rep = pd.concat([by_cl[c] for c in drawn], ignore_index=True)
        a = rep[rep.quadrant == PRIMARY[0]]
        c_ = rep[rep.quadrant == PRIMARY[1]]
        for col in CONTINUOUS:
            boot[col][b] = (stat(a, col, "med") - stat(c_, col, "med"))
        for col in LIGAND_FLAGS:
            boot[col][b] = (stat(a, col, "prop")
                            - stat(c_, col, "prop"))

    results, rows = {}, []
    for col in list(CONTINUOUS) + list(LIGAND_FLAGS):
        kind = "prop" if col in LIGAND_FLAGS else "med"
        cells = {q: stat(per_sys[per_sys.quadrant == q], col, kind)
                 for q in QUADRANTS}
        est = cells[PRIMARY[0]] - cells[PRIMARY[1]]
        d = boot[col][~np.isnan(boot[col])]
        lo, hi = np.percentile(d, [2.5, 97.5])
        label = (CONTINUOUS[col][1] if col in CONTINUOUS
                 else col.replace("ligand_is_", "") + " fraction")
        results[col] = {
            "label": label, "aggregation":
                (CONTINUOUS[col][0] if col in CONTINUOUS
                 else "any ligand, per system"),
            "cells": {q: round(v, 4) for q, v in cells.items()},
            "fn_minus_nf": round(est, 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "excludes_zero": bool(lo > 0 or hi < 0)}
        rows.append((label, cells, est, lo, hi))

    print(f"\n{'feature':34}{'FF':>8}{'FN':>8}{'NF':>8}{'NN':>8}"
          f"{'FN-NF':>9}{'95% interval':>22}")
    for label, cells, est, lo, hi in rows:
        print(f"{label:34}" +
              "".join(f"{cells[q]:>8.3f}" for q in QUADRANTS) +
              f"{est:>+9.3f}" +
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}" +
              ("  *" if (lo > 0 or hi < 0) else ""))

    results["_meta"] = {
        "plinder_dir": str(Path(args.plinder_dir).expanduser()),
        "omitted_ligand_is_oligo":
            "declared with the other flags, but this release carries "
            "no such column: it provides ligand_is_oligosaccharide, "
            "ligand_is_oligonucleotide and ligand_is_oligopeptide as "
            "three separate flags. Not substituted post hoc. An "
            "earlier run against the 2024-06 release, whose schema "
            "does carry the column, reported a difference on it and "
            "is superseded.",
        "n_boot": args.n_boot, "seed": args.seed,
        "resampled": "family_cluster",
        "primary_contrast": "FN - NF",
        "n_comparisons": len(rows),
        "n_comparisons_primary": len(rows),
        "originally_planned_but_unavailable": ["ligand_is_oligo"],
        "multiplicity_correction": None,
        "n_systems_dropped": int(missing)}
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print(f"\n  {len(rows)} comparisons, no multiplicity correction. "
          f"An interval excluding\n  zero marks a candidate "
          f"explanation for the FN-NF gap, not its cause.")
    print("  FN and NF are already matched on length, pocket size, "
          "positive rate and\n  alignment depth, so a difference here "
          "is a difference in kind rather than\n  in evaluation "
          "geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
