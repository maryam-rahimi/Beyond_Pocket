#!/usr/bin/env python3
"""
Generate the dataset-feasibility tables for the Beyond Pocket manuscript.

Every number in these tables is computed here from PLINDER, not transcribed
from a terminal session. That matters more than it sounds: the three tables
report counts that change with the release, the cutoff and the quality
filter, and a hand-copied figure that silently goes stale is the failure
mode this script exists to prevent. Regenerate after any change to the
release or the cutoff and the manuscript follows.

TABLES
------
  pool          evaluation pool by trunk cutoff
  sp_candidates candidate sources for the pocket axis, with the group-size
                statistics that decide whether each can support a
                familiarity judgement at all
  provisional   quadrant counts under the Pfam stand-in for S_p, reported
                as systems AND distinct receptor-family clusters

Writes .tex (booktabs) and .md for each, plus a combined markdown file and
a JSON record of every figure.

INPUT
-----
Needs the per-target table written by build_quadrants.py, which carries the
S_f labels and receptor-family signatures. The quadrant table cannot be
rebuilt here without duplicating that script's chain-level join logic, and
duplicating it would risk the two drifting apart.

USAGE
-----
    python3 generate_dataset_tables.py \\
        --targets "03 Results/targets_boltz2.parquet" \\
        --outdir  "03 Results/dataset_tables"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import plinder.core as pc
except ImportError:
    sys.exit("plinder is required. On Newton use "
             "/apps/anaconda/anaconda-2024.10/bin/python")

# Documented structural cutoffs, each stated in that model's own report.
CUTOFF_BOLTZ2 = "2023-06-01"
CUTOFF_EARLIER = "2021-09-30"   # AF3, Boltz-1, Chai-1, Protenix

QUADRANTS = ["FF", "FN", "NF", "NN"]

# Candidate pocket-axis sources. The membership table is included even
# though it fails, because the manuscript's argument depends on showing why
# it fails rather than on quietly omitting it.
SP_SOURCES = [
    ("ligand_pocket_membership", "membership"),
    ("system_pocket_CATH", "annotation"),
    ("system_pocket_SCOP2", "annotation"),
    ("system_pocket_Pfam", "annotation"),
]


def system_dates():
    """One row per system with its entry's release date."""
    ann = pc.query_table("annotation",
                         columns=["system_id", "entry_pdb_id"]
                         ).drop_duplicates("system_id")
    meta = pc.query_table("entry_metadata",
                          columns=["entry_pdb_id", "entry_release_date"])
    df = ann.merge(meta, on="entry_pdb_id", how="left")
    df["d"] = pd.to_datetime(df["entry_release_date"], errors="coerce")
    return df


def table_pool(sysdf):
    """Pool sizes at each candidate cutoff.

    The boundary is exclusive: a system released ON the cutoff date counts
    as pre-cutoff, because a model reported as trained on structures "up to"
    that date has the boundary day inside its training set.
    """
    rows = {}
    for c in (CUTOFF_EARLIER, CUTOFF_BOLTZ2):
        t = pd.Timestamp(c)
        valid = sysdf["d"].notna()
        rows[c] = {
            "pre": int((valid & (sysdf["d"] <= t)).sum()),
            "post": int((valid & (sysdf["d"] > t)).sum()),
        }
    data = {
        "total_systems": int(len(sysdf)),
        "unparseable_dates": int(sysdf["d"].isna().sum()),
        "by_cutoff": rows,
    }

    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Evaluation pool by trunk training cutoff. The "
        + CUTOFF_EARLIER.replace("-", "--")
        + r" column is shown for reference: it is the cutoff used by "
          r"several contemporaneous co-folding models, and the comparison "
          r"bounds what the later Boltz-2 cutoff costs in available data. "
          r"The boundary is exclusive, so a system released on the cutoff "
          r"date counts as pre-cutoff.}",
        r"\label{tab:pool}",
        r"\begin{tabular}{l r r}", r"\toprule",
        rf" & cutoff {CUTOFF_EARLIER} & cutoff {CUTOFF_BOLTZ2} \\",
        r"\midrule",
        rf"PLINDER systems, total & \multicolumn{{2}}{{c}}"
        rf"{{{data['total_systems']:,}}} \\",
        rf"pre-cutoff (reference pool) & {rows[CUTOFF_EARLIER]['pre']:,} & "
        rf"{rows[CUTOFF_BOLTZ2]['pre']:,} \\",
        rf"post-cutoff (candidates) & {rows[CUTOFF_EARLIER]['post']:,} & "
        rf"{rows[CUTOFF_BOLTZ2]['post']:,} \\",
        rf"unparseable release dates & \multicolumn{{2}}{{c}}"
        rf"{{{data['unparseable_dates']:,}}} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    md = [
        "## Evaluation pool by trunk cutoff", "",
        f"| | cutoff {CUTOFF_EARLIER} | cutoff {CUTOFF_BOLTZ2} |",
        "|---|---|---|",
        f"| PLINDER systems, total | {data['total_systems']:,} | "
        f"{data['total_systems']:,} |",
        f"| pre-cutoff (reference) | {rows[CUTOFF_EARLIER]['pre']:,} | "
        f"{rows[CUTOFF_BOLTZ2]['pre']:,} |",
        f"| post-cutoff (candidates) | {rows[CUTOFF_EARLIER]['post']:,} | "
        f"{rows[CUTOFF_BOLTZ2]['post']:,} |",
        f"| unparseable dates | {data['unparseable_dates']:,} | "
        f"{data['unparseable_dates']:,} |", "",
    ]
    return "\n".join(tex), "\n".join(md), data


def _group_stats(labels):
    """Coverage, group count, mean group size, singleton fraction.

    `labels` is one label per system, NaN where the annotation is absent.
    Mean size near 1 with a high singleton fraction means the grouping is a
    deduplication rather than a similarity clustering, and cannot support a
    familiarity judgement: almost nothing shares a group with anything, so
    almost nothing is ever familiar.
    """
    nn = labels.notna()
    sz = labels[nn].value_counts()
    return {
        "coverage": float(nn.mean()),
        "groups": int(len(sz)),
        "mean_size": float(sz.mean()) if len(sz) else float("nan"),
        "singleton_frac": float((sz == 1).mean()) if len(sz) else float("nan"),
    }


def table_sp_candidates(sysdf):
    """Group statistics for each candidate pocket axis."""
    ann_cols = [c for t, kind in SP_SOURCES if kind == "annotation"
                for c in [t]]
    ann = pc.query_table("annotation",
                         columns=["system_id"] + ann_cols
                         ).drop_duplicates("system_id")

    stats = {}
    for name, kind in SP_SOURCES:
        if kind == "membership":
            pm = pc.query_table(name,
                                columns=["system_id",
                                         "representative_system_id"]
                                ).drop_duplicates()
            # Group sizes come from the membership table directly rather
            # than from one representative per system. A system may carry
            # several pocket representatives -- the quadrant code allows
            # that and resolves it with `any` -- so collapsing with
            # `first()` would both discard memberships and misstate how
            # large the groups are.
            sz = pm.groupby("representative_system_id")[
                "system_id"].nunique()
            covered = pm["system_id"].nunique()
            s = {
                "coverage": covered / len(sysdf),
                "groups": int(len(sz)),
                "mean_size": float(sz.mean()) if len(sz) else float("nan"),
                "singleton_frac": (float((sz == 1).mean()) if len(sz)
                                   else float("nan")),
                "self_representing": float(
                    (pm["system_id"] == pm["representative_system_id"]
                     ).mean()),
                "mean_reps_per_system": float(len(pm) / covered)
                if covered else float("nan"),
            }
        else:
            s = _group_stats(ann.set_index("system_id").reindex(
                sysdf["system_id"])[name])
        stats[name] = s

    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Candidate sources for the pocket axis. Coverage is the "
        r"fraction of systems carrying the annotation; group size and "
        r"singleton fraction describe whether the grouping can support a "
        r"familiarity judgement at all. A mean size near one, with most "
        r"groups singletons, marks a deduplication rather than a similarity "
        r"clustering.}",
        r"\label{tab:sp_candidates}",
        r"\begin{tabular}{l r r r r}", r"\toprule",
        r"source & coverage & groups & mean size & singletons \\",
        r"\midrule",
    ]
    md = ["## Candidate sources for the pocket axis", "",
          "| source | coverage | groups | mean size | singletons |",
          "|---|---|---|---|---|"]
    for name, _ in SP_SOURCES:
        s = stats[name]
        esc = name.replace("_", r"\_")
        tex.append(
            rf"\texttt{{{esc}}} & {s['coverage']:.1%} & {s['groups']:,} & "
            rf"{s['mean_size']:.2f} & {s['singleton_frac']:.1%} \\".replace(
                "%", r"\%"))
        md.append(f"| `{name}` | {s['coverage']:.1%} | {s['groups']:,} | "
                  f"{s['mean_size']:.2f} | {s['singleton_frac']:.1%} |")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md.append("")
    return "\n".join(tex), "\n".join(md), stats


def table_provisional(sysdf, targets, cutoff):
    """Quadrant counts with Pfam standing in for S_p.

    Reported as systems AND distinct receptor-family clusters. Systems are
    not independent evaluation units -- many share a PDB entry, many entries
    share a family -- so the cluster count is what governs the statistical
    weight a cell carries, and the two can differ by nearly an order of
    magnitude.
    """
    t = pd.Timestamp(cutoff)
    pf = pc.query_table("annotation",
                        columns=["system_id", "system_pocket_Pfam"]
                        ).drop_duplicates("system_id")
    dated = pf.merge(sysdf[["system_id", "d"]], on="system_id", how="left")
    pre = set(dated.loc[dated["d"].notna() & (dated["d"] <= t),
                        "system_pocket_Pfam"].dropna())

    m = targets.merge(pf, on="system_id", how="left")
    m["fam_p"] = m["system_pocket_Pfam"].map(
        lambda v: (v in pre) if pd.notna(v) else None)
    x = m[m["fam_p"].notna()].copy()
    x["q"] = (x["fam_f"].map({True: "F", False: "N"}).astype(str)
              + x["fam_p"].map({True: "F", False: "N"}).astype(str))

    cells = {q: {"systems": int((x["q"] == q).sum()),
                 "clusters": int(x.loc[x["q"] == q,
                                       "family_cluster"].nunique())}
             for q in QUADRANTS}
    n = int(len(x))
    off_diag = cells["FN"]["systems"] + cells["NF"]["systems"]

    tex = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{Provisional quadrant counts, Pfam standing in for "
        rf"\(S_p\). Targets are post-cutoff, validated, and carry a Pfam "
        rf"annotation ($n={n:,}$). Systems are reported alongside distinct "
        rf"receptor-family clusters, because systems are not independent "
        rf"evaluation units: many share a PDB entry and many entries share "
        rf"a family. The cluster count is what governs the weight a cell "
        rf"carries.}}",
        r"\label{tab:provisional}",
        r"\begin{tabular}{l r r}", r"\toprule",
        r" & pocket novel & pocket familiar \\",
        r" & systems / clusters & systems / clusters \\",
        r"\midrule",
        rf"family novel & \(NN\) {cells['NN']['systems']:,} / "
        rf"{cells['NN']['clusters']:,} & \(NF\) "
        rf"{cells['NF']['systems']:,} / {cells['NF']['clusters']:,} \\",
        rf"family familiar & \(FN\) {cells['FN']['systems']:,} / "
        rf"{cells['FN']['clusters']:,} & \(FF\) "
        rf"{cells['FF']['systems']:,} / {cells['FF']['clusters']:,} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    md = [
        "## Provisional quadrants (Pfam stands in for S_p)", "",
        f"n = {n:,} targets", "",
        "| | pocket novel | pocket familiar |", "|---|---|---|",
        f"| **family novel** | NN {cells['NN']['systems']:,} / "
        f"{cells['NN']['clusters']:,} clusters | NF "
        f"{cells['NF']['systems']:,} / {cells['NF']['clusters']:,} |",
        f"| **family familiar** | FN {cells['FN']['systems']:,} / "
        f"{cells['FN']['clusters']:,} | FF {cells['FF']['systems']:,} / "
        f"{cells['FF']['clusters']:,} |", "",
        f"Off-diagonal (axes disagree): {off_diag:,} targets.", "",
    ]
    data = {"n_targets": n, "cells": cells, "off_diagonal": off_diag,
            "note": "PROVISIONAL: Pfam is a domain family, not a pocket "
                    "similarity measure."}
    return "\n".join(tex), "\n".join(md), data


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True,
                    help="per-target parquet/csv from build_quadrants.py")
    ap.add_argument("--cutoff", default=CUTOFF_BOLTZ2)
    ap.add_argument("--outdir", default="dataset_tables")
    args = ap.parse_args(argv)

    p = Path(args.targets)
    if not p.exists():
        sys.exit(f"No such file: {p}\nRun build_quadrants.py --write-table "
                 f"first.")
    targets = (pd.read_parquet(p) if p.suffix == ".parquet"
               else pd.read_csv(p))
    for col in ("system_id", "fam_f", "family_cluster", "release_date"):
        if col not in targets.columns:
            sys.exit(f"Target table lacks '{col}'. It must come from "
                     f"build_quadrants.py --write-table.")

    print(f"targets: {len(targets):,} from {p.name}")

    # --- verify the claims the captions will make ----------------------
    # The captions assert that targets are post-cutoff and validated. Both
    # are properties of how the target table was BUILT, and neither is
    # recoverable from its filename. Checking them here against PLINDER and
    # against the table's own dates closes a reproducibility trap: a table
    # built with --no-validation-filter, or at a different cutoff, would
    # otherwise be typeset under a caption asserting the opposite.
    cut = pd.Timestamp(args.cutoff)
    td = pd.to_datetime(targets["release_date"], errors="coerce")
    n_nodate = int(td.isna().sum())
    n_wrongside = int((td.notna() & (td <= cut)).sum())
    if n_nodate or n_wrongside:
        sys.exit(
            f"Target/cutoff mismatch for {args.cutoff}: "
            f"{n_nodate:,} target(s) with no release date and "
            f"{n_wrongside:,} not strictly after the cutoff.\n"
            f"The target table was probably built at a different cutoff. "
            f"Rebuild it with\nbuild_quadrants.py --cutoff "
            f"{args.cutoff}, or pass the cutoff it was built at.")

    val = pc.query_table(
        "annotation",
        columns=["system_id", "system_pass_validation_criteria"]
    ).drop_duplicates("system_id")
    chk = targets[["system_id"]].merge(val, on="system_id", how="left")
    n_bad = int((~chk["system_pass_validation_criteria"].fillna(False)).sum())
    if n_bad:
        sys.exit(
            f"{n_bad:,} of {len(targets):,} targets fail or lack "
            f"system_pass_validation_criteria.\nThe provisional table's "
            f"caption asserts the targets are validated, so this script "
            f"refuses to\nwrite it. Rebuild the target table without "
            f"--no-validation-filter, or amend the caption.")
    print(f"  verified: all post-{args.cutoff}, all validated")

    sysdf = system_dates()
    print(f"systems: {len(sysdf):,}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    record, md_all = {}, ["# Dataset feasibility tables", ""]
    for name, fn in [("pool", lambda: table_pool(sysdf)),
                     ("sp_candidates", lambda: table_sp_candidates(sysdf)),
                     ("provisional",
                      lambda: table_provisional(sysdf, targets,
                                                args.cutoff))]:
        print(f"  building {name}...")
        tex, md, data = fn()
        (outdir / f"{name}.tex").write_text(tex + "\n")
        (outdir / f"{name}.md").write_text(md + "\n")
        record[name] = data
        md_all += [md, ""]

    (outdir / "all_tables.md").write_text("\n".join(md_all) + "\n")
    record["cutoff"] = args.cutoff
    record["targets_file"] = str(p)
    (outdir / "dataset_tables.json").write_text(
        json.dumps(record, indent=2))

    print(f"\nWrote to {outdir}/:")
    for f in sorted(outdir.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
