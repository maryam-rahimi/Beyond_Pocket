#!/usr/bin/env python3
"""
Generate the pocket-axis (S_p) tables for the Beyond Pocket manuscript.

Reads the per-ligand pocket-similarity scores and the target table, and
emits three tables. As with the dataset tables, every number is computed
here rather than transcribed, so a change of release, cutoff or metric
propagates to the manuscript by rerunning.

TABLES
------
  sp_metrics      distribution and cross-correlation of the three candidate
                  pocket metrics, which is what exposes the ceiling-mass
                  problem in pocket_qcov
  sp_quadrants    quadrant counts across candidate S_p thresholds under one
                  working metric and aggregation, in systems and distinct
                  family clusters
  sp_metric_choice  quadrant counts by metric and aggregation, used to
                  compare candidate definitions and identify the weakest
                  supported cell under each

INPUT
-----
  sp_per_ligand.parquet    system_id, query_ligand_id, sp_qcov, sp_fident,
                           sp_pli  -- each the max against the pre-cutoff
                           pool, computed per LIGAND so that the system-level
                           aggregation remains a choice rather than a
                           hard-coded assumption
  targets_boltz2.parquet   fam_f and family_cluster from build_quadrants.py

USAGE
-----
    python3 generate_sp_tables.py \\
        --per-ligand "03 Results/sp_per_ligand.parquet" \\
        --targets    "03 Results/targets_boltz2.parquet" \\
        --outdir     "03 Results/sp_tables"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

QUADRANTS = ["FF", "FN", "NF", "NN"]

METRICS = [
    ("sp_qcov", r"\texttt{pocket\_qcov}", "pocket_qcov"),
    ("sp_fident", r"\texttt{pocket\_fident\_qcov}", "pocket_fident_qcov"),
    ("sp_pli", r"\texttt{pli\_qcov}", "pli_qcov"),
]

# Thresholds reported in the sweep. 50, 70, 95 and 100 are the values
# PLINDER itself uses to build similarity graphs for its splits, so they
# carry external justification rather than being chosen here; 80 and 90 are
# included to show the behaviour between them. 100 is retained because a
# substantial share of targets fall below it, so it is informative rather
# than a degenerate endpoint.
THRESHOLDS = [50, 70, 80, 90, 95, 100]

# The metric and aggregation used for the threshold-sweep table. This is a
# WORKING CHOICE for that table, not a conclusion: the metric-choice table
# below reports all combinations neutrally, and which definition the
# manuscript adopts should follow from those numbers rather than from a
# constant set here in advance.
SWEEP_METRIC = "sp_qcov"
SWEEP_AGG = "min"


def quadrant_counts(targets, sp_series, threshold):
    """Systems and distinct family clusters per quadrant."""
    m = targets.merge(sp_series.rename("S_p").reset_index(), on="system_id")
    q = (m["fam_f"].map({True: "F", False: "N"}).astype(str)
         + (m["S_p"] >= threshold).map({True: "F", False: "N"}).astype(str))
    return {k: {"systems": int((q == k).sum()),
                "clusters": int(m.loc[q == k, "family_cluster"].nunique())}
            for k in QUADRANTS}, int(len(m))


def table_metrics(per_ligand):
    """Distribution and correlation of the candidate metrics.

    The percentile columns are the point of this table. A metric whose
    median sits at the ceiling cannot support the continuous familiarity
    regression, however well it serves the binary split, and that trade-off
    is invisible from the quadrant counts alone.

    Three distributions are reported per metric, not one. The scores are
    stored per LIGAND, but S_p is a system-level quantity, and the two are
    different random variables: a system whose ligands score {100, 42}
    contributes both values to the per-ligand distribution while its
    system-level minimum is 42. A ceiling-mass claim about whichever axis is
    adopted therefore has to be made against the AGGREGATED distribution. Minimum
    and maximum are both shown because they bracket the aggregation choice,
    and the gap between them is exactly what that choice is worth.
    """
    cols = [c for c, _, _ in METRICS if c in per_ligand.columns]
    frames = {"per-ligand": per_ligand[cols]}
    for agg in ("min", "max"):
        frames[f"system ({agg})"] = per_ligand.groupby(
            "system_id")[cols].agg(agg)

    desc = {k: v.describe(percentiles=[.1, .25, .5, .75, .9]).T
            for k, v in frames.items()}
    corr = frames["per-ligand"].corr()

    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Candidate pocket-similarity metrics, each taken as the "
        r"maximum against the pre-cutoff pool. All are directed, "
        r"pocket-level, and reported by PLINDER as integer percentages. "
        r"Scores are stored per ligand, while $S_p$ is a system-level "
        r"quantity, so each metric is shown three ways: the raw per-ligand "
        r"distribution and the system-level distribution under minimum and "
        r"maximum aggregation. The percentiles matter as much as the means, "
        r"since a metric whose median sits at the ceiling can separate the "
        r"binary quadrants while leaving the continuous familiarity "
        r"regression with little resolution.}",
        r"\label{tab:sp_metrics}",
        r"\begin{tabular}{l l r r r r r r}", r"\toprule",
        r"metric & level & mean & 10\% & 25\% & median & 75\% & 90\% \\",
        r"\midrule",
    ]
    md = ["## Candidate pocket-similarity metrics", "",
          "| metric | level | mean | 10% | 25% | median | 75% | 90% |",
          "|---|---|---|---|---|---|---|---|"]
    for key, tex_name, md_name in METRICS:
        if key not in cols:
            continue
        for li, (level, dd) in enumerate(desc.items()):
            r = dd.loc[key]
            label_tex = tex_name if li == 0 else ""
            label_md = f"`{md_name}`" if li == 0 else ""
            tex.append(f"{label_tex} & {level} & {r['mean']:.1f} & "
                       f"{r['10%']:.0f} & {r['25%']:.0f} & {r['50%']:.0f} & "
                       f"{r['75%']:.0f} & {r['90%']:.0f} " + r"\\")
            md.append(f"| {label_md} | {level} | {r['mean']:.1f} | "
                      f"{r['10%']:.0f} | {r['25%']:.0f} | {r['50%']:.0f} | "
                      f"{r['75%']:.0f} | {r['90%']:.0f} |")
        tex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    md += ["", "Pairwise correlation:", "",
           "| | " + " | ".join(m for _, _, m in METRICS if m) + " |",
           "|---|" + "---|" * len(cols)]
    for key, _, md_name in METRICS:
        if key not in corr.index:
            continue
        md.append(f"| `{md_name}` | "
                  + " | ".join(f"{corr.loc[key, c]:.2f}" for c in cols)
                  + " |")
    md.append("")

    data = {"n_ligands": int(len(per_ligand)),
            "n_systems": int(per_ligand["system_id"].nunique()),
            "describe": {k: json.loads(v.to_json())
                         for k, v in desc.items()},
            "correlation_per_ligand": json.loads(corr.to_json())}
    return "\n".join(tex), "\n".join(md), data


def table_quadrants(per_ligand, targets):
    """Threshold sweep under the working metric and aggregation.

    Which definition the manuscript adopts is a question for
    table_metric_choice, not this one; this table shows how occupancy moves
    with the threshold holding the definition fixed.
    """
    sp = per_ligand.groupby("system_id")[SWEEP_METRIC].agg(SWEEP_AGG)
    rows, data = [], {}
    for t in THRESHOLDS:
        cells, n = quadrant_counts(targets, sp, t)
        data[t] = cells
        rows.append((t, cells))
    n_targets = n

    tex = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{Quadrant occupancy across candidate $S_p$ thresholds "
        rf"($n={n_targets:,}$ targets). $S_p$ is the maximum "
        rf"\texttt{{pocket\_qcov}} against any pre-cutoff system, aggregated "
        rf"over a system's ligands by minimum. Each cell gives systems and, "
        rf"after the slash, distinct receptor-family clusters; the cluster "
        rf"count governs statistical weight. The thresholds 50, 70 and 95 "
        rf"are those PLINDER uses to build its own similarity graphs.}}",
        r"\label{tab:sp_quadrants}",
        r"\begin{tabular}{r r r r r}", r"\toprule",
        r"threshold & FF & FN & NF & NN \\",
        r"\midrule",
    ]
    md = [f"## Quadrant occupancy by S_p threshold (n = {n_targets:,})", "",
          "systems / family clusters", "",
          "| threshold | FF | FN | NF | NN |", "|---|---|---|---|---|"]
    for t, cells in rows:
        f = lambda k: f"{cells[k]['systems']:,} / {cells[k]['clusters']:,}"
        tex.append(f"{t} & {f('FF')} & {f('FN')} & {f('NF')} & {f('NN')} "
                   + r"\\")
        md.append(f"| {t} | {f('FF')} | {f('FN')} | {f('NF')} | {f('NN')} |")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md.append("")
    return "\n".join(tex), "\n".join(md), {"n_targets": n_targets,
                                           "sweep": data}


def table_metric_choice(per_ligand, targets):
    """Weakest-cell comparison across metric and aggregation.

    Reporting the weakest cell rather than NN is deliberate: the four-cell
    comparison is only as strong as its thinnest cell. The table is
    descriptive -- it states how metric choice and ligand aggregation affect
    four-cell support, and leaves which definition to adopt to the
    manuscript.
    """
    rows, data = [], {}
    for key, tex_name, md_name in METRICS:
        if key not in per_ligand.columns:
            continue
        for agg in ("max", "min"):
            sp = per_ligand.groupby("system_id")[key].agg(agg)
            for t in (70, 90):
                cells, _ = quadrant_counts(targets, sp, t)
                cl = {k: cells[k]["clusters"] for k in QUADRANTS}
                weakest = min(cl.values())
                rows.append((tex_name, md_name, agg, t, cl, weakest))
                data[f"{key}|{agg}|{t}"] = {"clusters": cl,
                                            "weakest": weakest}

    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Family-cluster counts per quadrant by pocket metric, "
        r"system-level aggregation and threshold. The weakest cell is the "
        r"figure of merit, because the four-cell comparison is only as "
        r"strong as its thinnest cell. The comparison is descriptive: it "
        r"shows how metric choice and ligand aggregation affect four-cell "
        r"support, and which definition to adopt follows from these "
        r"numbers rather than being assumed.}",
        r"\label{tab:sp_metric_choice}",
        r"\begin{tabular}{l l r r r r r r}", r"\toprule",
        r"metric & agg. & thr. & FF & FN & NF & NN & weakest \\",
        r"\midrule",
    ]
    md = ["## Metric and aggregation comparison (family clusters)", "",
          "| metric | agg | thr | FF | FN | NF | NN | weakest |",
          "|---|---|---|---|---|---|---|---|"]
    for tex_name, md_name, agg, t, cl, weakest in rows:
        tex.append(f"{tex_name} & {agg} & {t} & {cl['FF']:,} & "
                   f"{cl['FN']:,} & {cl['NF']:,} & {cl['NN']:,} & "
                   + r"\textbf{" + f"{weakest:,}" + r"} \\")
        md.append(f"| `{md_name}` | {agg} | {t} | {cl['FF']:,} | "
                  f"{cl['FN']:,} | {cl['NF']:,} | {cl['NN']:,} | "
                  f"**{weakest:,}** |")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md.append("")
    return "\n".join(tex), "\n".join(md), data


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-ligand", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--outdir", default="sp_tables")
    args = ap.parse_args(argv)

    pl_path, tg_path = Path(args.per_ligand), Path(args.targets)
    for p in (pl_path, tg_path):
        if not p.exists():
            sys.exit(f"No such file: {p}")
    per_ligand = pd.read_parquet(pl_path)
    targets = pd.read_parquet(tg_path)

    for c in ("system_id", "query_ligand_id"):
        if c not in per_ligand.columns:
            sys.exit(f"per-ligand table lacks '{c}'")
    for c in ("system_id", "fam_f", "family_cluster"):
        if c not in targets.columns:
            sys.exit(f"target table lacks '{c}'")

    n_multi = (per_ligand.groupby("system_id")["query_ligand_id"]
               .nunique() > 1).mean()
    print(f"per-ligand rows : {len(per_ligand):,}")
    print(f"systems         : {per_ligand.system_id.nunique():,}")
    print(f"multi-ligand    : {n_multi:.1%}")
    print(f"sweep table uses: {SWEEP_METRIC}, {SWEEP_AGG} over ligands")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    record = {"sweep_metric": SWEEP_METRIC,
              "sweep_aggregation": SWEEP_AGG,
              "multi_ligand_fraction": float(n_multi)}
    md_all = ["# Pocket-axis (S_p) tables", ""]
    for name, fn in [("sp_metrics", lambda: table_metrics(per_ligand)),
                     ("sp_quadrants",
                      lambda: table_quadrants(per_ligand, targets)),
                     ("sp_metric_choice",
                      lambda: table_metric_choice(per_ligand, targets))]:
        print(f"  building {name}...")
        tex, md, data = fn()
        (outdir / f"{name}.tex").write_text(tex + "\n")
        (outdir / f"{name}.md").write_text(md + "\n")
        record[name] = data
        md_all += [md, ""]

    (outdir / "all_tables.md").write_text("\n".join(md_all) + "\n")
    (outdir / "sp_tables.json").write_text(json.dumps(record, indent=2))

    print(f"\nWrote to {outdir}/:")
    for f in sorted(outdir.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
