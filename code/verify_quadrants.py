#!/usr/bin/env python3
"""
Re-derive the four familiarity regimes from source tables and check the
stored assignment against the result.

WHY THIS IS SEPARATE FROM build_quadrants.py
--------------------------------------------
Re-running the assignment code would agree with itself. This script
reconstructs both axes from the tables they were built from, without
importing or consulting that code, and derives each system's cell from
ITS OWN reconstruction -- never from the stored value it is checking.
That distinction is the whole point: a verifier that reconstructs a
quantity and then quietly derives its verdict from the stored one
certifies the assignment in precisely the case where the assignment is
wrong.

BOTH AXES, OR NO VERDICT
------------------------
S_p is reconstructed from the per-ligand similarity file. S_f is
reconstructed only when --chain-clusters and --all-systems are given;
without them the family axis is taken from the stored table, a wrong
family assignment would pass unseen, and the verdict is withheld
rather than reported as clean.

WHAT IS TESTED HERE
-------------------
  Whether the stored S_p equals the value re-derived by aggregating the
  per-ligand scores.

  Whether the stored cell equals the cell derived from the
  reconstructed S_p and the reconstructed S_f.

  Whether every target could be checked at all. A target with no row in
  the similarity file is reported as UNVERIFIABLE, never as agreeing:
  "compared and nothing matched" and "never compared" are
  indistinguishable from these tables, and counting the second as
  agreement would let a coverage gap read as a clean audit.

  Whether any target falls on or before the cutoff, which would place
  it inside its own reference pool.

  How strongly the two axes are associated. They measure different
  things by construction, but a strong association would mean the four
  cells are close to two.

RISKS CONSIDERED BUT NOT TESTED HERE
------------------------------------
These need the script that BUILT the per-ligand file. A clean run below
does not establish any of them.

  DIRECTION. qcov is coverage and therefore asymmetric: a small pocket
  inside a large one is fully covered one way and barely covered the
  other. The export is directed, so the wrong direction misassigns
  systematically rather than randomly. This is a property of the
  export's schema and must be read from the resource's documentation.

  COVERAGE. Whether every eligible target ligand was compared against
  the pre-cutoff pool at all. This script can count who is missing from
  the file; it cannot say why.

  SELF-COMPARISON. Whether a system was excluded as its own reference,
  and whether exclusion was by system or by parent entry. The
  per-ligand file carries no reference identity, so the maximising
  partner is not visible here.

  THE CUTOFF ACTUALLY USED UPSTREAM. --cutoff is what the operator
  asserts and is recorded as an assertion. If the per-ligand file was
  built against another date, everything below is consistent and the
  assignment is still wrong.

USAGE
-----
    python3 verify_quadrants.py \\
        --targets targets_with_sp_boltz1.parquet \\
        --per-ligand sp_per_ligand_boltz1.parquet \\
        --chain-clusters receptor_chain_clusters.parquet \\
        --all-systems all_systems_with_dates.parquet \\
        --cutoff 2021-09-30 \\
        --cutoff-source "Boltz-1 paper, structural training cutoff" \\
        --threshold 80 --out quadrant_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUADRANTS = ("FF", "FN", "NF", "NN")


def check_chain_table(chains: pd.DataFrame,
                      allsys: pd.DataFrame) -> dict:
    """Establish that the chain table lists EVERY receptor chain.

    The family rule calls a system novel only when every chain is
    resolved and none is familiar. "Every chain" is read from this
    table, so a table holding only the chains that matched a cluster
    makes every system look fully resolved, and a receptor whose
    chains are partly unmapped is declared novel. That is the failure
    the asymmetry was written to prevent, reappearing inside the
    verifier meant to detect it.

    Rows with a null cluster_id are the evidence that unmatched chains
    are present. Their total absence is reported, because a table
    legitimately containing no unmatched chain and a table that
    dropped them look identical.
    """
    out = {"n_rows": int(len(chains)),
           "n_systems": int(chains.system_id.nunique()),
           "n_null_cluster": int(chains.cluster_id.isna().sum())}
    dup = chains.duplicated(["system_id", "chain_id"]).sum()
    out["n_duplicate_chain_rows"] = int(dup)
    if "n_receptor_chains" in allsys.columns:
        got = chains.groupby("system_id").size().rename("got")
        exp = allsys.set_index("system_id")["n_receptor_chains"]
        j = pd.concat([got, exp], axis=1, join="inner")
        short = j[j.got < j.n_receptor_chains]
        out["counts_checked"] = True
        out["n_systems_missing_chains"] = int(len(short))
        out["examples"] = short.head(5).reset_index().to_dict("records")
    else:
        out["counts_checked"] = False
    return out


def reconstruct_fam(chains: pd.DataFrame, allsys: pd.DataFrame,
                    cutoff: str) -> pd.DataFrame:
    """Rebuild the family axis from chain clusters and release dates.

    Familiar if ANY receptor chain sits in a cluster holding a
    pre-cutoff system; novel only if EVERY chain is resolved and none
    is familiar; unresolved otherwise. The asymmetry is deliberate:
    treating a partly unresolved system as novel would manufacture
    novel-family targets out of gaps in the cluster index.
    """
    cut = pd.Timestamp(cutoff)
    dates = pd.to_datetime(allsys.release_date, errors="coerce")
    pre = set(allsys.loc[dates <= cut, "system_id"])

    pre_chains = chains[chains.system_id.isin(pre)]
    familiar_clusters = set(pre_chains.cluster_id.dropna())

    rows = []
    for sid, g in chains.groupby("system_id"):
        assigned = g.cluster_id.notna()
        if (g.loc[assigned, "cluster_id"].isin(familiar_clusters)).any():
            rows.append((sid, True, "familiar"))
        elif assigned.all():
            rows.append((sid, False, "novel"))
        else:
            rows.append((sid, None, "unresolved"))
    return pd.DataFrame(rows,
                        columns=["system_id", "fam_re", "fam_status"])


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--per-ligand", required=True)
    ap.add_argument("--chain-clusters",
                    help="ONE ROW PER RECEPTOR CHAIN: columns "
                         "system_id, chain_id, cluster_id. A chain "
                         "with no cluster must be present with a null "
                         "cluster_id. A file holding only the chains "
                         "that matched would make every system look "
                         "fully resolved, and a partly unmapped "
                         "receptor would be declared novel -- the "
                         "exact failure the asymmetric rule exists to "
                         "prevent. Without this the family axis is "
                         "NOT reconstructed and no verdict is issued.")
    ap.add_argument("--all-systems",
                    help="every system with release_date, not only the "
                         "targets: needed to decide which clusters "
                         "hold a pre-cutoff member. If it also carries "
                         "n_receptor_chains, the chain table is "
                         "checked against it.")
    ap.add_argument("--cutoff", required=True,
                    help="training cutoff, YYYY-MM-DD. No default: a "
                         "default would be inherited silently, and an "
                         "assignment built against the wrong cutoff is "
                         "internally consistent and still wrong.")
    ap.add_argument("--cutoff-source", required=True,
                    help="where that date comes from; recorded as an "
                         "assertion, not a verified fact")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--metric", default="sp_qcov")
    ap.add_argument("--agg", default="min", choices=["min", "max"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")
    if bool(args.chain_clusters) != bool(args.all_systems):
        sys.exit("--chain-clusters and --all-systems go together: "
                 "reconstructing the family axis needs both.")

    tg = pd.read_parquet(args.targets)
    lig = pd.read_parquet(args.per_ligand)
    for c in ("system_id", "quadrant", "S_p", "fam_f", "release_date"):
        if c not in tg.columns:
            sys.exit(f"--targets has no '{c}'")
    if args.metric not in lig.columns:
        sys.exit(f"--per-ligand has no '{args.metric}'")

    report = {"cutoff": args.cutoff,
              "cutoff_source": args.cutoff_source,
              "cutoff_is_asserted_not_verified": True,
              "threshold": args.threshold, "metric": args.metric,
              "agg": args.agg, "n_targets": int(len(tg))}
    print(f"targets {len(tg):,}   per-ligand rows {len(lig):,}")
    print(f"cutoff  {args.cutoff}  (asserted: {args.cutoff_source})")

    # --- the family axis ----------------------------------------------
    fam_reconstructed = False
    if args.chain_clusters:
        ch = pd.read_parquet(args.chain_clusters)
        al = pd.read_parquet(args.all_systems)
        for c in ("system_id", "chain_id", "cluster_id"):
            if c not in ch.columns:
                sys.exit(f"--chain-clusters has no '{c}'. One row per "
                         f"receptor chain is required, including "
                         f"chains with a null cluster_id.")
        for c in ("system_id", "release_date"):
            if c not in al.columns:
                sys.exit(f"--all-systems has no '{c}'")
        chk = check_chain_table(ch, al)
        report["chain_table"] = chk
        print(f"\nCHAIN TABLE")
        print(f"   {chk['n_rows']:,} rows over {chk['n_systems']:,} "
              f"systems, {chk['n_null_cluster']:,} with no cluster")
        if chk["n_duplicate_chain_rows"]:
            print(f"   {chk['n_duplicate_chain_rows']:,} duplicate "
                  f"(system, chain) rows")
        if chk["counts_checked"]:
            print(f"   chain counts checked against --all-systems: "
                  f"{chk['n_systems_missing_chains']:,} systems short")
        else:
            print(f"   --all-systems carries no n_receptor_chains, so "
                  f"completeness is\n   NOT established. If unmatched "
                  f"chains were dropped from this table,\n   a partly "
                  f"unmapped receptor reads as fully resolved.")
        if chk["n_null_cluster"] == 0:
            print(f"   no row has a null cluster_id. Either every "
                  f"chain matched, or the\n   unmatched ones are "
                  f"absent; these are indistinguishable here.")
        fam = reconstruct_fam(ch, al, args.cutoff)
        tg = tg.merge(fam, on="system_id", how="left")
        fam_reconstructed = True
        n_unres = int(tg.fam_re.isna().sum())
        bad_fam = tg[tg.fam_re.notna() &
                     (tg.fam_re.astype(bool) != tg.fam_f.astype(bool))]
        print(f"\nFAMILY AXIS, reconstructed")
        print(f"   disagree with stored fam_f : {len(bad_fam):,}")
        print(f"   unresolved or unmatched    : {n_unres:,}")
        report["fam_rederivation"] = {
            "reconstructed": True, "n_disagree": int(len(bad_fam)),
            "n_unresolved": n_unres,
            "examples": bad_fam.head(5)[
                ["system_id", "fam_f", "fam_status"]].to_dict("records")}
    else:
        print(f"\nFAMILY AXIS, NOT reconstructed")
        print(f"   --chain-clusters was not supplied, so the family "
              f"axis is taken from\n   the table being audited. A "
              f"wrong family assignment would pass unseen.")
        report["fam_rederivation"] = {"reconstructed": False,
                                      "n_disagree": None}

    # --- 1. who cannot be checked at all -------------------------------
    scored = set(lig.system_id)
    absent = tg[~tg.system_id.isin(scored)]
    print(f"\n1. COVERAGE")
    print(f"   {len(absent):,} targets have no row in the similarity "
          f"file")
    if len(absent):
        print(f"   their stored S_p: "
              f"{sorted(absent.S_p.unique())[:8]}")
        print(f"   their stored cells: "
              f"{ {q: int((absent.quadrant == q).sum()) for q in QUADRANTS} }")
    report["absent_from_export"] = {
        "n": int(len(absent)),
        "by_quadrant": {q: int((absent.quadrant == q).sum())
                        for q in QUADRANTS}}

    # --- 2. reconstruct S_p --------------------------------------------
    g = lig.groupby("system_id")[args.metric]
    sp_re = (g.min() if args.agg == "min" else g.max()).rename("sp_re")
    cmp_ = tg[["system_id", "S_p", "quadrant", "fam_f"] +
              (["fam_re"] if fam_reconstructed else [])].merge(
        sp_re, on="system_id", how="left")
    verifiable = cmp_.sp_re.notna()
    sp_bad = cmp_[verifiable &
                  ~np.isclose(cmp_.S_p, cmp_.sp_re.fillna(-1),
                              atol=1e-6)]
    print(f"\n2. RECONSTRUCTED S_p")
    print(f"   checkable {int(verifiable.sum()):,}, disagree with "
          f"stored S_p {len(sp_bad):,}")
    if len(sp_bad):
        print(f"   e.g. "
              f"{sp_bad.head(3)[['system_id','S_p','sp_re']].to_dict('records')}")
    report["sp_rederivation"] = {
        "n_checkable": int(verifiable.sum()),
        "n_disagree": int(len(sp_bad)),
        "examples": sp_bad.head(5)[
            ["system_id", "S_p", "sp_re"]].to_dict("records")}

    # --- 3. the cell, from the RECONSTRUCTED axes only -----------------
    # Deriving the cell from the stored S_p would let a reconstruction
    # that disagrees still produce a matching cell, and the verifier
    # would certify the assignment in exactly the case it exists to
    # catch. Both inputs here are reconstructions.
    fam_src = (cmp_.fam_re if fam_reconstructed else cmp_.fam_f)
    fam_known = fam_src.notna()
    pocket_fam_re = cmp_.sp_re >= args.threshold
    derived = np.where(fam_src.fillna(False).astype(bool),
                       np.where(pocket_fam_re, "FF", "FN"),
                       np.where(pocket_fam_re, "NF", "NN"))
    checkable = verifiable & fam_known
    mism = (derived != cmp_.quadrant) & checkable
    mismatch = cmp_[mism]
    n_unver = int((~checkable).sum())
    print(f"\n3. CELL, derived from the reconstructed axes")
    print(f"   checkable    {int(checkable.sum()):,}")
    print(f"   disagree     {len(mismatch):,}")
    print(f"   UNVERIFIABLE {n_unver:,}  (neither agreeing nor not)")
    if len(mismatch):
        pairs = (mismatch.assign(derived=derived[mism])
                 .groupby(["quadrant", "derived"]).size())
        for (a, b), n in pairs.items():
            print(f"     stored {a} -> derived {b}: {n}")
    report["cell_rederivation"] = {
        "n_checkable": int(checkable.sum()),
        "n_disagree": int(len(mismatch)),
        "n_unverifiable": n_unver,
        "examples": mismatch.head(5)[
            ["system_id", "quadrant", "S_p"]].to_dict("records")}

    # --- 4. the cutoff boundary ----------------------------------------
    rd = pd.to_datetime(tg.release_date, errors="coerce")
    cut = pd.Timestamp(args.cutoff)
    at_or_before, undated = int((rd <= cut).sum()), int(rd.isna().sum())
    print(f"\n4. CUTOFF BOUNDARY")
    print(f"   on or before the cutoff {at_or_before:,}, undated "
          f"{undated:,}")
    if at_or_before:
        print(f"   -> these sit inside their own reference pool.")
    report["boundary"] = {"at_or_before_cutoff": at_or_before,
                          "undated": undated}

    # --- 5. axis association -------------------------------------------
    tab = pd.crosstab(tg.fam_f.astype(bool), tg.S_p >= args.threshold)
    print(f"\n5. AXIS ASSOCIATION")
    if tab.shape == (2, 2):
        a, b = int(tab.loc[True, False]), int(tab.loc[True, True])
        c, d = int(tab.loc[False, False]), int(tab.loc[False, True])
        orat = ((b * c) / (a * d)) if a and d else float("nan")
        phi = ((b * c - a * d) /
               np.sqrt((a + b) * (c + d) * (a + c) * (b + d)))
        print(f"   odds ratio {orat:.2f}, phi {phi:+.3f}")
        print(f"   1.0 and 0.0 would mean independent. A strong "
              f"association would mean\n   the four cells are close "
              f"to two.")
        report["axis_association"] = {"odds_ratio": round(orat, 4),
                                      "phi": round(float(phi), 4)}

    # --- verdict --------------------------------------------------------
    n_cell = report["cell_rederivation"]["n_disagree"]
    n_sp = report["sp_rederivation"]["n_disagree"]
    n_fam = report["fam_rederivation"]["n_disagree"]
    # A target inside its own reference pool is not a near-miss to be
    # noted and passed over: it was scored against a corpus containing
    # itself, so it blocks the freeze outright.
    boundary_clean = (at_or_before == 0 and undated == 0)
    chain_ok = (report.get("chain_table", {}).get("counts_checked")
                and report["chain_table"]
                .get("n_systems_missing_chains", 1) == 0
                and report["chain_table"]
                .get("n_duplicate_chain_rows", 1) == 0)
    frozen = (fam_reconstructed and chain_ok and boundary_clean
              and n_cell == 0 and n_sp == 0 and n_fam == 0
              and n_unver == 0)
    report["verdict"] = {
        "family_axis_reconstructed": fam_reconstructed,
        "chain_table_complete": bool(chain_ok),
        "boundary_clean": boundary_clean,
        "at_or_before_cutoff": at_or_before, "undated": undated,
        "sp_disagreements": n_sp, "cell_disagreements": n_cell,
        "family_disagreements": n_fam, "unverifiable": n_unver,
        "frozen": frozen}
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 64}")
    if frozen:
        print("Both axes reconstruct to the stored values and every "
              "target was\ncheckable. The assignment is reproducible "
              "from source.")
    else:
        why = []
        if not fam_reconstructed:
            why.append("the family axis was not reconstructed")
        elif not chain_ok:
            why.append("the chain table is not shown to be complete")
        if at_or_before:
            why.append(f"{at_or_before:,} targets are at or before the "
                       f"cutoff")
        if undated:
            why.append(f"{undated:,} targets are undated")
        if n_sp:
            why.append(f"{n_sp:,} S_p values disagree")
        if n_fam:
            why.append(f"{n_fam:,} family labels disagree")
        if n_cell:
            why.append(f"{n_cell:,} cells disagree")
        if n_unver:
            why.append(f"{n_unver:,} targets were not checkable")
        print("NOT verified: " + "; ".join(why) + ".")
    print(f"{'=' * 64}")
    print(f"\nWrote {out}")
    print("\n  Direction, upstream coverage, self-comparison and the "
          "cutoff actually\n  used are outside this script. A clean "
          "run does not establish them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
