#!/usr/bin/env python3
"""
Assign PLINDER systems to the four familiarity quadrants.

WHAT THIS DOES
--------------
Every evaluation target needs two familiarity labels:

    S_f  protein-family familiarity
    S_p  pocket familiarity

and their 2x2 gives FF / FN / NF / NN. This script produces a BINARY version
of both from PLINDER's cluster-membership tables:

    a target is FAMILIAR on an axis when at least one system released on or
    before the trunk's training cutoff shares its cluster on that axis

which answers "had the model seen anything like this before training?", at
whatever threshold PLINDER's clustering used.

THE TWO AXES, AND WHY THEY JOIN DIFFERENTLY
-------------------------------------------
S_f is chain-level. `protein_sequence_clusters` is keyed on
(entry_pdb_id, chain_asym_id), so the join must run through the system's OWN
receptor chains, listed in `system_protein_chains_asym_id`. Collapsing to
the PDB entry instead would be wrong: one entry can hold a familiar chain A
and a novel chain B, and a system whose receptor is only B would inherit A's
familiarity and be miscounted as FF. That inflates the familiar cells and
starves NN -- the regime the study exists to measure -- so the chain-level
join is not a refinement, it is the difference between a correct and a
misleading answer.

Note that `system_protein_chains_asym_id` holds chain INSTANCES ("1.A"),
while the cluster tables hold asym ids ("A"). The instance prefix is
stripped before joining; without that the join silently matches nothing.

S_p is already system-level. `ligand_pocket_membership` maps system_id
directly to representative_system_id, so no chain join is involved.

Within an axis, a system counts as familiar if ANY of its own chains (or its
pocket) is familiar. `any` rather than `all` is the conservative direction
for a novelty claim: one familiar chain means the model has seen related
structure, so calling such a system novel would overstate novelty.

THE REFERENCE POOL
------------------
"Familiar" is judged against ALL pre-cutoff systems, not only validated
ones. The question is what the model could have been exposed to, and the
trunk trained on the PDB rather than on a quality-filtered subset. Quality
filters apply to the TARGETS -- where ground-truth pocket labels must be
trustworthy -- not to the reference pool.

WHAT THIS IS NOT
----------------
Not the continuous S_f / S_p the protocol specifies. Cluster membership
carries no similarity value, so there is no threshold to sweep and no
continuous regression: the threshold belongs to PLINDER's clustering
procedure and is not ours to tune. The manuscript should say so.

USAGE
-----
    python3 build_quadrants.py --cutoff 2023-06-01 \
        --out quadrants.json --write-table targets.parquet

    python3 build_quadrants.py --no-validation-filter   # larger, noisier
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import plinder.core as pc
except ImportError:
    sys.exit("plinder is required. On Newton use "
             "/apps/anaconda/anaconda-2024.10/bin/python")

QUADRANTS = ["FF", "FN", "NF", "NN"]
_INSTANCE = re.compile(r"^\d+\.")


def strip_instance(chain):
    """'1.A' -> 'A'. Chain instances carry a biounit prefix the cluster
    tables do not use."""
    return _INSTANCE.sub("", str(chain))


def family_signature(entry_id, chains, chain_rep):
    """A deterministic family label built from ALL of a system's receptor
    chains, as the sorted set of their sequence-cluster representatives.

    Using only the first chain would reduce a heteromeric receptor with
    chains in clusters C_A and C_B to C_A alone, which does not change the
    FF/FN/NF/NN label but distorts the distinct-cluster counts those labels
    are judged by -- and those counts are what decide whether a cell carries
    enough statistical weight.

    Sorting and de-duplicating makes the label order-independent, so a
    homomer whose chains share one cluster collapses to that single id while
    a heteromer becomes "clusterA;clusterB".

    The signature is all-or-nothing: if ANY receptor chain lacks a cluster
    assignment the whole signature is None, because a partial composition
    would be reported as though the receptor were fully characterised.
    Note this is a different question from the S_f label, and the two can
    diverge: a heteromer with one familiar chain and one unmapped chain is
    legitimately FAMILIAR (one witness proves it) yet has no complete
    signature. Such a system keeps its quadrant and contributes no cluster
    to that cell's count. That undercounts clusters rather than inventing
    them, which is the safe direction, and the count of affected targets is
    reported so the shortfall is visible.

    This counts receptor COMPOSITIONS, not disjoint family groups. Two
    systems with receptors {A} and {A,B} get different signatures and count
    as two, although they share family A and are not fully independent. It
    is strictly better than a first-chain label, but it is an upper bound on
    independence rather than an exact measure, and the manuscript should say
    so where these counts are used.
    """
    if chains is None or len(chains) == 0:
        return None
    reps = []
    for ch in chains:
        rep = chain_rep.get((entry_id, strip_instance(ch)))
        if rep is None:
            return None
        reps.append(rep)
    return ";".join(sorted(set(reps)))


def load_systems():
    """One row per system: entry, release date, receptor chains, quality."""
    cols = ["system_id", "entry_pdb_id", "system_protein_chains_asym_id",
            "system_pass_validation_criteria", "system_num_protein_chains"]
    ann = pc.query_table("annotation", columns=cols).drop_duplicates(
        "system_id")
    meta = pc.query_table("entry_metadata",
                          columns=["entry_pdb_id", "entry_release_date"])
    df = ann.merge(meta, on="entry_pdb_id", how="left")
    df["release_date"] = pd.to_datetime(df["entry_release_date"],
                                        errors="coerce")
    return df


def _seq_table():
    seq = pc.query_table("protein_sequence_clusters")
    seq["_rep"] = (seq["representative_entry_pdb_id"].astype(str) + "|"
                   + seq["representative_chain_asym_id"].astype(str))
    return seq


def sequence_familiarity(systems, cutoff, seq):
    """Per system: does a pre-cutoff system share any of ITS chains' clusters?"""
    dates = systems[["entry_pdb_id", "release_date"]].drop_duplicates(
        "entry_pdb_id")
    seq_dated = seq.merge(dates, on="entry_pdb_id", how="left")
    pre_reps = set(seq_dated.loc[
        seq_dated["release_date"].notna()
        & (seq_dated["release_date"] <= cutoff), "_rep"].unique())

    chain_rep = dict(zip(
        zip(seq["entry_pdb_id"], seq["chain_asym_id"]), seq["_rep"]))

    pairs = systems[["system_id", "entry_pdb_id",
                     "system_protein_chains_asym_id"]].explode(
        "system_protein_chains_asym_id")
    pairs = pairs.rename(columns={"system_protein_chains_asym_id": "chain"})
    pairs = pairs[pairs["chain"].notna()].copy()
    pairs["chain_asym_id"] = pairs["chain"].map(strip_instance)
    pairs["_rep"] = [chain_rep.get((e, c))
                     for e, c in zip(pairs["entry_pdb_id"],
                                     pairs["chain_asym_id"])]

    matched = pairs["_rep"].notna()
    if matched.sum() == 0:
        sys.exit("No receptor chain matched a sequence cluster. The chain-id "
                 "convention may\nhave changed; inspect "
                 "system_protein_chains_asym_id against\n"
                 "protein_sequence_clusters.chain_asym_id before trusting "
                 "any counts.")

    pairs["_familiar"] = pairs["_rep"].isin(pre_reps) & matched

    # Familiarity and novelty need different evidence, so the aggregation is
    # deliberately asymmetric:
    #
    #   F           at least one receptor chain is KNOWN familiar. One
    #               witness suffices -- the model has seen related structure.
    #   N           every receptor chain is mapped and none is familiar.
    #               Novelty is a claim about ALL of them, so it needs
    #               complete coverage.
    #   unresolved  no familiar chain, but at least one chain has no cluster
    #               assignment. That chain might belong to a pre-cutoff
    #               family; we cannot tell.
    #
    # Treating the last case as novel is the specific failure to avoid: it
    # manufactures NF and NN targets out of gaps in the cluster index, in
    # the cells the study is built to measure. An earlier version of this
    # function resolved a system when ANY chain matched, which protected
    # only systems where nothing matched at all.
    groups = pairs.groupby("system_id")
    any_familiar = groups["_familiar"].any()
    all_resolved = groups["_rep"].apply(lambda s: s.notna().all())
    fam = any_familiar.astype("boolean").where(any_familiar | all_resolved)

    complete = float(all_resolved.mean())
    return fam, float(matched.mean()), complete


def pocket_familiarity(systems, cutoff):
    """Per system: does a pre-cutoff system share its pocket representative?"""
    pm = pc.query_table("ligand_pocket_membership",
                        columns=["system_id", "representative_system_id"])
    pm = pm.drop_duplicates(["system_id", "representative_system_id"])

    pm_dated = pm.merge(systems[["system_id", "release_date"]],
                        on="system_id", how="left")
    pre_reps = set(pm_dated.loc[
        pm_dated["release_date"].notna()
        & (pm_dated["release_date"] <= cutoff),
        "representative_system_id"].unique())

    pm["_familiar"] = pm["representative_system_id"].isin(pre_reps)
    return pm.groupby("system_id")["_familiar"].any()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="2023-06-01",
                    help="trunk training cutoff (default 2023-06-01, "
                         "Boltz-2). Targets are released strictly after it; "
                         "the reference pool is everything on or before.")
    ap.add_argument("--no-validation-filter", action="store_true",
                    help="keep targets failing "
                         "system_pass_validation_criteria. Larger pool, "
                         "less trustworthy pocket ground truth.")
    ap.add_argument("--out", help="write counts JSON here")
    ap.add_argument("--write-table",
                    help="write per-system assignments (.parquet or .csv)")
    args = ap.parse_args(argv)

    cutoff = pd.Timestamp(args.cutoff)
    print("=" * 70)
    print("QUADRANT ASSIGNMENT  (binary, from cluster membership)")
    print("=" * 70)
    print(f"cutoff     : {args.cutoff}  (targets released strictly after)")
    print("S_f        : protein_sequence_clusters, via the system's own chains")
    print("S_p        : ligand_pocket_membership (system-level)")
    print("reference  : ALL pre-cutoff systems, validated or not")

    systems = load_systems()
    n_all = len(systems)
    n_unknown = int(systems["release_date"].isna().sum())
    post = systems["release_date"] > cutoff
    print(f"\nsystems    : {n_all:,}")
    print(f"  pre-cutoff   : "
          f"{int((systems['release_date'] <= cutoff).sum()):,}   (reference)")
    print(f"  post-cutoff  : {int(post.sum()):,}   (candidate targets)")
    if n_unknown:
        print(f"  unknown date : {n_unknown:,}   (excluded from both)")

    print("\ncomputing familiarity...")
    seq = _seq_table()
    fam_f, coverage, complete = sequence_familiarity(systems, cutoff, seq)
    fam_p = pocket_familiarity(systems, cutoff)
    print(f"  sequence-cluster coverage: {coverage:.1%} of receptor chains")
    # Chain coverage and system coverage differ, and the gap is entirely
    # multichain receptors: a 99% chain rate still leaves partially
    # unresolved systems, and those cannot be called novel.
    print(f"  complete-system coverage : {complete:.1%} of systems have "
          f"EVERY receptor chain mapped")

    tgt = systems[post].copy()
    if not args.no_validation_filter:
        before = len(tgt)
        tgt = tgt[tgt["system_pass_validation_criteria"].fillna(False)]
        print(f"  validation filter: {before:,} -> {len(tgt):,} targets")

    tgt["fam_f"] = tgt["system_id"].map(fam_f)
    tgt["fam_p"] = tgt["system_id"].map(fam_p)

    # Per-axis, because the two fail for different reasons: S_f from
    # incomplete receptor-chain cluster coverage, S_p from a system absent
    # from the pocket-membership table.
    n_unres_f = int(tgt["fam_f"].isna().sum())
    n_unres_p = int(tgt["fam_p"].isna().sum())
    n_unres = int((tgt["fam_f"].isna() | tgt["fam_p"].isna()).sum())
    tgt = tgt[tgt["fam_f"].notna() & tgt["fam_p"].notna()].copy()
    if n_unres:
        print(f"  unresolved S_f: {n_unres_f:,}")
        print(f"  unresolved S_p: {n_unres_p:,}")
        print(f"  dropped (either): {n_unres:,}, leaving {len(tgt):,}")

    tgt["quadrant"] = (tgt["fam_f"].map({True: "F", False: "N"}).astype(str)
                       + tgt["fam_p"].map({True: "F", False: "N"}).astype(str))

    # Cluster-aware counting. Systems are not independent evaluation units,
    # and the distinct-cluster count is what governs how much weight a cell
    # really carries, so the label must reflect the whole receptor rather
    # than an arbitrary chain of it.
    chain_rep = dict(zip(
        zip(seq["entry_pdb_id"], seq["chain_asym_id"]), seq["_rep"]))
    tgt["family_cluster"] = [
        family_signature(e, ch, chain_rep)
        for e, ch in zip(tgt["entry_pdb_id"],
                         tgt["system_protein_chains_asym_id"])]

    n_sig_missing = int(tgt["family_cluster"].isna().sum())
    if n_sig_missing:
        print(f"  incomplete family signatures: {n_sig_missing:,} assigned "
              f"targets\n    (these keep their quadrant but contribute no "
              f"cluster to its count)")

    cells = {}
    for q in QUADRANTS:
        sub = tgt[tgt["quadrant"] == q]
        cells[q] = {"systems": int(len(sub)),
                    "entries": int(sub["entry_pdb_id"].nunique()),
                    "clusters": int(sub["family_cluster"].nunique())}

    print(f"\nassigned   : {len(tgt):,} targets")
    print(f"\n{'':6}{'systems':>12}{'entries':>10}{'clusters':>10}")
    for q in QUADRANTS:
        c = cells[q]
        flag = "   <-- true-discovery regime" if q == "NN" else ""
        print(f"{q:6}{c['systems']:>12,}{c['entries']:>10,}"
              f"{c['clusters']:>10,}{flag}")

    weakest = min(c["clusters"] for c in cells.values())
    print(f"\nweakest cell: {weakest:,} family clusters")
    print("\n  Counts appear three ways because systems are not independent "
          "units:\n  many share an entry, and many entries share a family "
          "cluster. The\n  cluster column governs a cell's statistical "
          "weight.")
    print("\n  'Familiar' means a pre-cutoff system shares this target's "
          "cluster, at\n  whatever threshold PLINDER's clustering used -- "
          "not a tuned threshold,\n  and not a continuous similarity.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "cutoff": args.cutoff,
            "reference_pool": "all pre-cutoff systems",
            "validation_filter": not args.no_validation_filter,
            "total_systems": n_all,
            "post_cutoff_systems": int(post.sum()),
            "assigned_targets": int(len(tgt)),
            "sequence_cluster_chain_coverage": coverage,
            "complete_system_coverage": complete,
            "unresolved_sf": n_unres_f,
            "unresolved_sp": n_unres_p,
            "incomplete_family_signatures": n_sig_missing,
            "cells": cells,
            "weakest_cell_clusters": int(weakest),
        }, indent=2))
        print(f"\nWrote {args.out}")

    if args.write_table:
        p = Path(args.write_table)
        cols = ["system_id", "entry_pdb_id", "release_date", "fam_f",
                "fam_p", "quadrant", "family_cluster"]
        (tgt[cols].to_parquet(p) if p.suffix == ".parquet"
         else tgt[cols].to_csv(p, index=False))
        print(f"Wrote {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
