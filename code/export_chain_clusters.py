#!/usr/bin/env python3
"""
Export the receptor-chain cluster table and the system date table that
the continuous family axis needs.

READ FROM DISK, NOT THROUGH THE PACKAGE
---------------------------------------
The tables are read as parquet files from a release directory given
on the command line, rather than fetched through plinder.core.

This is not a convenience. The installed package refuses any release
but 2024-06 -- "R2 downloads support only plinder/2024-06/v2" -- and
its default silently points there, so a script that calls the API
gets a different snapshot from the one this study was built on,
without saying so. That is how a composition analysis came to run
against a release missing 57% of the test systems: it reported 43%
coverage and looked like a property of the data.

Reading named files from a named directory makes the release an
argument rather than an ambient setting, and a wrong one fails rather
than substituting.

WHY THIS AND NOT A HAND-ROLLED QUERY
------------------------------------
build_quadrants.py already derives the binary family axis from
PLINDER, and the continuous axis must agree with it on the one thing
they share: which chains belong to which family. Rebuilding that join
from a different reading of the schema would produce a continuous
covariate that disagrees with the binary one for reasons nobody could
later reconstruct.

So the loaders and the join below are taken from that script
unchanged: the same `annotation` columns, the same `entry_metadata`
merge, the same `protein_sequence_clusters` table, the same
representative key built from representative_entry_pdb_id and
representative_chain_asym_id, and the same instance-prefix rule.

  system_protein_chains_asym_id holds chain INSTANCES, "1.A". The
  cluster tables hold asym ids, "A". Dropping the prefix is what makes
  the join work, and getting it wrong yields a table of nulls rather
  than an error.

WHAT IS EXPORTED
----------------
  receptor_chain_clusters.parquet   one row per receptor chain, with
                                    unmatched chains RETAINED and
                                    cluster_id null
  all_systems_with_dates.parquet    one row per system over the whole
                                    corpus, not only the targets

Both are needed whole. The continuous count asks how many pre-cutoff
systems share a target's cluster, so the reference side must cover
every pre-cutoff system; filtering either table to the evaluation
targets would silently undercount support.

Unmatched chains are retained for the same reason the binary axis is
asymmetric: a chain with no cluster assignment is unresolved, not
unfamiliar, and dropping its row would make a partly resolved receptor
look complete.

FAIL-CLOSED CHECKS
------------------
Four, each of which would otherwise produce a well-formed table that
is wrong:

  No chain matches a cluster. Almost certainly the instance-prefix
  convention has changed, and every count downstream would be zero.

  Duplicate (system, chain instance) rows that are not identical. The
  explode should give one row per chain.

  One (entry, asym id) carrying several representatives. The join is a
  lookup and a conflict makes it arbitrary.

  Release dates that will not parse. The pre-cutoff pool is defined by
  date, so an unparseable one is a system of unknown vintage rather
  than a late one.

USAGE
-----
    python3 export_chain_clusters.py \\
        --plinder-dir ~/.local/share/plinder/2026-09 \\
        --outdir .
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Exactly as in build_quadrants.py.
_INSTANCE = re.compile(r"^\d+\.")


def strip_instance(chain):
    """'1.A' -> 'A'. Chain instances carry a biounit prefix the
    cluster tables do not use."""
    return _INSTANCE.sub("", str(chain))


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plinder-dir", required=True,
                    help="the release directory, e.g. "
                         "~/.local/share/plinder/2026-09. Required "
                         "and not inferred: the package's default "
                         "points at another release and would "
                         "substitute it silently.")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chains_out = outdir / "receptor_chain_clusters.parquet"
    systems_out = outdir / "all_systems_with_dates.parquet"
    for p in (chains_out, systems_out):
        if p.exists():
            sys.exit(f"{p} exists; refusing to overwrite.")

    root = Path(args.plinder_dir).expanduser()
    ann_p = root / "index" / "annotation_table.parquet"
    meta_p = root / "index" / "entry_metadata.parquet"
    seq_p = root / "protein_clusters" / "sequence.parquet"
    for p in (ann_p, meta_p, seq_p):
        if not p.exists():
            sys.exit(f"{p} does not exist. Check --plinder-dir names "
                     f"a release directory\nwith index/ and "
                     f"protein_clusters/ beneath it.")
    print(f"release directory {root}")

    # --- systems, as load_systems() builds them ------------------------
    cols = ["system_id", "entry_pdb_id",
            "system_protein_chains_asym_id",
            "system_pass_validation_criteria",
            "system_num_protein_chains"]
    print("reading annotation_table.parquet ...")
    ann = pd.read_parquet(ann_p, columns=cols).drop_duplicates(
        "system_id")
    print("reading entry_metadata.parquet ...")
    meta = pd.read_parquet(meta_p, columns=["entry_pdb_id",
                                            "entry_release_date"])
    systems = ann.merge(meta, on="entry_pdb_id", how="left")
    systems["release_date"] = pd.to_datetime(
        systems["entry_release_date"], errors="coerce")
    print(f"  {len(systems):,} systems, "
          f"{systems.entry_pdb_id.nunique():,} entries")

    bad_dates = int(systems.release_date.isna().sum())
    if bad_dates:
        ex = systems.loc[systems.release_date.isna(),
                         "system_id"].head(5).tolist()
        sys.exit(f"{bad_dates:,} systems have no parseable release "
                 f"date, e.g. {ex}.\nThe pre-cutoff pool is defined "
                 f"by date; a system of unknown vintage cannot be "
                 f"placed\ninside or outside it, and treating it as "
                 f"post-cutoff would understate support.")

    # --- the cluster table, as _seq_table() builds it ------------------
    print("reading protein_clusters/sequence.parquet ...")
    seq = pd.read_parquet(seq_p)
    seq["_rep"] = (seq["representative_entry_pdb_id"].astype(str) + "|"
                   + seq["representative_chain_asym_id"].astype(str))
    print(f"  {len(seq):,} chain rows, "
          f"{seq._rep.nunique():,} representatives")

    dup = seq.duplicated(["entry_pdb_id", "chain_asym_id"], keep=False)
    if dup.any():
        conflict = (seq[dup].groupby(["entry_pdb_id", "chain_asym_id"])
                    ._rep.nunique())
        conflict = conflict[conflict > 1]
        if len(conflict):
            sys.exit(f"{len(conflict):,} (entry, chain) pairs carry "
                     f"more than one representative, e.g.\n"
                     f"{conflict.head(3).to_dict()}.\nThe join is a "
                     f"lookup; a conflict makes which representative "
                     f"wins arbitrary.")
    chain_rep = dict(zip(zip(seq["entry_pdb_id"], seq["chain_asym_id"]),
                         seq["_rep"]))

    # --- one row per receptor chain ------------------------------------
    pairs = systems[["system_id", "entry_pdb_id",
                     "system_protein_chains_asym_id"]].explode(
        "system_protein_chains_asym_id")
    pairs = pairs.rename(
        columns={"system_protein_chains_asym_id": "chain_instance"})
    pairs = pairs[pairs.chain_instance.notna()].copy()
    pairs["chain_id"] = pairs.chain_instance.map(strip_instance)
    pairs["cluster_id"] = [chain_rep.get((e, c)) for e, c in
                           zip(pairs.entry_pdb_id, pairs.chain_id)]

    d = pairs.duplicated(["system_id", "chain_instance"], keep=False)
    if d.any():
        same = (pairs[d].groupby(["system_id", "chain_instance"])
                .cluster_id.nunique(dropna=False))
        same = same[same > 1]
        if len(same):
            sys.exit(f"{len(same):,} (system, chain instance) pairs "
                     f"appear more than once with different "
                     f"clusters.\nThe explode should give one row per "
                     f"chain.")
        pairs = pairs.drop_duplicates(["system_id", "chain_instance"])
        print(f"  dropped identical duplicate chain rows")

    matched = pairs.cluster_id.notna()
    if matched.sum() == 0:
        sys.exit("No receptor chain matched a sequence cluster. The "
                 "chain-id convention may\nhave changed; inspect "
                 "system_protein_chains_asym_id against\n"
                 "protein_sequence_clusters.chain_asym_id before "
                 "trusting any counts.")

    # --- provenance, comparable with the binary axis's own numbers -----
    by_sys = pairs.groupby("system_id").cluster_id
    all_res = by_sys.apply(lambda s: s.notna().all())
    print(f"\nreceptor chains        {len(pairs):,}")
    print(f"  matched              {int(matched.sum()):,} "
          f"({matched.mean():.4f})")
    print(f"  unmatched, retained  {int((~matched).sum()):,}")
    print(f"systems                {pairs.system_id.nunique():,}")
    print(f"  all chains resolved  {int(all_res.sum()):,} "
          f"({all_res.mean():.4f})")
    print(f"  partially resolved   {int((~all_res).sum()):,}")
    print(f"release dates parsed   {len(systems):,} of {len(systems):,}")
    print(f"\n  compare the two rates against the binary axis: it "
          f"reported 0.9817 and\n  0.9805 for chain coverage and "
          f"complete systems. A large divergence means\n  this join "
          f"is not the one that built S_f.")

    pairs[["system_id", "entry_pdb_id", "chain_instance", "chain_id",
           "cluster_id"]].to_parquet(chains_out)
    systems[["system_id", "entry_pdb_id",
             "release_date"]].to_parquet(systems_out)
    meta_json = {
        "source": str(root),
        "read_from": "parquet files on disk, not the package API",
        "tables": ["annotation", "entry_metadata",
                   "protein_sequence_clusters"],
        "cluster_id": "representative_entry_pdb_id | "
                      "representative_chain_asym_id, as in "
                      "build_quadrants.py",
        "chain_rule": "system_protein_chains_asym_id instances have "
                      "the biounit prefix stripped: 1.A -> A",
        "unmatched_chains": "retained with a null cluster_id",
        "n_systems": int(len(systems)),
        "n_receptor_chains": int(len(pairs)),
        "chain_match_rate": round(float(matched.mean()), 6),
        "complete_system_rate": round(float(all_res.mean()), 6),
    }
    Path(str(chains_out) + ".meta.json").write_text(
        json.dumps(meta_json, indent=2))
    print(f"\nWrote {chains_out}")
    print(f"Wrote {systems_out}")
    print("\n  Neither table is filtered to the evaluation targets. "
          "The continuous count\n  asks how many PRE-CUTOFF systems "
          "share a target's cluster, so the reference\n  side must "
          "cover the whole corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
