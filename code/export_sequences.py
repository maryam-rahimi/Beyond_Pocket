#!/usr/bin/env python3
"""
Export the receptor sequences for every manifest system to a plain file.

Run with the PLINDER-capable interpreter; the extractor then reads this
file and needs no PLINDER at all.

WHY SPLIT THIS OUT
------------------
PLINDER and Boltz pin incompatible dependency trees. Installing PLINDER
into the Boltz environment pulls a pyarrow that requires NumPy >= 2.0,
while Boltz pins NumPy < 2.0; and the PLINDER version that resolves in
that environment exposes a different table API from the one on the
cluster's base interpreter. Chasing those differences inside the GPU
environment is a recurring cost for no benefit: the extractor needs
exactly two things from PLINDER, a chain list and a sequence per chain,
both of which are static once the manifest is fixed.

Exporting them once also makes the extraction reproducible without
PLINDER present at all, and puts the sequences actually used under
version control alongside the representations.

THE LENGTH CONTRACT
-------------------
The manifest records each system's expected residue count, and the probes
downstream are per-residue, so the exported sequences must sum to exactly
that. A mismatch is reported per system and the system is marked
unusable rather than exported, because a quietly truncated or extended
sequence would produce a representation that cannot be aligned to its
labels while looking entirely well-formed.

USAGE
-----
    /apps/anaconda/anaconda-2024.10/bin/python export_sequences.py \\
        --manifest extraction_manifest.parquet \\
        --out sequences.json
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
    sys.exit("plinder required; use the PLINDER-capable interpreter, "
             "e.g.\n/apps/anaconda/anaconda-2024.10/bin/python")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite. Pass another --out.")

    man = pd.read_parquet(args.manifest)
    # Cheap assertions on what becomes a frozen scientific input. Neither
    # is expected to fire; both would be invisible downstream if they did.
    if man.system_id.duplicated().any():
        sys.exit("manifest contains duplicate system_id values")
    if man.length.isna().any():
        sys.exit("manifest contains missing lengths")
    want = set(man.system_id)
    expected_len = dict(zip(man.system_id, man.length))
    print(f"manifest: {len(man):,} systems")

    print("loading chain membership ...")
    ann = pc.query_table(
        "annotation",
        columns=["system_id", "entry_pdb_id",
                 "system_protein_chains_asym_id"]).drop_duplicates("system_id")
    ann = ann[ann.system_id.isin(want)]
    print(f"  {len(ann):,} systems found")

    print("loading sequences ...")
    ec = pc.query_table("entry_chains",
                        columns=["entry_pdb_id", "chain_asym_id",
                                 "chain_sequence"])
    seqs = {(e, c): s for e, c, s in
            zip(ec.entry_pdb_id, ec.chain_asym_id, ec.chain_sequence)
            if isinstance(s, str)}
    print(f"  {len(seqs):,} chains")

    entry_of = dict(zip(ann.system_id, ann.entry_pdb_id))
    chains_of = dict(zip(ann.system_id, ann.system_protein_chains_asym_id))

    records, problems = {}, []
    for sysid in man.system_id:
        entry = entry_of.get(sysid)
        chains = chains_of.get(sysid)
        if entry is None or chains is None or len(chains) == 0:
            problems.append({"system_id": sysid, "reason": "no annotation"})
            continue

        chain_list, missing = [], False
        for inst in chains:
            inst = str(inst)
            asym = inst.split(".", 1)[-1]
            seq = seqs.get((entry, asym))
            if seq is None:
                problems.append({"system_id": sysid,
                                 "reason": f"no sequence for {entry} {asym}"})
                missing = True
                break
            chain_list.append({"instance": inst, "asym": asym,
                               "sequence": seq, "length": len(seq)})
        if missing:
            continue

        total = sum(c["length"] for c in chain_list)
        want_len = int(expected_len[sysid])
        if total != want_len:
            # Fail this system rather than export it. A sequence that does
            # not match the label length produces a representation that
            # cannot be aligned to its labels, and nothing downstream
            # would notice.
            problems.append({"system_id": sysid,
                             "reason": f"length {total} != manifest "
                                       f"{want_len}"})
            continue

        records[sysid] = {"entry_pdb_id": entry, "chains": chain_list,
                          "length": total}

    out.write_text(json.dumps(records, indent=1))
    print(f"\nexported {len(records):,} of {len(man):,} systems "
          f"({100*len(records)/len(man):.1f}%)")
    if problems:
        pp = Path(str(out) + ".problems.json")
        pp.write_text(json.dumps(problems, indent=1))
        reasons = {}
        for p in problems:
            key = p["reason"].split(" for ")[0].split(" !=")[0]
            reasons[key] = reasons.get(key, 0) + 1
        print(f"{len(problems):,} systems not exported:")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v:,}")
        print(f"Wrote {pp}")
        print("\n  These are excluded rather than exported with a mismatched "
              "sequence: a\n  representation that cannot be aligned to its "
              "labels is worse than a\n  missing one, because nothing "
              "downstream would detect it.")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
