#!/usr/bin/env python3
"""
Reorder the extraction manifest so that a partial run covers all four
familiarity regimes.

The sampler writes systems grouped by quadrant, and the extractor
processes the manifest in order. So an interrupted run fills FF first
and the other cells not at all: the 104 systems from the first attempt
are entirely FF, which is why the pilot has one quadrant and no
comparison.

That has now happened twice -- once to a walltime limit, once to the
alignment server. A run is more likely to be cut short than to finish
cleanly, so the order should be chosen for what a partial run leaves
behind.

INTERLEAVED, NOT SHUFFLED
-------------------------
A uniform shuffle would spread the regimes but not evenly: with 500 FF
and 459 NN, a random prefix of 200 would hold roughly 52 FF and 48 NN,
and the ratio would wander. Interleaving round-robin by quadrant makes
every prefix as balanced as the cell sizes permit, so stopping at any
point gives the most usable set available at that point.

Within each quadrant the order is shuffled, with a fixed seed, so a
prefix is not systematically the shortest or the most familiar systems
of its cell.

THE RUN SIGNATURE CHANGES
-------------------------
The extractor hashes the manifest into its run signature, so a reordered
manifest will not resume into an existing output directory. That is the
signature working as intended: representations built from two different
manifests are two collections.

The 104 systems already extracted were produced under the old manifest.
They must therefore be either re-extracted under the new one or kept as
a separate collection; mixing them silently is exactly what the
signature prevents. Re-extracting 104 systems is about 35 minutes, and
one provenance group is worth more than the time saved.

NOTHING ELSE CHANGES
--------------------
Only the row order. The same systems, the same quadrant assignments, the
same clusters, the same lengths. The script asserts this rather than
asserting it in prose.

USAGE
-----
    python3 shuffle_manifest.py \\
        --manifest extraction_manifest.parquet \\
        --out extraction_manifest_interleaved.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUADRANTS = ("FF", "FN", "NF", "NN")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite. Pass another --out.")

    man = pd.read_parquet(args.manifest)
    print(f"manifest: {len(man):,} systems")

    # The permutation check below compares SETS, which erase
    # duplicates, and set_index("system_id").loc[...] on a duplicated
    # index returns more rows than it was asked for. A duplicate here
    # would therefore pass the check and corrupt the comparison.
    if man["system_id"].duplicated().any():
        dup = man.loc[man["system_id"].duplicated(keep=False),
                      "system_id"].tolist()
        sys.exit(f"manifest contains duplicate system_id values, "
                 f"e.g. {dup[:5]}")

    # Rows are distributed by quadrant, so a label outside the expected
    # set would never be placed and would vanish. The row-count check
    # would notice, but far too late to say why.
    if man["quadrant"].isna().any():
        sys.exit(f"{int(man['quadrant'].isna().sum())} systems have no "
                 f"quadrant; every system must belong to a regime")
    bad_q = sorted(set(man["quadrant"]) - set(QUADRANTS))
    if bad_q:
        sys.exit(f"unexpected quadrant values: {bad_q}. Expected only "
                 f"{list(QUADRANTS)}; a stray label would be silently "
                 f"dropped from the reordering.")

    print("\ncurrent order, first 200 rows:")
    head = man.head(200).quadrant.value_counts()
    for q in QUADRANTS:
        print(f"  {q}: {int(head.get(q, 0)):>4}")

    rng = np.random.default_rng(args.seed)
    by_q = {}
    for q in QUADRANTS:
        sub = man[man.quadrant == q]
        if sub.empty:
            continue
        order = rng.permutation(len(sub))
        by_q[q] = sub.iloc[order].reset_index(drop=True)

    # Round robin: take one system from each non-exhausted quadrant in
    # turn, so every prefix is as balanced as the cell sizes allow.
    rows, i = [], 0
    while any(i < len(v) for v in by_q.values()):
        for q in QUADRANTS:
            v = by_q.get(q)
            if v is not None and i < len(v):
                rows.append(v.iloc[i])
        i += 1
    new = pd.DataFrame(rows).reset_index(drop=True)

    # The reorder must be a permutation of the same systems, nothing
    # added, dropped or altered.
    if len(new) != len(man):
        sys.exit(f"row count changed: {len(man)} -> {len(new)}")
    if set(new.system_id) != set(man.system_id):
        sys.exit("the system set changed; this is not a reordering")
    merged = man.set_index("system_id").loc[new.system_id]
    for col in man.columns:
        if col == "system_id":
            continue
        a = merged[col].to_numpy()
        b = new.set_index("system_id").loc[new.system_id, col].to_numpy()
        if not np.array_equal(pd.isna(a), pd.isna(b)) or \
                not (pd.Series(a).fillna("~").to_numpy()
                     == pd.Series(b).fillna("~").to_numpy()).all():
            sys.exit(f"column '{col}' changed during reordering")
    print("\nverified: same systems, same values, order only")

    print("\nnew order, prefix balance:")
    print(f"{'prefix':>8}" + "".join(f"{q:>6}" for q in QUADRANTS))
    for n in (100, 200, 400, 800, len(new)):
        if n > len(new):
            continue
        c = new.head(n).quadrant.value_counts()
        print(f"{n:>8}" + "".join(f"{int(c.get(q, 0)):>6}"
                                  for q in QUADRANTS))

    new.to_parquet(out)
    Path(str(out) + ".meta.json").write_text(json.dumps({
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": hashlib.sha256(
            Path(args.manifest).read_bytes()).hexdigest(),
        "order": "round-robin across quadrants, shuffled within",
        "seed": args.seed,
        "n_systems": int(len(new)),
        "cells": {q: int((new.quadrant == q).sum()) for q in QUADRANTS},
    }, indent=2))
    print(f"\nWrote {out}")
    print("\n  This changes the extractor's run signature, so it will "
          "not resume\n  into the existing output directory. Extract to "
          "a new --outdir, and\n  treat the 104 systems from the "
          "previous manifest as superseded\n  rather than mixing the "
          "two.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
