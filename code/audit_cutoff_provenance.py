#!/usr/bin/env python3
"""
Confirm that every artefact the manuscript reports came from the
2021-09-30 cutoff and not the 2023-06-01 one.

WHY
---
A table generated against the Boltz-2 cutoff reached the manuscript
once. Both pipelines were run and both sets of outputs were kept, so
the question is not whether the mistake was possible but whether it
happened anywhere else.

The cutoff is not a cosmetic parameter. It decides which systems are
evaluation targets at all, and it decides each surviving target's
familiarity against the pre-cutoff reference pool. A system released
between the two dates is a target under one cutoff and part of the
reference pool under the other, so a single wrong file changes both
the population and the labels.

THE DISCRIMINATING TEST
-----------------------
Systems released strictly between 2021-09-30 and 2023-06-01 exist in
the Boltz-1 target pool and cannot exist in the Boltz-2 one. Their
presence identifies the pipeline, and it does so from the data itself
rather than from a filename or a timestamp.

    none present        ->  the Boltz-2 pool, or a pool so small the
                            test is uninformative; either way, stop
    many present        ->  the Boltz-1 pool

The converse also matters: nothing may predate 2021-09-30, since that
would mean the reference pool leaked into the targets.

WHAT IS CHECKED
---------------
  the target table            population and labels
  the probe index             the same assignment, carried forward
  every JSON result           its recorded cutoff, where it records one
  the extraction manifest     the systems actually run through the trunk

Each is checked against the target table rather than against a
declared constant, because a constant in this script could be wrong
in the same way a constant in a generator was.

USAGE
-----
    python3 audit_cutoff_provenance.py \\
        --results-dir ~/Beyond\\ Pocket/03\\ Results \\
        --expected-cutoff 2021-09-30 \\
        --other-cutoff 2023-06-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--expected-cutoff", default="2021-09-30")
    ap.add_argument("--other-cutoff", default="2023-06-01")
    ap.add_argument("--targets", default="targets_with_sp_boltz1.parquet")
    ap.add_argument("--index", default="probe_index.parquet")
    args = ap.parse_args(argv)

    root = Path(args.results_dir).expanduser()
    if not root.is_dir():
        sys.exit(f"{root} is not a directory")
    want = pd.Timestamp(args.expected_cutoff)
    other = pd.Timestamp(args.other_cutoff)
    if not want < other:
        sys.exit("--expected-cutoff must be the earlier of the two")

    problems, notes = [], []

    # ---- the target table decides everything downstream -------------
    tg_path = root / args.targets
    if not tg_path.exists():
        sys.exit(f"{tg_path} does not exist")
    tg = pd.read_parquet(tg_path)
    for c in ("system_id", "release_date"):
        if c not in tg.columns:
            sys.exit(f"{tg_path.name} has no '{c}'")
    # A quadrant column here is optional and its absence is
    # deliberate. An earlier build of this file carried one, left
    # over from a different threshold, which disagreed with the
    # index on about half the systems; the index is the authority
    # and the column was removed so nothing could join on it. When
    # it is absent the index comparison below is simply not
    # available, which is not a problem to report.
    tg_has_quadrant = "quadrant" in tg.columns
    d = pd.to_datetime(tg.release_date, errors="coerce")
    if d.isna().any():
        problems.append(f"{tg_path.name}: {int(d.isna().sum())} "
                        f"unparseable release dates")

    before = int((d <= want).sum())
    between = int(((d > want) & (d <= other)).sum())
    after = int((d > other).sum())
    print(f"target table {tg_path.name}: {len(tg):,} systems")
    print(f"  on or before {args.expected_cutoff}: {before:,}")
    print(f"  between the two cutoffs:            {between:,}")
    print(f"  after {args.other_cutoff}:          {after:,}")

    if before:
        problems.append(
            f"{tg_path.name}: {before} targets are on or before "
            f"{args.expected_cutoff}; the reference pool has leaked "
            f"into the evaluation targets")
    if between == 0:
        problems.append(
            f"{tg_path.name}: no target falls between the two "
            f"cutoffs. Those systems are targets under "
            f"{args.expected_cutoff} and reference under "
            f"{args.other_cutoff}, so their absence is what a "
            f"{args.other_cutoff} pool looks like")
    else:
        frac = between / len(tg)
        print(f"  -> {frac:.0%} of targets exist only under the "
              f"{args.expected_cutoff} cutoff")

    # ---- the index must carry the same assignment -------------------
    # Existence of the index is checked first and always. Whether
    # the target table still carries a quadrant column only decides
    # whether the comparison can be made, never whether the index
    # itself is verified.
    ix_path = root / args.index
    if not ix_path.exists():
        problems.append(f"{ix_path.name} not found. It is an "
                        f"expected artefact of this study, and an "
                        f"audit that passes without it has not "
                        f"checked the assignment every result uses.")
    else:
        ix = pd.read_parquet(ix_path)
        lack = {"system_id", "quadrant"} - set(ix.columns)
        if lack:
            problems.append(f"{ix_path.name} lacks {sorted(lack)}; "
                            f"index provenance was not checked.")
        elif not tg_has_quadrant:
            notes.append(f"{tg_path.name} carries no quadrant "
                         f"column, so agreement with "
                         f"{ix_path.name} was not tested. The "
                         f"absence is intentional: an earlier build "
                         f"carried one from a different threshold "
                         f"and the index is the authority. See the "
                         f"repository README.")
        else:
            m = ix[["system_id", "quadrant"]].merge(
                tg[["system_id", "quadrant"]], on="system_id",
                how="left", suffixes=("_ix", "_tg"))
            absent = int(m.quadrant_tg.isna().sum())
            disagree = int((m.quadrant_ix != m.quadrant_tg).sum()) \
                - absent
            print(f"\nindex {ix_path.name}: {len(ix):,} systems")
            print(f"  absent from the target table: {absent:,}")
            print(f"  quadrant disagreements:       {disagree:,}")
            if absent:
                problems.append(
                    f"{ix_path.name}: {absent} systems are not in "
                    f"{tg_path.name}; the index was built from a "
                    f"different target pool")
            if disagree:
                ex = m[(m.quadrant_ix != m.quadrant_tg)
                       & m.quadrant_tg.notna()].head(3)
                problems.append(
                    f"{ix_path.name}: {disagree} systems carry a "
                    f"different quadrant from the target table, "
                    f"e.g. {ex.to_dict('records')}")

    # ---- any JSON that records a cutoff -----------------------------
    print(f"\nJSON results recording a cutoff:")
    seen_any = False
    for p in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except Exception as e:
            problems.append(f"{p.relative_to(root)} could not be "
                            f"parsed as JSON: {e}")
            continue
        found = []

        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    if ("cutoff" in k.lower()
                            and isinstance(v, str)
                            and v[:4].isdigit()):
                        found.append((f"{path}{k}", v))
                    walk(v, f"{path}{k}.")
            elif isinstance(o, list):
                # Every element. This runs once and completeness
                # matters more than traversal cost; a wrong cutoff in
                # the fifty-first entry is exactly the kind of thing
                # an audit exists to find.
                for v in o:
                    walk(v, path)

        walk(obj)
        for key, val in found:
            seen_any = True
            ok = val.startswith(args.expected_cutoff)
            flag = "" if ok else "   <-- WRONG CUTOFF"
            print(f"  {p.relative_to(root)}: {key} = {val}{flag}")
            if not ok:
                problems.append(
                    f"{p.relative_to(root)} records {key} = {val}, "
                    f"not {args.expected_cutoff}")
    if not seen_any:
        problems.append("no JSON in the tree records a cutoff. The "
                        "quadrant and pool files are expected to, "
                        "so either they are missing or they stopped "
                        "recording it.")

    # ---- the systems actually extracted -----------------------------
    man = root / "extraction_manifest.parquet"
    if man.exists():
        mf = pd.read_parquet(man)
        if "system_id" not in mf.columns:
            problems.append("extraction_manifest.parquet has no "
                            "system_id; the extracted population "
                            "was not checked.")
        else:
            miss = set(mf.system_id) - set(tg.system_id)
            print(f"\nextraction manifest: {len(mf):,} systems, "
                  f"{len(miss):,} absent from the target table")
            if miss:
                problems.append(
                    f"extraction_manifest.parquet: {len(miss)} "
                    f"extracted systems are not in {tg_path.name}, "
                    f"e.g. {sorted(miss)[:3]}")
    else:
        problems.append("extraction_manifest.parquet not found. It "
                        "records the systems actually run through "
                        "the trunk, so its absence leaves the "
                        "extracted population unverified.")

    # ---- files that look like the other pipeline --------------------
    suspect = [p.relative_to(root) for p in sorted(root.rglob("*"))
               if p.is_file()
               and any(t in p.name.lower() or t in str(p.parent).lower()
                       for t in ("boltz2", "superseded",
                                 args.other_cutoff))]
    if suspect:
        print(f"\nfiles naming the other pipeline, present on disk:")
        # All of them, for the same reason: these are the files to
        # compare against Overleaf by hand, and a truncated list
        # would leave some unchecked.
        for p in suspect:
            print(f"  {p}")
        notes.append(f"{len(suspect)} files name the other pipeline "
                     f"or are marked superseded. Their presence is "
                     f"not an error; reaching the manuscript would "
                     f"be. Check the Overleaf tables against these "
                     f"names.")

    print()
    if problems:
        print(f"{len(problems)} PROBLEM"
              f"{'' if len(problems) == 1 else 'S'}:")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"No problem found. Every artefact checked is "
              f"consistent with the {args.expected_cutoff} cutoff.")
    for n in notes:
        print(f"\n  note: {n}")

    print(f"\n  What this cannot check: a table whose numbers were "
          f"generated correctly\n  and then superseded by a later "
          f"run, or a file in Overleaf that no longer\n  matches the "
          f"one on disk. Compare the manuscript's tables against the "
          f"files\n  in this directory by their contents, not their "
          f"names.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
