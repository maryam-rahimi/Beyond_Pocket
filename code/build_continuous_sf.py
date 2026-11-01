#!/usr/bin/env python3
"""
Build a continuous protein-family support axis to sit beside the
binary one.

WHY
---
The family axis in this study is binary: a target is familiar if any
of its receptor chains shares a cluster with a pre-cutoff system. That
is a low bar. Roughly four targets in five clear it, and the familiar
group spans everything from a family with one pre-cutoff relative to
one with hundreds.

MIRAGE measures the same construct as a count -- the number of
pre-cutoff structures in the target's cluster -- and finds it the
dominant axis for binding affinity, with the gap measured between
singletons and the most heavily represented families. A binary split
cannot see that gradient: both ends of it fall in the same cell.

So the family axis reading as null here is consistent with two
different things. Family support may genuinely not matter for pocket
decodability, or it may matter across a range this design collapses.
The continuous axis distinguishes them.

WHAT IS COMPUTED
----------------
For each target, the number of distinct pre-cutoff systems sharing a
receptor-chain cluster with it:

    S_f_count(i) = |{ j : d_j <= cutoff, cluster(j) ∩ cluster(i) ≠ ∅ }|

counted over systems rather than chains, so a multi-chain pre-cutoff
entry is one unit of support rather than several.

EVERY receptor chain must carry a cluster assignment before a count
is issued. A partly resolved receptor yields a LOWER BOUND, not a
count, and the two failure modes are different in kind:

  A receptor with one mapped chain having no pre-cutoff relatives and
  one unmapped chain would otherwise be given a count of zero. The
  unmapped chain might have relatives; absent and zero are different
  claims.

  A receptor with one mapped chain having twelve relatives and one
  unmapped chain is known to be familiar, but its support is at least
  twelve rather than exactly twelve. A regression consuming that as
  an exact covariate would be consuming a bound.

The binary axis is asymmetric for the first reason: one familiar
chain establishes familiarity, but novelty needs every chain
resolved. A continuous count needs every chain resolved either way,
which is stricter, and the difference is reported.

The count is reported raw and as log10(1 + count), since the
distribution is heavy-tailed and the regression that consumes it
should see the latter.

WHAT THIS IS NOT
----------------
Not a claim that cluster membership measures training exposure. A
pre-cutoff relative was publicly available; whether the model trained
on it is not established by a release date. The same caveat applies to
the binary axis and should be stated once for both.

USAGE
-----
    python3 build_continuous_sf.py \\
        --chain-clusters receptor_chain_clusters.parquet \\
        --all-systems all_systems_with_dates.parquet \\
        --targets targets_with_sp_boltz1.parquet \\
        --cutoff 2021-09-30 \\
        --cutoff-source "Boltz-1 paper, structural training cutoff" \\
        --out targets_with_sf_count.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain-clusters", required=True,
                    help="one row per receptor chain: system_id, "
                         "chain_id, cluster_id, with unmatched chains "
                         "present and cluster_id null")
    ap.add_argument("--all-systems", required=True,
                    help="every system with release_date, not only "
                         "the targets")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--cutoff-source", required=True,
                    help="recorded as an assertion, not a verified "
                         "fact")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    ch = pd.read_parquet(args.chain_clusters)
    al = pd.read_parquet(args.all_systems)
    tg = pd.read_parquet(args.targets)
    for c in ("system_id", "chain_id", "cluster_id"):
        if c not in ch.columns:
            sys.exit(f"--chain-clusters has no '{c}'")
    for c in ("system_id", "release_date"):
        if c not in al.columns:
            sys.exit(f"--all-systems has no '{c}'")
    if "system_id" not in tg.columns:
        sys.exit("--targets has no system_id")

    cut = pd.Timestamp(args.cutoff)
    dates = pd.to_datetime(al.release_date, errors="coerce")
    if dates.isna().any():
        sys.exit(f"{int(dates.isna().sum())} systems have an "
                 f"unparseable release_date; the pre-cutoff pool "
                 f"cannot be formed")
    pre = set(al.loc[dates <= cut, "system_id"])
    print(f"cutoff {args.cutoff}  (asserted: {args.cutoff_source})")
    print(f"pre-cutoff systems {len(pre):,} of {len(al):,}")

    # Which pre-cutoff SYSTEMS sit in each cluster. Systems rather
    # than chains, so a multi-chain pre-cutoff entry contributes one
    # unit of support and not several.
    members = defaultdict(set)
    pre_ch = ch[ch.system_id.isin(pre) & ch.cluster_id.notna()]
    for cid, sid in zip(pre_ch.cluster_id, pre_ch.system_id):
        members[cid].add(sid)
    print(f"clusters with pre-cutoff members {len(members):,}")

    # Per target: the union over its chains' clusters, excluding
    # itself. A target with no assigned chain is unresolved, not zero.
    by_target = {s: g for s, g in ch.groupby("system_id")}
    rows = []
    for sid in tg.system_id:
        g = by_target.get(sid)
        if g is None or len(g) == 0:
            rows.append((sid, None, None, "no_chains"))
            continue
        if g.cluster_id.isna().any():
            # Any unmapped chain makes the count a lower bound. Issued
            # as missing rather than as a number, because a bound in a
            # column of exact values is indistinguishable from one.
            rows.append((sid, None, None, "partially_resolved"))
            continue
        support = set()
        for cid in g.cluster_id.dropna().unique():
            support |= members.get(cid, set())
        support.discard(sid)
        n = len(support)
        rows.append((sid, n, float(np.log10(1 + n)), "resolved"))

    res = pd.DataFrame(rows, columns=["system_id", "S_f_count",
                                      "S_f_log10", "S_f_status"])
    merged = tg.merge(res, on="system_id", how="left",
                      validate="one_to_one")

    known = merged[merged.S_f_status == "resolved"]
    counts = merged.S_f_status.value_counts().to_dict()
    n_unres = int(len(merged) - len(known))
    print(f"\ntargets {len(merged):,}")
    for k, v in sorted(counts.items()):
        print(f"  {k:22}{v:>8,}")
    if n_unres:
        print(f"  -> {n_unres:,} carry no exact count. A partly "
              f"resolved receptor gives a\n     lower bound, and a "
              f"bound in a column of counts cannot be told from "
              f"one.")
    if len(known):
        q = known.S_f_count.quantile([0, .25, .5, .75, .9, 1])
        print(f"support count: min {q[0]:.0f}, median {q[0.5]:.0f}, "
              f"p90 {q[0.9]:.0f}, max {q[1]:.0f}")
        print(f"singletons (count 0): "
              f"{int((known.S_f_count == 0).sum()):,}")

        # The point of the exercise: how much of the range the binary
        # axis collapses.
        if "fam_f" in merged.columns:
            print(f"\n{'binary':10}{'n':>8}{'median':>10}"
                  f"{'p10':>8}{'p90':>8}{'max':>8}")
            for v, name in ((True, "familiar"), (False, "novel")):
                k = known[known.fam_f.astype(bool) == v]
                if k.empty:
                    continue
                print(f"{name:10}{len(k):>8,}"
                      f"{k.S_f_count.median():>10.0f}"
                      f"{k.S_f_count.quantile(.1):>8.0f}"
                      f"{k.S_f_count.quantile(.9):>8.0f}"
                      f"{k.S_f_count.max():>8.0f}")
            fam = known[known.fam_f.astype(bool)]
            if len(fam):
                print(f"\n  the familiar group spans "
                      f"{fam.S_f_count.min():.0f} to "
                      f"{fam.S_f_count.max():.0f} pre-cutoff "
                      f"relatives.\n  A binary split places both ends "
                      f"in the same cell.")

    merged.to_parquet(out)
    Path(str(out) + ".meta.json").write_text(json.dumps({
        "cutoff": args.cutoff,
        "cutoff_source": args.cutoff_source,
        "cutoff_is_asserted_not_verified": True,
        "definition": "number of distinct pre-cutoff systems sharing "
                      "a receptor-chain cluster with the target, "
                      "self excluded",
        "requires": "every receptor chain assigned; a partly "
                    "resolved receptor yields a lower bound and is "
                    "issued as missing",
        "n_targets": int(len(merged)),
        "status_counts": {k: int(v) for k, v in counts.items()},
        "n_without_exact_count": n_unres,
        "n_singletons": int((known.S_f_count == 0).sum())
        if len(known) else None,
    }, indent=2))
    print(f"\nWrote {out}")
    print("\n  A pre-cutoff relative was publicly available. Whether "
          "the model trained\n  on it is not established by a release "
          "date, and that caveat applies to\n  the binary axis "
          "equally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
