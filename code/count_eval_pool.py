#!/usr/bin/env python3
"""
Count the post-cutoff evaluation pool, per candidate frozen trunk.

WHAT THIS DECIDES
-----------------
Every target in the two-axis design must be novel *to the model*, not merely
dissimilar to something. That makes the trunk's training cutoff a hard
boundary: the pre-cutoff pool is what S_f and S_p are measured against, and
only systems released on or after the cutoff can serve as evaluation
targets. A later cutoff therefore buys a cleaner novelty claim at the cost
of a smaller pool.

Documented structural cutoffs for the candidate trunks:

    Boltz-2                       2023-06-01
    Boltz-1 / Boltz-1x            2021-09-30
    Chai-1                        2021-09-30
    AlphaFold 3                   2021-09-30
    Protenix-v1                   2021-09-30

Against a dataset snapshot from 2024-04, Boltz-2 leaves roughly ten months
of releases while the 2021-09-30 models leave about two and a half years.
That difference is the whole question this script answers: is the Boltz-2
window large enough to populate four quadrants, or does the primary trunk
need to be one with an earlier cutoff?

The answer is NOT the row count. Several pockets share a protein and several
proteins share a family cluster, and those are not independent evaluation
units -- the same reason the quadrant feasibility check judges on clusters.
A pool of 4,000 systems drawn from 50 clusters cannot support a four-cell
design no matter how large the first number looks.

SCHEMA
------
Dataset schemas differ and change between releases, so this script does not
assume column names. Run it with --inspect first: it lists the columns and
flags the ones that look like dates, ids and cluster labels. Then pass the
ones you want explicitly.

USAGE
-----
    # what columns does this table actually have?
    python3 count_eval_pool.py --input annotations.parquet --inspect

    # count the pool for every candidate trunk
    python3 count_eval_pool.py --input annotations.parquet \\
        --date-col entry_release_date \\
        --id-col system_id \\
        --cluster-col pli_unique_qcov_cluster

    # one cutoff, written out
    python3 count_eval_pool.py --input annotations.parquet \\
        --date-col entry_release_date --id-col system_id \\
        --cluster-col family_cluster \\
        --cutoffs 2023-06-01 --out pool_counts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas --break-system-packages")


# Documented structural cutoffs. Each is stated in that model's own report;
# none is inferred from a release date. Sources are given so the manuscript
# can cite them rather than repeating a number of unknown provenance.
TRUNK_CUTOFFS = [
    ("Boltz-2",      "2023-06-01", "Boltz-2 report, App. A.1.1"),
    ("Boltz-1",      "2021-09-30", "Boltz-1 report"),
    ("Chai-1",       "2021-09-30", "Chai-1 report"),
    ("AlphaFold 3",  "2021-09-30", "AF3 supplementary"),
    ("Protenix-v1",  "2021-09-30", "Protenix report"),
]

# A conservative PLANNING TARGET, not a kill criterion: 30 clusters per cell
# times four cells, under an idealised allocation in which no cluster is
# shared between cells.
#
# That allocation does not hold. The quadrants partition TARGETS, not
# clusters: one family cluster can contribute a familiar-pocket target to FF
# and a novel-pocket target to FN, and will then be counted in the distinct-
# cluster totals of both. So a pool with fewer than 120 distinct clusters can
# still, in principle, put 30 distinct clusters in each of four cells. A pool
# below this number is a reason to look carefully, not a proof of
# infeasibility. The definitive test is the actual S_f/S_p assignment in
# quadrant_feasibility.py.
MIN_POOL_CLUSTERS = 120


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"No such file: {path}")
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".csv", ".gz", ".tsv", ".txt"):
        sep = "\t" if suf in (".tsv", ".txt") else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    sys.exit(f"Unrecognised extension '{suf}'. Use parquet, csv or tsv.")


def inspect(df: pd.DataFrame) -> None:
    """List columns, flagging plausible date, id and cluster fields."""
    print(f"{len(df.columns)} columns, {len(df):,} rows\n")
    date_like, id_like, cluster_like = [], [], []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("date", "deposit", "release", "year")):
            date_like.append(c)
        elif "cluster" in lc or "_id" == lc[-3:] and "cluster" in lc:
            cluster_like.append(c)
        elif lc.endswith("_id") or lc in ("system_id", "entry_pdb_id",
                                          "pdb_id", "entry_id"):
            id_like.append(c)

    for label, cols in (("date-like", date_like),
                        ("id-like", id_like),
                        ("cluster-like", cluster_like)):
        print(f"{label} columns ({len(cols)}):")
        for c in cols[:25]:
            sample = df[c].dropna().head(2).tolist()
            print(f"  {c:<45} e.g. {sample}")
        if len(cols) > 25:
            print(f"  ... and {len(cols) - 25} more")
        print()

    other = [c for c in df.columns
             if c not in date_like + id_like + cluster_like]
    print(f"other columns ({len(other)}), first 40:")
    for c in other[:40]:
        print(f"  {c}")
    if len(other) > 40:
        print(f"  ... and {len(other) - 40} more")


def parse_dates(df, date_col):
    """Parse the release-date column, reporting what failed."""
    d = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    bad = int(d.isna().sum())
    if bad == len(df):
        sys.exit(f"Column '{date_col}' could not be parsed as dates. "
                 f"First values: {list(df[date_col].head(3))}")
    return d, bad


def count_pool(df, dates, cutoff, id_col, cluster_col, strict=True):
    """Systems and clusters on each side of a cutoff, plus unknown dates.

    Three categories, not two. A row whose date will not parse is UNKNOWN,
    and is reported separately rather than folded into either side. The
    temptation is to let `NaT >= cutoff` return False and sweep those rows
    into the pre-cutoff pool, which is what a two-way split silently does --
    but this count underwrites a novelty claim, and a row of unknown date
    cannot be evidence for or against that claim. Counting it as pre-cutoff
    inflates the reference pool with structures that might postdate it;
    counting it as post-cutoff would put possibly-trained-on structures into
    the evaluation set. It belongs in neither.

    `strict` selects the boundary convention. With strict=True the cutoff
    date itself is treated as PRE-cutoff (post means date > cutoff), which
    is correct when the model reports training on structures "up to" that
    date -- the boundary day is then inside training. Pass strict=False only
    where a report states the cutoff is exclusive.
    """
    valid = dates.notna()
    ts = pd.Timestamp(cutoff)
    post = valid & ((dates > ts) if strict else (dates >= ts))
    pre = valid & ~post
    sub = df[post]

    out = {
        "cutoff": str(cutoff),
        "boundary": "exclusive (date > cutoff)" if strict
                    else "inclusive (date >= cutoff)",
        "pre_systems": int(pre.sum()),
        "post_systems": int(post.sum()),
        "unknown_date_systems": int((~valid).sum()),
    }
    out["post_ids"] = int(sub[id_col].nunique()) if id_col else None
    out["post_clusters"] = (int(sub[cluster_col].nunique())
                            if cluster_col else None)
    out["pre_clusters"] = (int(df[pre][cluster_col].nunique())
                           if cluster_col else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="annotation table (parquet, csv or tsv)")
    ap.add_argument("--inspect", action="store_true",
                    help="list columns and exit, without counting")
    ap.add_argument("--date-col", help="initial release / deposition date")
    ap.add_argument("--id-col", help="system identifier")
    ap.add_argument("--cluster-col",
                    help="family or pocket cluster label. Without this the "
                         "script can only report row counts, which overstate "
                         "usable sample size, so supply it if one exists.")
    ap.add_argument("--cutoffs", nargs="+", default=None,
                    help="cutoff dates to evaluate (YYYY-MM-DD). Default: "
                         "every documented trunk cutoff.")
    ap.add_argument("--min-pool-clusters", type=int, default=MIN_POOL_CLUSTERS,
                    help=f"planning target for post-cutoff clusters, 30 per "
                         f"cell x 4 cells (default {MIN_POOL_CLUSTERS}). A "
                         f"screening heuristic, not a kill criterion: cells "
                         f"can share clusters, so a smaller pool is not "
                         f"proof of infeasibility.")
    ap.add_argument("--inclusive-cutoff", action="store_true",
                    help="treat the cutoff date itself as POST-cutoff "
                         "(date >= cutoff). Default is exclusive "
                         "(date > cutoff), correct when a model reports "
                         "training on structures 'up to' that date, since "
                         "the boundary day is then inside training. Use "
                         "this only where a report states the cutoff is "
                         "exclusive.")
    ap.add_argument("--out", help="write counts JSON here")
    args = ap.parse_args(argv)

    df = load(Path(args.input))

    if args.inspect:
        inspect(df)
        return 0

    if not args.date_col:
        sys.exit("--date-col is required (run --inspect to find it).")
    for col in filter(None, (args.date_col, args.id_col, args.cluster_col)):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not in table. "
                     f"Run --inspect to list what is available.")

    dates, bad = parse_dates(df, args.date_col)
    if bad:
        print(f"NOTE: {bad:,} of {len(df):,} rows have an unparseable "
              f"{args.date_col}. They are reported separately as "
              f"unknown-date and\n      counted in NEITHER pool: a row of "
              f"unknown date cannot be evidence for\n      or against a "
              f"novelty claim.", file=sys.stderr)

    print("=" * 72)
    print("EVALUATION POOL BY TRUNK CUTOFF")
    print("=" * 72)
    print(f"table          : {args.input}")
    print(f"rows           : {len(df):,}")
    print(f"date column    : {args.date_col}")
    print(f"snapshot spans : {dates.min().date()} .. {dates.max().date()}")
    print(f"boundary       : " + ("inclusive (date >= cutoff)"
          if args.inclusive_cutoff else "exclusive (date > cutoff)"))
    if bad:
        print(f"unknown dates  : {bad:,} rows, excluded from both pools")
    if args.cluster_col:
        print(f"cluster column : {args.cluster_col} "
              f"({df[args.cluster_col].nunique():,} distinct overall)")
    else:
        print("cluster column : NOT SUPPLIED -- row counts only, which "
              "overstate\n                 usable sample size. Supply "
              "--cluster-col if one exists.")

    if args.cutoffs:
        rows = [(f"cutoff {c}", c, "user-supplied") for c in args.cutoffs]
    else:
        rows = TRUNK_CUTOFFS

    print(f"\n{'trunk':<16} {'cutoff':<12} {'post rows':>11} "
          f"{'post clusters':>14}   vs planning target")
    print("-" * 72)

    results = []
    for name, cutoff, source in rows:
        r = count_pool(df, dates, cutoff, args.id_col, args.cluster_col,
                       strict=not args.inclusive_cutoff)
        r["trunk"], r["cutoff_source"] = name, source
        results.append(r)

        cl = r["post_clusters"]
        if cl is None:
            verdict = "no cluster column"
        elif cl >= args.min_pool_clusters:
            verdict = "above planning target"
        elif cl >= args.min_pool_clusters // 2:
            verdict = "below target, check"
        else:
            verdict = "well below target"
        cl_s = "--" if cl is None else f"{cl:,}"
        print(f"{name:<16} {cutoff:<12} {r['post_systems']:>11,} "
              f"{cl_s:>14}   {verdict}")

    print("\n" + "=" * 72)
    print("READING THIS")
    print("=" * 72)
    print("  The post-cutoff cluster count is an upper bound on the number "
          "of distinct\n  clusters available to any single quadrant. A "
          "larger pool makes four-cell\n  feasibility more plausible, but "
          "the definitive test is the actual S_f/S_p\n  quadrant assignment "
          "in quadrant_feasibility.py. The target here is a\n  planning "
          "heuristic (30 clusters x 4 cells) that assumes no cluster is\n"
          "  shared between cells; clusters CAN appear in more than one "
          "cell, so a\n  pool below it is a reason to look closely rather "
          "than a proof of failure.")

    if args.cluster_col:
        by_name = {r["trunk"]: r for r in results}
        b2, b1 = by_name.get("Boltz-2"), by_name.get("Boltz-1")
        if b2 and b1 and b2["post_clusters"] is not None:
            if b2["post_clusters"] < args.min_pool_clusters <= (
                    b1["post_clusters"] or 0):
                print(f"\n  Boltz-2's pool ({b2['post_clusters']:,} clusters) "
                      f"is below the planning target while\n  Boltz-1's "
                      f"({b1['post_clusters']:,}) is above it. This does not "
                      f"by itself rule Boltz-2 out\n  -- run the quadrant "
                      f"assignment on both pools before deciding -- but if "
                      f"the\n  four cells do come out thin, the design is "
                      f"salvageable without abandoning\n  the architecture: "
                      f"make Boltz-1 the primary trunk (same family, same "
                      f"open\n  weights, same accessible Pairformer stack, "
                      f"earlier cutoff) and keep\n  Boltz-2 as a second "
                      f"model testing whether the pattern survives a later\n"
                      f"  cutoff.")
            elif b2["post_clusters"] >= args.min_pool_clusters:
                print(f"\n  Boltz-2's pool ({b2['post_clusters']:,} clusters) "
                      f"is above the planning target, so the\n  later cutoff "
                      f"looks affordable. Confirm with the quadrant "
                      f"feasibility check\n  before treating it as settled: "
                      f"a pool can be large and still split badly\n  if S_f "
                      f"and S_p are strongly correlated within it.")

    print("\n  Whatever the pool size, the cutoff is only half the novelty "
          "claim. A\n  post-cutoff release date establishes the structure "
          "was not in the\n  trunk's training set; it does not establish "
          "that a similar family or\n  pocket was absent. Both axes are "
          "still needed, and 'novel' still means\n  below a stated "
          "similarity threshold.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "input": args.input,
            "n_rows": int(len(df)),
            "date_col": args.date_col,
            "cluster_col": args.cluster_col,
            "unknown_date_rows": bad,
            "boundary": ("inclusive" if args.inclusive_cutoff
                         else "exclusive"),
            "snapshot_min": str(dates.min().date()),
            "snapshot_max": str(dates.max().date()),
            "min_pool_clusters": args.min_pool_clusters,
            "results": results,
        }, indent=2))
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
