#!/usr/bin/env python3
"""
Restrict the extraction manifest to systems whose alignments are
cached, preserving the interleaving.

WHAT THIS PRODUCES
------------------
A CACHE-AVAILABLE SUBSET of the manifest: the systems whose alignments
could be obtained. It is not a random sample of the manifest and should
not be described as one.

The public MSA server returned unparsable responses for some systems
and then, on retry, held a request pending for over six hours without
returning. Waiting longer remains possible; six hours establishes that
a request stalled, not that it would never return. This script exists
because the alternative to proceeding is an open-ended wait, and
because generating the missing alignments from a local database would
mix two alignment sources in one collection, which is a worse
provenance problem than a documented subset.

ON WHETHER THE EXCLUSION IS HARMLESS
------------------------------------
It cannot be established that it is. Comparable loss rates across cells
and similar medians are consistent with attrition unrelated to protein
properties, but they do not demonstrate it: the server's behaviour may
depend on sequence characteristics that also bear on the outcome, and
nothing here rules that out.

What this script does is measure the attrition and record every
excluded system, so the subset can be described accurately and the
exclusion revisited. --max-loss is an operational guard against a
grossly uneven loss, not evidence of unbiasedness.

EVERY EXCLUSION REASON IS RECORDED SEPARATELY
---------------------------------------------
A system can be absent from the cache for several reasons, and they are
not equivalent: no sequence record, no completion marker, a marker
disagreeing with the chain count, a missing or empty CSV, or a CSV that
does not parse. Each is counted and recorded per system rather than
collapsed into one narrative.

RE-INTERLEAVING
---------------
Filtering preserves relative order but not balance: if failures cluster
early within one quadrant, a retained prefix can be unbalanced even
when total loss is small. The retained systems are therefore
re-interleaved round-robin, as the original manifest was built, so
every prefix is as balanced as the surviving cell sizes allow.

THE SIGNATURE CHANGES
---------------------
A different manifest means a different extraction run signature, so
this cannot resume into a collection built from the full manifest.
That is the signature working: representations from two different
target sets are two collections.

USAGE
-----
    python3 restrict_manifest_to_cached.py \\
        --manifest extraction_manifest_interleaved.parquet \\
        --sequences sequences.json \\
        --msa-cache msa_cache/ \\
        --out extraction_manifest_cached.parquet
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
MARKER = ".complete"

# Bound at startup when parse checking is requested, so a missing Boltz
# is a refusal rather than a silent pass. See cache_status().
_parse_csv = None


def cache_status(sysdir: Path, n_chains: int, parse: bool) -> str:
    """Why a system is or is not usable, as a specific reason.

    Returns "ok" or a short reason. Reasons are kept distinct because
    they are not the same event: a missing marker means the fetch never
    completed, a CSV that will not parse means it completed and
    produced something unusable, and these would be investigated
    differently.
    """
    # all([]) is True, so a zero-chain record with a matching marker
    # would otherwise pass with no alignment files at all.
    if n_chains < 1:
        return "no chains in sequence record"
    marker = sysdir / MARKER
    if not marker.exists():
        return "no completion marker"
    try:
        recorded = int(json.loads(marker.read_text()).get("chains", -1))
    except Exception:  # noqa: BLE001
        return "unreadable completion marker"
    if recorded != n_chains:
        return f"marker records {recorded} chains, sequences say "\
               f"{n_chains}"
    for i in range(n_chains):
        p = sysdir / f"{i}.csv"
        if not p.exists():
            return f"missing {i}.csv"
        if p.stat().st_size == 0:
            return f"empty {i}.csv"
    if parse:
        # The contract the extractor will apply: Boltz must be able to
        # read these. A non-empty CSV can still fail to parse.
        #
        # parse_csv is imported once at startup, not here: a fallback
        # on ImportError would return "ok" without parsing anything
        # while the metadata still recorded parse_checked: true, which
        # is a false provenance claim rather than a skipped check.
        for i in range(n_chains):
            try:
                msa = _parse_csv(sysdir / f"{i}.csv", max_seqs=8192)
            except Exception as e:  # noqa: BLE001
                return f"{i}.csv does not parse: {type(e).__name__}"
            if len(msa.sequences) == 0:
                return f"{i}.csv parses to an empty alignment"
    return "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--msa-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-loss", type=float, default=15.0,
                    help="refuse if any cell loses more than this "
                         "percent (default 15). An operational guard "
                         "against a grossly uneven loss, not evidence "
                         "that the exclusion is unbiased.")
    ap.add_argument("--no-parse-check", action="store_true",
                    help="skip re-parsing each cached CSV. Faster, but "
                         "a non-empty file that does not parse would "
                         "then reach the extractor.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the within-quadrant order used when "
                         "re-interleaving")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")

    global _parse_csv
    if not args.no_parse_check:
        try:
            from boltz.data.parse.csv import parse_csv as _pc
        except ImportError as e:
            sys.exit(
                f"parse validation needs Boltz, which is not importable "
                f"here ({e}).\nRun this in the Boltz environment, or "
                f"pass --no-parse-check to skip validation\n"
                f"deliberately -- the metadata will record that it was "
                f"skipped.")
        _parse_csv = _pc
        print("parse validation enabled (Boltz available)")
    else:
        print("parse validation SKIPPED (--no-parse-check): a non-empty "
              "CSV that does\n  not parse would reach the extractor")

    man = pd.read_parquet(args.manifest)
    sequences = json.loads(Path(args.sequences).read_text())
    cache = Path(args.msa_cache)
    print(f"manifest: {len(man):,} systems")

    # --- manifest validation -------------------------------------------
    if man.system_id.duplicated().any():
        sys.exit("manifest contains duplicate system_id values")
    if man.system_id.isna().any():
        sys.exit("manifest contains missing system_id values")
    if man.quadrant.isna().any():
        sys.exit("manifest contains missing quadrant values")
    bad_q = sorted(set(man.quadrant) - set(QUADRANTS))
    if bad_q:
        sys.exit(f"unexpected quadrant values: {bad_q}")
    present_q = [q for q in QUADRANTS if (man.quadrant == q).any()]
    if len(present_q) != len(QUADRANTS):
        sys.exit(f"manifest has only {present_q}; all four regimes are "
                 f"required for the comparison this manifest serves")

    reasons = []
    for r in man.itertuples():
        rec = sequences.get(r.system_id)
        if rec is None:
            reasons.append("no sequence record")
            continue
        reasons.append(cache_status(cache / r.system_id,
                                    len(rec.get("chains", [])),
                                    parse=not args.no_parse_check))
    man = man.assign(cache_reason=reasons)
    sel = man[man.cache_reason == "ok"].drop(columns="cache_reason")
    dropped = man[man.cache_reason != "ok"]

    print(f"usable  : {len(sel):,}")
    print(f"excluded: {len(dropped):,} "
          f"({100 * len(dropped) / len(man):.1f}%)")
    if len(dropped):
        # Reasons are reported distinctly: a missing marker and an
        # unparsable CSV are different events with different causes.
        counts = (dropped.cache_reason.str.replace(r"\d+", "N", regex=True)
                  .value_counts())
        print("\nexclusion reasons:")
        for k, v in counts.items():
            print(f"  {v:>5,}  {k}")
    print()

    print(f"{'cell':6}{'kept':>7}{'total':>7}{'lost':>7}{'loss %':>9}")
    worst, cells = 0.0, {}
    for q in QUADRANTS:
        tot = int((man.quadrant == q).sum())
        if tot == 0:
            continue
        lost = int((dropped.quadrant == q).sum())
        pct = 100 * lost / tot
        worst = max(worst, pct)
        cells[q] = {"kept": tot - lost, "total": tot, "lost": lost,
                    "loss_pct": round(pct, 2)}
        print(f"{q:6}{tot - lost:>7,}{tot:>7,}{lost:>7,}{pct:>9.1f}")

    if worst > args.max_loss:
        sys.exit(f"\na cell lost {worst:.1f}%, above --max-loss "
                 f"{args.max_loss}.\nAn exclusion that falls unevenly "
                 f"across cells is a confound, not attrition.")

    # Comparability of what was dropped against what was kept. An
    # exclusion that removed systematically longer proteins or larger
    # pockets would bias the retained set even with even cell rates.
    print(f"\n{'':22}{'kept':>9}{'excluded':>11}")
    for col in ("length", "n_pocket", "S_p"):
        if col not in man.columns:
            continue
        a = float(sel[col].median())
        b = float(dropped[col].median()) if len(dropped) else float("nan")
        print(f"median {col:<15}{a:>9.0f}{b:>11.0f}")

    # Re-interleave. Filtering preserves relative order but not
    # balance: failures clustering early within one quadrant would
    # leave a prefix skewed even when total loss is small. Rebuilt
    # round-robin, as the original manifest was.
    rng = np.random.default_rng(args.seed)
    by_q = {}
    for q in QUADRANTS:
        sub = sel[sel.quadrant == q].reset_index(drop=True)
        by_q[q] = sub.iloc[rng.permutation(len(sub))].reset_index(
            drop=True)
    rows, i = [], 0
    while any(i < len(v) for v in by_q.values()):
        for q in QUADRANTS:
            if i < len(by_q[q]):
                rows.append(by_q[q].iloc[i])
        i += 1
    sel = pd.DataFrame(rows).reset_index(drop=True)

    print(f"\nprefix balance after re-interleaving:")
    print(f"{'prefix':>8}" + "".join(f"{q:>6}" for q in QUADRANTS))
    for n in (100, 200, 400, 800, len(sel)):
        if n > len(sel):
            continue
        c = sel.head(n).quadrant.value_counts()
        print(f"{n:>8}" + "".join(f"{int(c.get(q, 0)):>6}"
                                  for q in QUADRANTS))

    sel.to_parquet(out)
    Path(str(out) + ".meta.json").write_text(json.dumps({
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": hashlib.sha256(
            Path(args.manifest).read_bytes()).hexdigest(),
        "n_before": int(len(man)), "n_after": int(len(sel)),
        "n_excluded": int(len(dropped)),
        "subset": "cache-available: systems whose alignments could be "
                  "obtained. Not a random sample of the manifest.",
        "parse_checked": not args.no_parse_check,
        "reinterleaved": True, "seed": args.seed,
        "cells": cells,
        "exclusions": dict(zip(dropped.system_id,
                               dropped.cache_reason)),
    }, indent=2))
    print(f"\nWrote {out}")
    print(f"Wrote {out}.meta.json, with a reason for every exclusion")
    print("\n  This is a cache-available subset, not a random sample. "
          "Comparable loss\n  rates across cells do not establish that "
          "the exclusion is unrelated to\n  protein properties; report "
          "the attrition rather than discounting it.")
    print("\n  This manifest has a different hash, so extraction cannot "
          "resume into a\n  collection built from the full one. Use a "
          "fresh --outdir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
