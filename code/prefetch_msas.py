#!/usr/bin/env python3
"""
Fetch every manifest system's alignment once, on CPU, before any GPU
time is spent.

WHY THIS EXISTS
---------------
The first production run interleaved alignment fetching with extraction
and spent 93% of its wall clock waiting. Measured over 104 systems in
13.6 hours: 0.92 hours of actual work, median 18.7 s per system, the
rest blocked on the public MSA server. Extrapolated, inline fetching
needs about 250 GPU-hours for 1,913 systems against roughly 17 for the
same work with alignments already on disk.

The fix is not a faster GPU or more storage. It is to stop holding a
reserved GPU while waiting on a throttled public endpoint. Fetching
here costs only wall clock.

WHY IT CALLS compute_msa RATHER THAN run_mmseqs2
------------------------------------------------
An earlier version called `run_mmseqs2` directly and wrote its own CSV.
That was wrong for multi-chain systems, and wrong in a way that would
have parsed cleanly and produced a different alignment without
complaint.

Boltz's CSV has a `key` column, and `key` does NOT mean "which chain
this row came from". From compute_msa:

    keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
    ...
    keys = keys + [-1] * len(unpaired)

The key is a row index into the PAIRED alignment, shared across the
per-chain files so Boltz can tell which rows of chain A are homologues
of which rows of chain B. Unpaired rows are -1. Writing the chain index
there instead would tell Boltz that every row of chain A pairs with row
0 and every row of chain B with row 1 -- structurally valid, silently
meaningless.

Single-chain systems would have appeared to work, since compute_msa sets
`paired_msas = [""] * len(data)` when there is one chain, making every
key -1 regardless. That is exactly the kind of bug that passes a
small test and corrupts 14% of the dataset.

So this script calls `compute_msa` itself: the same function the inline
path uses, which runs paired MMseqs2 for multi-chain inputs, runs
unpaired for all, drops the duplicated query row when a paired set
exists, assigns the keys, and writes one CSV per entity. Nothing about
the alignment is reimplemented here.

LAYOUT
------
One directory per system, one CSV per chain, named by the chain's index
in `sequences.json` so the extraction YAML can name them in order:

    msa_cache/<system_id>/0.csv
    msa_cache/<system_id>/1.csv
    msa_cache/<system_id>/.complete

`compute_msa` writes `msa_dir / f"{name}.csv"` for each key of the data
dict, so the keys are chosen to be "0", "1", ... .

The `.complete` marker is written only after every chain's CSV has been
re-read through Boltz's own `parse_csv` and found non-empty. A size
check would accept a malformed file forever; parsing is the same
contract the extractor will apply later.

CONCURRENCY
-----------
Default is one request at a time. The endpoint has already throttled
this client once, `run_mmseqs2` responds to rate limiting by sleeping
and retrying, and several workers would produce several independently
retrying streams -- plausibly making throttling worse rather than
better. Raise it only if measurement shows it helps.

USAGE
-----
    python3 prefetch_msas.py --sequences sequences.json \\
        --outdir msa_cache/ --limit 20        # validate first
    python3 prefetch_msas.py --sequences sequences.json \\
        --outdir msa_cache/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from boltz.main import compute_msa
    from boltz.data.parse.csv import parse_csv
except ImportError as e:
    sys.exit(f"needs the boltz environment: {e}")

DEFAULT_URL = "https://api.colabfold.com"
MARKER = ".complete"
CACHE_MANIFEST = "cache_manifest.json"


def cache_fingerprint(outdir: Path, sequences: dict) -> str:
    """Hash the cached CSVs themselves, deterministically.

    Identifies WHICH alignments a representation collection was built
    from. Without it, regenerating the cache and resuming extraction
    into the same directory leaves one collection resting on two
    different MSA sets under a single run signature -- the files would
    all exist, the signature would match, and nothing would record that
    the inputs had changed.

    Computed here, once, so the GPU job can record a single string
    rather than re-hashing nearly two thousand directories at startup.
    """
    h = hashlib.sha256()
    for sysid in sorted(sequences):
        d = outdir / sysid
        for i in range(len(sequences[sysid]["chains"])):
            p = d / f"{i}.csv"
            if not p.exists():
                continue
            h.update(f"{sysid}/{i}".encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def write_cache_manifest(outdir: Path, sequences: dict, url: str,
                         pairing: str, complete: bool,
                         seq_path: Path) -> dict:
    try:
        import importlib.metadata as _md
        bv = _md.version("boltz")
    except Exception:  # noqa: BLE001
        bv = "unknown"
    try:
        import boltz.model as bm
        root = Path(bm.__file__).parent
        hs = hashlib.sha256()
        for f in sorted(root.rglob("*.py")):
            hs.update(f.relative_to(root).as_posix().encode())
            hs.update(f.read_bytes())
        src = hs.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        src = "unknown"
    man = {
        # The sequences this cache was built from, so the cache artifact
        # is auditable on its own rather than only in combination with
        # the extraction run that consumed it.
        "sequences_sha256": hashlib.sha256(
            seq_path.read_bytes()).hexdigest(),
        "msa_server_url": url,
        "pairing": pairing,
        "boltz_version": bv,
        "boltz_source_sha256": src,
        "systems_expected": len(sequences),
        "systems_cached": sum(
            1 for sid, rec in sequences.items()
            if is_cached(outdir / sid, len(rec["chains"]))),
        "complete": complete,
        "cache_sha256": cache_fingerprint(outdir, sequences),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (outdir / CACHE_MANIFEST).write_text(json.dumps(man, indent=2))
    return man


def chain_csvs(sysdir: Path, n_chains: int) -> list:
    return [sysdir / f"{i}.csv" for i in range(n_chains)]


def clean_tmp(sysdir: Path) -> int:
    """Remove compute_msa's working directories. Returns how many."""
    n = 0
    if sysdir.is_dir():
        for junk in sysdir.iterdir():
            if junk.is_dir() and "_tmp" in junk.name:
                shutil.rmtree(junk, ignore_errors=True)
                n += 1
    return n


def is_cached(sysdir: Path, n_chains: int, verify: bool = False) -> bool:
    """A system counts as cached only if its marker exists and every
    chain CSV is present.

    With --verify, each CSV is additionally re-read through Boltz's own
    parser: a non-empty file can still be malformed, and the alignment
    is worth more than the seconds this costs.
    """
    marker = sysdir / MARKER
    if not marker.exists():
        return False
    # The marker records how many chains it was written for; a directory
    # whose marker disagrees with the sequence record is not this
    # system's cache.
    try:
        if int(json.loads(marker.read_text()).get("chains", -1)) != n_chains:
            return False
    except Exception:  # noqa: BLE001
        return False
    paths = chain_csvs(sysdir, n_chains)
    if not all(p.exists() and p.stat().st_size > 0 for p in paths):
        return False
    if verify:
        for p in paths:
            try:
                msa = parse_csv(p, max_seqs=8192)
                if len(msa.sequences) == 0:
                    return False
            except Exception:  # noqa: BLE001
                return False
    return True


def fetch_one(sysid: str, seqs: list, outdir: Path, url: str,
              pairing: str) -> tuple:
    """Fetch one system's alignments through Boltz's own pipeline."""
    sysdir = outdir / sysid
    n = len(seqs)
    if is_cached(sysdir, n):
        return sysid, "cached", 0

    # Rebuild from scratch: a half-written directory from an interrupted
    # run must not contribute stale chain files to a new attempt.
    if sysdir.exists():
        shutil.rmtree(sysdir)
    sysdir.mkdir(parents=True)

    t0 = time.time()
    try:
        # Keys become the CSV filenames: compute_msa writes
        # msa_dir / f"{name}.csv" for each key of this dict.
        data = {str(i): s for i, s in enumerate(seqs)}
        compute_msa(data=data, target_id=sysid.replace("/", "_"),
                    msa_dir=sysdir, msa_server_url=url,
                    msa_pairing_strategy=pairing)

        # Validate through the same parser the extractor will use,
        # before declaring the system done.
        depths = []
        for p in chain_csvs(sysdir, n):
            if not p.exists():
                raise RuntimeError(f"compute_msa did not write {p.name}")
            msa = parse_csv(p, max_seqs=8192)
            if len(msa.sequences) == 0:
                raise RuntimeError(f"{p.name} parsed to an empty alignment")
            depths.append(len(msa.sequences))

        # compute_msa leaves its working directories behind, named with
        # a suffix after _tmp ("..._paired_tmp_pairgreedy-env",
        # "..._unpaired_tmp_env"), so a "*_tmp" glob misses them. They
        # are raw MMseqs output, not inputs Boltz reads, and removing
        # them later would change cache_sha256 -- so they go now, before
        # the fingerprint is computed.
        for junk in sysdir.iterdir():
            if junk.is_dir() and "_tmp" in junk.name:
                shutil.rmtree(junk, ignore_errors=True)

        (sysdir / MARKER).write_text(json.dumps({
            "chains": n, "depths": depths,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }))
        return sysid, f"ok depths={depths}", round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(sysdir, ignore_errors=True)
        return sysid, f"FAILED {type(e).__name__}: {e}", 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--pairing", default="greedy")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests (default 1; the endpoint "
                         "has already throttled this client and "
                         "run_mmseqs2 retries internally, so more "
                         "workers may make throttling worse)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*",
                    help="fetch only these system ids, for testing")
    ap.add_argument("--verify-cached", action="store_true",
                    help="re-parse cached CSVs instead of trusting the "
                         "completion marker")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sequences = json.loads(Path(args.sequences).read_text())
    print(f"sequences : {len(sequences):,} systems")

    # Sweep any working directories left by an earlier run, before the
    # fingerprint is computed over the cache.
    swept = sum(clean_tmp(outdir / sysid) for sysid in sequences)
    if swept:
        print(f"removed {swept} leftover MMseqs working director"
              f"{'y' if swept == 1 else 'ies'}")

    todo, cached = [], 0
    for sysid, rec in sequences.items():
        if args.only and sysid not in args.only:
            continue
        seqs = [c["sequence"] for c in rec["chains"]]
        if is_cached(outdir / sysid, len(seqs), args.verify_cached):
            cached += 1
            continue
        todo.append((sysid, seqs))
    if args.limit:
        todo = todo[:args.limit]

    multi = sum(1 for _, s in todo if len(s) > 1)
    print(f"cached    : {cached:,}")
    print(f"to fetch  : {len(todo):,}  ({multi:,} multi-chain) "
          f"with {args.workers} worker(s)")
    if not todo:
        # The manifest is still rewritten. A cache populated by several
        # scoped runs has "complete": false in each of their manifests;
        # a finalising run with nothing left to fetch would otherwise
        # exit here and leave that stale flag in place, so a complete
        # cache would look incomplete and the extractor would refuse it.
        print("nothing to fetch")
        have = sum(1 for sid, rec in sequences.items()
                   if is_cached(outdir / sid, len(rec["chains"])))
        complete = have >= len(sequences)
        man = write_cache_manifest(outdir, sequences, args.url,
                                   args.pairing, complete,
                                   Path(args.sequences))
        print(f"{have:,} of {len(sequences):,} systems cached "
              f"({'complete' if complete else 'INCOMPLETE'})")
        print(f"cache_sha256: {man['cache_sha256'][:16]}...")
        print(f"Wrote {outdir/CACHE_MANIFEST}")
        scoped = args.limit is not None or args.only is not None
        return 0 if (complete or scoped) else 1

    t_start = time.time()
    done = failed = 0
    fail_log = outdir / "failures.txt"
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, sid, seqs, outdir, args.url,
                          args.pairing): sid for sid, seqs in todo}
        for fut in as_completed(futs):
            sysid, status, secs = fut.result()
            done += 1
            if status.startswith("FAILED"):
                failed += 1
                with fail_log.open("a") as f:
                    f.write(f"{sysid}\t{status}\n")
                print(f"  [{done}/{len(todo)}] {sysid} {status[:80]}")
            elif done <= 3 or done % 25 == 0:
                rate = (time.time() - t_start) / done
                print(f"  [{done}/{len(todo)}] {sysid} {status} {secs}s "
                      f"| {rate:.0f}s/sys, "
                      f"~{(len(todo)-done)*rate/3600:.1f}h left")

    el = time.time() - t_start
    have = sum(1 for sysid, rec in sequences.items()
               if is_cached(outdir / sysid, len(rec["chains"])))
    print(f"\n{have:,} of {len(sequences):,} systems cached")
    print(f"{failed:,} failures this run"
          + (f", logged in {fail_log.name}" if failed else ""))
    print(f"elapsed {el/3600:.2f}h ({el/max(1,done):.0f}s per system)")

    complete = have >= len(sequences)
    man = write_cache_manifest(outdir, sequences, args.url, args.pairing,
                               complete, Path(args.sequences))
    print(f"cache_sha256: {man['cache_sha256'][:16]}... "
          f"(record this in the extraction run signature)")
    print(f"Wrote {outdir/CACHE_MANIFEST}")

    # A scoped run (--limit or --only) is a deliberate partial fetch, so
    # it must not report failure merely for being partial: a validation
    # of 20 systems exiting non-zero would look like a broken command and
    # would stop any `set -e` wrapper around it.
    scoped = args.limit is not None or args.only is not None
    if scoped:
        print("\n  Scoped run: completeness not asserted. Rerun without "
              "--limit/--only\n  to fetch the rest.")
        return 1 if failed else 0

    if not complete:
        print("\n  Rerun to retry the rest; cached systems are skipped.\n"
              "  Do not start extraction until this is complete, or it "
              "will fetch\n  inline and reintroduce the stall.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
