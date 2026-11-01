#!/usr/bin/env python3
"""
Report whether an extraction collection is complete; exit non-zero if not.

The extractor is deliberately tolerant of per-system failures -- it logs
them and carries on -- so it can exit 0 having produced 1,915 of 1,918
systems. A batch scheduler seeing that exit status marks the job
COMPLETED, and the shortfall is never noticed. "The process exited
cleanly" and "the extraction finished" are different claims; this checks
the second.

It reads `manifest_complete` from provenance.json, which the extractor
computes by revalidating every manifest system against the same contract
resume uses, rather than counting files in a directory.

USAGE
-----
    python3 check_complete.py <outdir>          # report, exit 1 if short
    python3 check_complete.py <outdir> --quiet  # one line, always exit 0
"""

import json
import sys
from pathlib import Path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit("usage: check_complete.py <outdir> [--quiet]")
    quiet = "--quiet" in argv
    outdir = Path([a for a in argv if not a.startswith("-")][0])
    prov = outdir / "provenance.json"

    if not prov.exists():
        if quiet:
            print("systems   : no provenance.json")
            return 0
        sys.exit(f"missing {prov} after extraction")

    d = json.loads(prov.read_text())
    done = d.get("systems_on_disk", "?")
    total = d.get("manifest_systems", "?")
    complete = d.get("manifest_complete", False)
    unresolved = d.get("unresolved_failures", "?")

    if quiet:
        print(f"systems   : {done}/{total} "
              f"{'complete' if complete else 'INCOMPLETE'}")
        return 0

    print(f"validated : {done} / {total} systems")
    print(f"complete  : {complete}")
    if not complete:
        sys.exit(f"collection INCOMPLETE: {unresolved} systems unresolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
