#!/usr/bin/env python3
"""
Combine the per-layer null outputs of the job array into the single
threshold file and replicate file the onset audit expects.

WHY THIS REFUSES A PARTIAL SET
------------------------------
The onset is the first layer beginning five consecutive layers above
threshold. A missing layer does not make the estimate noisier; it
changes what "consecutive" means. If layer 23 is absent, layers 22 and
24 become adjacent, and five consecutive rows of the merged table
would span six stages of the trunk. A run that does not exist would be
reported as one.

So the merge requires every layer from 0 to --n-layers-1, names the
ones that are missing, and writes nothing until they are present.

WHAT IS CHECKED BEYOND COMPLETENESS
-----------------------------------
Each per-layer file carries the configuration it was produced under,
and all of them must agree, or the thresholds would come from
differently configured probes. The number of permutations must also
match: a layer calibrated with 50 replicates and another with 100
would give thresholds at different order statistics while looking
identical in the merged table.

Agreement with the CURVE run is checked by calibrate_null.py when each
layer runs, against --curves-meta, and is not re-checked here: that
verification is not recorded in the per-layer metadata, so this script
can establish that the layers agree with one another but not that they
agree with the curves they threshold. The array passes --curves-meta
for exactly that reason, and a layer run without it would have failed
at the time rather than here.

Replicate identity is checked too. calibrate_null.py derives replicate
b's shuffle from a per-replicate seed, so the same b means the same
permutation at every layer. The merged replicate file is only a set of
null TRAJECTORIES if that holds, and the onset audit depends on it, so
each layer must carry the same set of perm_ids.

USAGE
-----
    python3 merge_null.py --indir null_s/ \\
        --out null_tau.parquet --n-layers 48
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
    ap.add_argument("--indir", required=True,
                    help="directory of per-layer null outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-layers", type=int, required=True,
                    help="the trunk's depth; every layer 0..n-1 must "
                         "be present")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite.")
    indir = Path(args.indir)
    if not indir.is_dir():
        sys.exit(f"--indir {indir} is not a directory")

    tau_files = sorted(indir.glob("null_layer_*.parquet"))
    tau_files = [p for p in tau_files if "_replicates" not in p.name]
    if not tau_files:
        sys.exit(f"no per-layer outputs in {indir}")
    print(f"found {len(tau_files)} per-layer threshold files")

    taus, reps, metas = [], [], []
    for p in tau_files:
        t = pd.read_parquet(p)
        if "layer" not in t.columns:
            sys.exit(f"{p.name} has no layer column")
        # One file, one layer. Checking only that the union of layers
        # covers 0..47 would accept a file holding two layers while
        # another per-layer artefact was wrong.
        vals = sorted(set(t["layer"].astype(int)))
        if len(vals) != 1:
            sys.exit(f"{p.name} contains layers {vals}; each array "
                     f"output must hold exactly one layer")
        taus.append(t)
        m = Path(str(p) + ".meta.json")
        if not m.exists():
            sys.exit(f"{p.name} has no .meta.json; its configuration "
                     f"cannot be established")
        metas.append((p.name, json.loads(m.read_text())))
        r = Path(str(p).replace(".parquet", "") + "_replicates.parquet")
        if not r.exists():
            sys.exit(f"{p.name} has no replicate file; the onset audit "
                     f"needs the trajectories, not only the "
                     f"thresholds")
        reps.append(pd.read_parquet(r))

    tau = pd.concat(taus, ignore_index=True)
    rep = pd.concat(reps, ignore_index=True)

    key = ["feature_set", "quadrant", "layer"]
    d = tau.duplicated(key, keep=False)
    if d.any():
        sys.exit(f"duplicate threshold rows, e.g. "
                 f"{tau.loc[d, key].head(3).to_dict('records')}")
    rkey = ["feature_set", "quadrant", "layer", "perm_id"]
    d = rep.duplicated(rkey, keep=False)
    if d.any():
        sys.exit(f"duplicate replicate rows, e.g. "
                 f"{rep.loc[d, rkey].head(3).to_dict('records')}")

    # Every layer must be present. A gap changes what "consecutive"
    # means rather than merely weakening the estimate.
    want = set(range(args.n_layers))
    have = set(tau.layer.astype(int))
    missing = sorted(want - have)
    extra = sorted(have - want)
    if missing:
        sys.exit(f"{len(missing)} layers are missing: "
                 f"{missing[:12]}{' ...' if len(missing) > 12 else ''}\n"
                 f"The onset is defined on the whole trunk, so five "
                 f"consecutive rows of a\ngapped table would span more "
                 f"than five stages. Resubmit the array; finished\n"
                 f"layers are skipped.")
    if extra:
        sys.exit(f"layers outside 0..{args.n_layers - 1}: {extra}")

    # One configuration across layers. Agreement with the curve run
    # was enforced per layer at runtime and is not visible here.
    IGNORE = {"layers", "hours", "n_perm"}
    ref_name, ref = metas[0]
    for name, m in metas[1:]:
        diff = [k for k in set(ref) | set(m)
                if k not in IGNORE and ref.get(k) != m.get(k)]
        if diff:
            sys.exit(f"{name} was produced under a different "
                     f"configuration from {ref_name}; {diff} differ.\n"
                     f"Thresholds from differently configured probes "
                     f"cannot be merged.")
    n_perms = {int(m["n_perm"]) for _, m in metas if "n_perm" in m}
    if len(n_perms) != 1:
        sys.exit(f"the layers used different permutation counts "
                 f"{sorted(n_perms)}.\nThe threshold is an order "
                 f"statistic of B values, so a different B is a "
                 f"different\nquantile even where the table looks the "
                 f"same.")

    # Replicate identity must be shared, or the merged replicate file
    # is a set of unrelated draws rather than trajectories.
    # Not merely that the layers agree with each other, but that each
    # carries the full set 0..B-1: agreeing on the same wrong subset
    # would pass a mutual-equality test.
    B = int(next(iter(n_perms)))
    expected = set(range(B))
    for layer, g in rep.groupby("layer"):
        ids = set(g.perm_id.astype(int))
        if ids != expected:
            short = sorted(ids)[:8]
            sys.exit(f"layer {layer} carries replicate ids "
                     f"{short}{' ...' if len(ids) > 8 else ''} "
                     f"({len(ids)} of {B}); expected 0..{B - 1}.\n"
                     f"The onset audit reads these as trajectories, "
                     f"which requires replicate b to\nbe the same "
                     f"permutation at every layer.")

    tau = tau.sort_values(["feature_set", "quadrant", "layer"])
    rep = rep.sort_values(["feature_set", "quadrant", "perm_id", "layer"])
    tau.to_parquet(out)
    rep_out = Path(str(out).replace(".parquet", "") + "_replicates.parquet")
    rep.to_parquet(rep_out)

    merged = dict(ref)
    merged["layers"] = sorted(have)
    merged["merged_from"] = len(tau_files)
    merged["n_perm"] = n_perms.pop()
    Path(str(out) + ".meta.json").write_text(json.dumps(merged, indent=2))

    print(f"\n{'feature':8}{'cell':6}{'layers':>8}{'tau min':>9}"
          f"{'tau max':>9}")
    for (f, q), g in tau.groupby(["feature_set", "quadrant"]):
        print(f"{f:8}{q:6}{len(g):>8}{g.tau.min():>9.3f}"
              f"{g.tau.max():>9.3f}")

    print(f"\nWrote {out}")
    print(f"Wrote {rep_out}")
    print(f"\n  {len(rep):,} replicate rows, {merged['n_perm']} "
          f"permutations per layer.\n  audit_onset_null.py reads both "
          f"files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
