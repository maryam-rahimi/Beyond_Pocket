#!/usr/bin/env python3
"""
Join extracted representations to pocket labels and define the splits the
depth probes will use.

Produces one index describing every usable system: its quadrant, its
receptor-family cluster, its length, its positive rate, and which split
it belongs to. The probes read this and the .npz files; they do no
joining or splitting of their own, so the split is defined once and
cannot drift between layers or between feature sets.

CLUSTER-DISJOINT SPLITS
-----------------------
Splits are made over receptor-family CLUSTERS, not over systems. A probe
trained on one member of a family and tested on another is being asked to
recognise a family it has already seen, which is the exact confound this
study exists to measure rather than to commit. Since family membership is
one of the two familiarity axes, leaking it across the split would make
the FF/NN contrast partly circular.

The sampler drew one system per cluster per quadrant, so within a
quadrant clusters and systems coincide. They do not coincide ACROSS
quadrants: the same family can contribute a system to FF and another to
FN when pocket familiarity differs between its members. Splitting per
quadrant would therefore still leak that family. The split is made over
the union of clusters, once, and applied to every quadrant.

FEATURE SETS
------------
Three, run identically:

    s       the per-residue track                     (384 features)
    zpool   the pooled pair summary                   (384 features)
    both    the two concatenated                      (768 features)

`s` is pre-registered as primary; the other two are ablations. Fixing
this before any curve is seen is the point -- choosing the headline
feature set afterwards would be selecting on the result.

The ablation is also informative in its own right. Pockets are a
pairwise property, so if pooled z becomes decodable earlier than s, that
locates pocket structure in the pair track before it reaches the
per-residue one.

WHAT IS EXCLUDED
----------------
  * systems whose representation file is absent or fails to load
  * systems whose stored length disagrees with their label length
  * ineligible systems (pocket annotation not backed for their negatives)
  * systems with no positive residues

Each exclusion is counted and reported, because they are not a random
sample of the target set.

USAGE
-----
    python3 assemble_probe_data.py \\
        --reps reps/systems \\
        --labels pocket_labels_boltz1.parquet \\
        --manifest extraction_manifest.parquet \\
        --out probe_index.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_SETS = ("s", "zpool", "both")
PRIMARY = "s"


def load_rep_meta(path: Path) -> dict | None:
    """Read a representation file's shape metadata without its arrays."""
    try:
        with np.load(path, allow_pickle=False) as d:
            if not {"s", "z_pool", "layers", "length"}.issubset(d.files):
                return None
            sarr, zarr = d["s"], d["z_pool"]
            # A malformed file must not reach the probe. s and z_pool
            # index the same layers and the same residues; if they
            # disagree the two feature sets are not describing the same
            # protein.
            if sarr.ndim != 3 or zarr.ndim != 3:
                return None
            if sarr.shape[0] != zarr.shape[0]:
                return None
            if sarr.shape[1] != zarr.shape[1]:
                return None
            return {
                "n_layers": int(d["s"].shape[0]),
                "length": int(np.asarray(d["length"]).reshape(-1)[0]),
                "s_width": int(d["s"].shape[2]),
                "z_width": int(d["z_pool"].shape[2]),
                "dtype": (str(np.asarray(d["dtype"]).reshape(-1)[0])
                          if "dtype" in d.files else "unknown"),
            }
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", required=True,
                    help="directory of per-system .npz files")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.3,
                    help="fraction of CLUSTERS held out (default 0.3)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite. Pass another --out.")

    man = pd.read_parquet(args.manifest)
    lab = pd.read_parquet(args.labels)
    repdir = Path(args.reps)
    if not repdir.is_dir():
        sys.exit(f"--reps {repdir} is not a directory")

    if man["system_id"].duplicated().any():
        sys.exit("duplicate system_id values in the manifest")
    if lab["system_id"].duplicated().any():
        sys.exit("duplicate system_id values in the label table")
    df = man.merge(lab[["system_id", "length", "eligible", "n_pocket"]],
                   on="system_id", how="left", suffixes=("", "_lab"),
                   validate="one_to_one")
    print(f"manifest: {len(df):,} systems")

    counts = {"no representation file": 0, "unreadable": 0,
              "wrong storage dtype": 0,
              "manifest/label length disagreement": 0,
              "representation/label length disagreement": 0,
              "no family cluster": 0,
              "ineligible": 0, "no positive residues": 0}
    rows = []
    for r in df.itertuples():
        p = repdir / f"{r.system_id}.npz"
        if not p.exists():
            counts["no representation file"] += 1
            continue
        meta = load_rep_meta(p)
        if meta is None:
            counts["unreadable"] += 1
            continue
        # The probe decodes stored arrays as bfloat16 bits. An fp16 file
        # from before that change would decode to plausible-looking
        # nonsense rather than failing, so the dtype is required here
        # rather than assumed.
        if meta["dtype"] != "bfloat16":
            counts["wrong storage dtype"] += 1
            continue

        # The probe is per-residue, so the LABEL length is what the
        # representation must match. r.length is the manifest value and
        # r.length_lab the label value; comparing against the former
        # would check the representation against the manifest and never
        # against the labels -- which is the alignment that matters and
        # the one this check exists to enforce.
        lab_len = int(getattr(r, "length_lab"))
        if int(r.length) != lab_len:
            counts["manifest/label length disagreement"] += 1
            continue
        if meta["length"] != lab_len:
            counts["representation/label length disagreement"] += 1
            continue

        # A system without a family cluster cannot take part in a
        # cluster-disjoint split: `isin(test_clusters)` is False for NaN,
        # so it would silently land in train and quietly break the
        # guarantee the split exists to provide.
        if pd.isna(r.family_cluster):
            counts["no family cluster"] += 1
            continue
        if not bool(getattr(r, "eligible", True)):
            counts["ineligible"] += 1
            continue
        if int(r.n_pocket) <= 0:
            counts["no positive residues"] += 1
            continue
        rows.append({
            "system_id": r.system_id, "quadrant": r.quadrant,
            "family_cluster": r.family_cluster,
            "fam_f": bool(r.fam_f), "S_p": float(r.S_p),
            "length": meta["length"], "n_layers": meta["n_layers"],
            "n_pocket": int(r.n_pocket),
            "positive_rate": int(r.n_pocket) / meta["length"],
            "s_width": meta["s_width"], "z_width": meta["z_width"],
            "path": str(p),
        })

    use = pd.DataFrame(rows)
    if use.empty:
        sys.exit("no usable systems; has extraction produced any files?")
    print(f"usable  : {len(use):,}")
    for k, v in counts.items():
        if v:
            print(f"  excluded, {k}: {v:,}")

    n_layers = sorted(use.n_layers.unique())
    if len(n_layers) != 1:
        sys.exit(f"inconsistent layer counts across systems: {n_layers}. "
                 f"The depth curves assume one common depth.")
    print(f"layers  : {n_layers[0]}")

    # --- cluster-disjoint split -----------------------------------------
    if use["family_cluster"].isna().any():
        sys.exit(f"{int(use['family_cluster'].isna().sum())} usable "
                 f"systems still have no family_cluster; a "
                 f"cluster-disjoint split cannot be constructed.")
    rng = np.random.default_rng(args.seed)
    clusters = np.asarray(sorted(use.family_cluster.unique()),
                          dtype=object)
    rng.shuffle(clusters)
    n_test = max(1, int(round(len(clusters) * args.test_frac)))
    test_clusters = set(clusters[:n_test])
    use["split"] = np.where(use.family_cluster.isin(test_clusters),
                            "test", "train")

    # The guarantee this split exists to provide, asserted rather than
    # assumed: no receptor family may appear on both sides.
    overlap = (set(use.loc[use.split == "train", "family_cluster"])
               & set(use.loc[use.split == "test", "family_cluster"]))
    if overlap:
        sys.exit(f"{len(overlap)} clusters appear in both splits; the "
                 f"split is not cluster-disjoint.")

    print(f"\nsplit over {len(clusters):,} clusters, "
          f"test fraction {args.test_frac}, seed {args.seed}")
    print(f"{'cell':8}{'train':>8}{'test':>7}{'clusters':>10}"
          f"{'med len':>9}{'pos rate':>10}")
    cells = {}
    for q in ("FF", "FN", "NF", "NN"):
        sub = use[use.quadrant == q]
        if sub.empty:
            continue
        tr = int((sub.split == "train").sum())
        te = int((sub.split == "test").sum())
        cells[q] = {"train": tr, "test": te,
                    "clusters": int(sub.family_cluster.nunique()),
                    "median_length": int(sub.length.median()),
                    "median_positive_rate": float(sub.positive_rate.median())}
        print(f"{q:8}{tr:>8,}{te:>7,}{sub.family_cluster.nunique():>10,}"
              f"{sub.length.median():>9,.0f}"
              f"{sub.positive_rate.median():>10.3f}")

    thin = [q for q, c in cells.items() if c["test"] < 20]
    if thin:
        print(f"\n  WARNING: {thin} have fewer than 20 test systems. A "
              f"per-layer AUPRC\n  estimated from that few proteins will "
              f"be noisy enough that the onset\n  rule may fire on noise; "
              f"treat those cells as provisional.")

    use.to_parquet(out)
    meta = {
        "n_systems": int(len(use)), "n_layers": int(n_layers[0]),
        "n_clusters": int(len(clusters)), "test_frac": args.test_frac,
        "seed": args.seed, "cells": cells, "excluded": counts,
        "feature_sets": list(FEATURE_SETS), "primary_feature_set": PRIMARY,
        "s_width": int(use.s_width.iloc[0]),
        "z_width": int(use.z_width.iloc[0]),
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out}")
    print(f"Wrote {out}.meta.json")
    print(f"\n  Feature sets to be run: {', '.join(FEATURE_SETS)}; "
          f"'{PRIMARY}' is primary.\n  Fixed here, before any curve is "
          f"seen, so the headline onset is not\n  chosen by looking at "
          f"the results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
