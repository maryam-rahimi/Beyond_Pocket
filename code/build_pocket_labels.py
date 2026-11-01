#!/usr/bin/env python3
"""
Build per-residue pocket labels for the Beyond Pocket target set.

For each target system, produces a binary vector over its receptor chains:
1 where a residue lines a ligand-binding pocket, 0 otherwise. This is the
y the depth probes predict.

THE RESIDUE COORDINATE SYSTEM
-----------------------------
PLINDER's `ligand_pocket_residues` gives three numbers per residue, and
only one of them indexes the sequence a model tokenizes:

  residue_label_seq_id   polymer sequence position, 1-based, from
                         _atom_site.label_seq_id.  USE THIS.
  residue_index          zero-based position among RESOLVED residues.
                         Not a sequence index.
  residue_auth_seq_id    author/PDB numbering, arbitrary offsets.

`residue_index` is the trap. It equals `label_seq_id - 1` on fully
resolved chains and lags behind it by the number of missing residues
otherwise. Since 75% of PLINDER chains have unresolved residues (median
8), the two disagree for about 68% of pocket residues, with offsets
reaching several thousand. Labels built from `residue_index` are
misaligned on most proteins while looking entirely well-formed.

The mapping used here, `j = residue_label_seq_id - 1`, is confirmed two
ways: PLINDER documents label_seq_id as the polymer position and
chain_sequence as the full SEQRES sequence, and an independent check
against 953 modified-residue records on gapped chains -- each carrying its
own label_seq anchor and parent residue -- found zero positional
mismatches. (21 records resolve to `X`, the code for a non-canonical
residue with no standard parent letter; that is the sequence being honest,
not a position error.)

SYSTEM-SPECIFIC RECORDS, NOT REPRESENTATIVES
--------------------------------------------
Labels come from each system's OWN rows in `ligand_pocket_residues`.
An earlier version instead joined each system to its entry in
`ligand_pocket_membership` and applied the REPRESENTATIVE's residues to
the member. Those are different proteins: membership maps a system to a
similar one, it does not license transferring residue coordinates between
them.

CHAIN INSTANCES
---------------
Keys are `chain_instance` ("1.A", "2.A"), not the bare asym id. In a
multi-copy assembly, stripping the prefix would collapse distinct copies
onto one another and silently overwrite offsets.

WHAT A ZERO MEANS
-----------------
The label answers: does this residue line a pocket of one of the system's
PROPER ligands? Proper excludes ions, crystallization additives and other
artifacts that PLINDER flags via `ligand_is_proper`.

Zeroing every unlisted residue is defensible because the pocket table is
effectively exhaustive at that scope: across a 20,000-system sample its
ligands match the system's proper ligands exactly for 99.2% of systems,
and never exceed them. Measured against ALL annotated ligands the match is
only 78.2%, and the difference is entirely the excluded artifacts -- which
is why the scope has to be stated rather than left implicit.

The consequence is a definitional one worth naming in the manuscript: a
residue contacting only an ion is labelled 0 here, though it is a genuine
contact for something. The task is proper-ligand pocket prediction, not
contact prediction.

For the remaining 0.8%, where pocket rows do NOT cover every proper
ligand, the system's negatives are not fully backed by annotation. Those
systems are retained with a stored system-level `eligible=False` flag
rather than quietly contributing unlabelled positives as negatives.
(A flag, not a per-residue mask: the whole system is in or out.) Systems with no pocket
rows at all are excluded outright.

USAGE
-----
    python3 build_pocket_labels.py \\
        --targets targets_boltz1.parquet \\
        --out pocket_labels.parquet \\
        --sets-out pocket_sets.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plinder.core as pc
except ImportError:
    sys.exit("plinder is required; use "
             "/apps/anaconda/anaconda-2024.10/bin/python")

# Residues that are standard amino acids in their own right rather than
# modifications of anything, so PLINDER writes them as plain one-letter
# codes instead of parenthesised tokens. Their records carry parent "?",
# but the one-letter code is still a usable positional anchor and is
# checked rather than discarded.
SPECIAL_ONE_LETTER = {
    "SEC": "U",   # selenocysteine
    "PYL": "O",   # pyrrolysine
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def verify_mapping(ec, n_chains: int = 400) -> dict:
    """Check `label_seq_id - 1` against independently anchored positions.

    `chain_modified_residues` records carry a label_seq and the residue
    they modify, e.g. "140:SEP:A:19>SER". `chain_sequence_noncanonical`
    independently marks the same residues as parenthesised tokens,
    "...L(SEP)K...", one token per sequence position.

    The test is POSITIONAL: does the parenthesised token sit at index
    `label_seq - 1`? It deliberately does not compare the stated parent
    residue against the canonical letter, because that comparison tests
    PLINDER's chemistry annotation rather than its coordinates, and the
    two can disagree. 1e33 is the case in point: record
    "69:DDZ:A:51>CYS" names CYS as the parent while the sequence carries
    A, and A is right -- DDZ is 3,3-dihydroxyalanine. The coordinate is
    nonetheless exactly correct, with (DDZ) at token 50 of 489 for
    label_seq 51. A parent-letter check would reject a sound mapping over
    a mislabelled parent.

    Restricted to chains with unresolved residues, where
    `label_seq_id - 1` and `residue_index` diverge, so the test
    discriminates between them rather than passing trivially.
    """
    import re
    rec = re.compile(r"[^:]+:([^:]+):[^:]+:(\d+)>(\S*)")
    tok = re.compile(r"\([A-Za-z0-9]+\)|[A-Za-z]")

    # .map(len) assumes every value supports len(); a null would raise
    # before verification could run, turning a data quirk into a crash in
    # the one function meant to catch data quirks.
    has_mod = ec.chain_modified_residues.apply(
        lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0)
    sub = ec[has_mod
             & ec.chain_num_unresolved_residues.fillna(0).gt(5)].head(n_chains)

    ok = bad = skipped = parent_mismatch = unanchorable = 0
    examples = []
    for _, r in sub.iterrows():
        seq, nc = r.chain_sequence, r.chain_sequence_noncanonical
        if not isinstance(seq, str) or not isinstance(nc, str):
            continue
        toks = tok.findall(nc)
        if len(toks) != len(seq):
            # Tokenisation disagrees with the canonical length, so this
            # chain cannot anchor anything; skip rather than guess.
            skipped += 1
            continue
        # Keep the token itself, not just "is non-canonical". Testing
        # identity catches a case membership alone would miss: two
        # modified residues whose positions are swapped would each land
        # on a non-canonical token and pass.
        noncanon = {i: t for i, t in enumerate(toks) if t.startswith("(")}
        for item in r.chain_modified_residues:
            m = rec.match(str(item))
            if not m:
                continue
            comp, lab, parent_raw = m.group(1), int(m.group(2)), m.group(3)
            j = lab - 1
            if not (0 <= j < len(seq)):
                bad += 1
                if len(examples) < 5:
                    examples.append(
                        f"{r.entry_pdb_id} {r.chain_asym_id} {item}: "
                        f"label_seq {lab} out of bounds for length "
                        f"{len(seq)}")
                continue
            want = f"({comp})"
            got = noncanon.get(j)

            # Order matters. An exact parenthesised match is the
            # strongest anchor and is taken first, whatever the record's
            # parent says. Only then do the special cases apply, and only
            # then is a record written off as unanchorable -- so evidence
            # is never discarded while it is still readable.
            here = toks[j]
            if got == want:
                ok += 1
            elif comp in SPECIAL_ONE_LETTER and here == SPECIAL_ONE_LETTER[comp]:
                # Standard amino acid written as its own letter rather
                # than a parenthesised modification: selenocysteine at
                # label 140 lands on U at index 139 in 1aa6, which
                # confirms the coordinate rather than merely failing to
                # contradict it.
                ok += 1
            elif comp in SPECIAL_ONE_LETTER:
                # The expected one-letter code is known and is NOT here.
                # That is evidence against the coordinate, not an absence
                # of evidence, so it must fail rather than be waved past
                # on the strength of a missing parent field.
                bad += 1
                if len(examples) < 5:
                    examples.append(
                        f"{r.entry_pdb_id} {r.chain_asym_id} {item}: "
                        f"token[{j}]={here!r}, expected "
                        f"{SPECIAL_ONE_LETTER[comp]!r} for {comp}")
            elif parent_raw in ("?", ""):
                # Unknown parent and no recognised one-letter form:
                # nothing here anchors a position either way.
                unanchorable += 1
            elif got is not None:
                bad += 1
                if len(examples) < 5:
                    examples.append(
                        f"{r.entry_pdb_id} {r.chain_asym_id} {item}: "
                        f"index {j} holds {got}, expected {want}")
            else:
                bad += 1
                if len(examples) < 5:
                    near = sorted(noncanon, key=lambda k: abs(k - j))[:3]
                    examples.append(
                        f"{r.entry_pdb_id} {r.chain_asym_id} {item}: "
                        f"token[{j}]={here!r}, expected {want}; "
                        f"nearest non-canonical at {near}")

            # Parent-letter agreement is tracked but NOT fatal: 1e33's
            # record names CYS while the sequence carries A, and A is
            # correct for DDZ. That is a chemistry-annotation
            # disagreement, not a coordinate error.
            parent = str(item).split(">")[-1].split()[0] if ">" in str(item) else ""
            exp = THREE_TO_ONE.get(parent)
            if exp is not None and seq[j] != exp and seq[j] != "X":
                parent_mismatch += 1
    return {"ok": bad == 0 and ok > 0, "anchored": ok,
            "misplaced": bad, "chains_skipped": skipped,
            "unanchorable_records": unanchorable,
            "canonical_parent_mismatch": parent_mismatch,
            "examples": examples}


def main(argv=None):  # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets-out")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the residue-mapping check (not advised)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite. Pass another --out.")
    tp = Path(args.targets)
    if not tp.exists():
        sys.exit(f"no such file: {tp}")

    targets = pd.read_parquet(tp)
    if "system_id" not in targets.columns:
        sys.exit("target table has no system_id column")
    want = set(targets.system_id)
    report = {"n_targets": len(want)}
    print(f"targets: {len(want):,}")

    # --- sequences -----------------------------------------------------
    print("\nloading chain sequences ...")
    ec = pc.query_table(
        "entry_chains",
        columns=["entry_pdb_id", "chain_asym_id", "chain_sequence",
                 "chain_sequence_noncanonical", "chain_length",
                 "chain_modified_residues",
                 "chain_num_unresolved_residues"])
    need = {"chain_sequence", "chain_sequence_noncanonical",
            "chain_modified_residues", "chain_num_unresolved_residues"}
    missing_cols = need - set(ec.columns)
    if missing_cols:
        sys.exit(f"entry_chains is missing {sorted(missing_cols)}; "
                 f"verification cannot run.")
    ec = ec[ec.chain_sequence.notna()]
    seqlen = {(e, c): len(s) for e, c, s in
              zip(ec.entry_pdb_id, ec.chain_asym_id, ec.chain_sequence)}
    print(f"  {len(seqlen):,} chains")

    # --- verify the coordinate system ----------------------------------
    if not args.skip_verify:
        print("\nverifying residue mapping (label_seq_id - 1) ...")
        v = verify_mapping(ec)
        report["verification"] = v
        print(f"  anchored {v['anchored']:,} | misplaced "
              f"{v['misplaced']:,} | chains skipped {v['chains_skipped']:,}")
        print(f"  unanchorable records: {v['unanchorable_records']:,} "
              f"(no parenthesised token, e.g. selenocysteine)")
        print(f"  canonical-parent disagreements: "
              f"{v['canonical_parent_mismatch']:,} (tracked, not fatal)")
        for e in v["examples"]:
            print(f"    {e}")
        if not v["ok"]:
            sys.exit("\nresidue-mapping verification FAILED. Do not build "
                     "labels against a\ncoordinate system that does not "
                     "reproduce independently anchored\npositions.")
        print("  confirmed on chains WITH unresolved residues, where "
              "residue_index\n  would have disagreed")

    # --- chain membership ----------------------------------------------
    print("\nloading system chain membership ...")
    ann = pc.query_table(
        "annotation",
        columns=["system_id", "entry_pdb_id",
                 "system_protein_chains_asym_id"]).drop_duplicates("system_id")
    ann = ann[ann.system_id.isin(want)]
    print(f"  {len(ann):,} of {len(want):,} targets found in annotation")
    report["in_annotation"] = int(len(ann))

    # --- pocket residues, each system's own ----------------------------
    # Proper ligands define the label's scope, and comparing them against
    # the pocket table's ligands is what decides whether a system's zeros
    # are backed by annotation.
    print("loading proper-ligand sets ...")
    lig = pc.query_table(
        "annotation",
        columns=["system_id", "ligand_id", "ligand_is_proper"])
    lig = lig[lig.system_id.isin(want) & lig.ligand_is_proper.fillna(False)]
    proper_of = lig.groupby("system_id")["ligand_id"].apply(set).to_dict()
    print(f"  {len(proper_of):,} systems with proper ligands")

    print("loading pocket residues ...")
    pr = pc.query_table(
        "ligand_pocket_residues",
        columns=["system_id", "ligand_id", "chain_instance",
                 "chain_asym_id", "residue_label_seq_id"])
    pr = pr[pr.system_id.isin(want)]
    print(f"  {len(pr):,} residue rows for "
          f"{pr.system_id.nunique():,} target systems")
    report["systems_with_pocket_rows"] = int(pr.system_id.nunique())

    no_rows = len(want) - pr.system_id.nunique()
    if no_rows:
        print(f"  {no_rows:,} targets have NO pocket rows and are excluded "
              f"rather than\n  exported as all-zero vectors: absent "
              f"annotation is not a negative label")

    # --- build ----------------------------------------------------------
    print("\nbuilding labels ...")
    entry_of = dict(zip(ann.system_id, ann.entry_pdb_id))
    chains_of = dict(zip(ann.system_id, ann.system_protein_chains_asym_id))

    rows, setrows = [], []
    counts = {"no annotation": 0, "missing sequence": 0,
              "out of chain bounds": 0, "chain not in system": 0,
              "no proper-ligand pocket rows": 0,
              "ineligible: proper ligand unannotated": 0}

    for sysid, grp in pr.groupby("system_id", sort=False):
        entry = entry_of.get(sysid)
        chains = chains_of.get(sysid)
        if entry is None or chains is None or len(chains) == 0:
            counts["no annotation"] += 1
            continue

        # Offsets keyed by chain INSTANCE, so 1.A and 2.A stay distinct.
        offsets, lengths, order, total, ok = {}, {}, [], 0, True
        for inst in chains:
            inst = str(inst)
            asym = inst.split(".", 1)[1] if "." in inst else inst
            n = seqlen.get((entry, asym))
            if n is None:
                ok = False
                break
            offsets[inst] = total
            lengths[inst] = n
            order.append(inst)
            total += n
        if not ok or total == 0:
            counts["missing sequence"] += 1
            continue

        # Normalise once, so the membership test below cannot fail on a
        # str/object mismatch between the annotation ids and these.
        prop = {str(x) for x in proper_of.get(sysid, set())}

        y = np.zeros(total, dtype=np.int8)
        pockets = {}
        for inst, lab, lig in zip(grp.chain_instance,
                                  grp.residue_label_seq_id,
                                  grp.ligand_id):
            # The target is PROPER-ligand pockets. Without this test a
            # residue contacting only an ion or a crystallization additive
            # becomes a positive, contradicting the stated definition and
            # changing what the probe is trained to find.
            if str(lig) not in prop:
                continue
            inst = str(inst)
            if inst not in offsets:
                counts["chain not in system"] += 1
                continue
            try:
                j = int(lab) - 1
            except (TypeError, ValueError):
                counts["out of chain bounds"] += 1
                continue
            # Bounds are checked against THIS CHAIN, before the offset is
            # applied. Checking only the concatenated total would let an
            # index just past chain A label the first residue of chain B,
            # and a negative index label the last residue of the chain
            # before it.
            if not (0 <= j < lengths[inst]):
                counts["out of chain bounds"] += 1
                continue
            g = offsets[inst] + j
            y[g] = 1
            pockets.setdefault(str(lig), set()).add(int(g))

        if not pockets:
            # Distinguished from "no pocket rows at all": this system HAS
            # pocket rows, but none for a proper ligand. Counting it keeps
            # the filtering visible instead of letting such systems vanish.
            counts["no proper-ligand pocket rows"] += 1
            continue

        # Eligible when the pocket table covers every proper ligand of this
        # system. Where it does not, some proper-ligand contacts are
        # unannotated, so this system's zeros include unlabelled positives
        # and it should not supply negatives to the probe.
        covered = set(pockets)
        missing = prop - covered
        eligible = len(missing) == 0 and len(prop) > 0
        if not eligible:
            counts["ineligible: proper ligand unannotated"] += 1

        rows.append({
            "system_id": sysid, "entry_pdb_id": entry,
            "chain_instances": order, "chain_lengths": [lengths[c] for c in order],
            "length": total, "n_pocket": int(y.sum()),
            "n_pockets": len(pockets), "labels": y.tobytes(),
            "n_proper_ligands": len(prop),
            "n_uncovered_proper": len(missing),
            "eligible": bool(eligible),
        })
        setrows.append({
            "system_id": sysid, "length": total,
            # Keyed by ligand so each set stays traceable to its pocket.
            "pockets": json.dumps({k: sorted(v) for k, v in pockets.items()}),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no labels built; check the target systems")
    df.to_parquet(out)

    frac = df.n_pocket / df.length
    report.update({
        "systems_labelled": int(len(df)),
        "coverage_of_targets": float(len(df) / len(want)),
        "median_length": int(df.length.median()),
        "median_pocket_residues": int(df.n_pocket.median()),
        "median_positive_rate": float(frac.median()),
        "skipped": counts,
    })

    print(f"\nlabelled {len(df):,} of {len(want):,} targets "
          f"({100*len(df)/len(want):.1f}% coverage)")
    print(f"  length         : median {df.length.median():.0f}, "
          f"max {df.length.max():,}")
    print(f"  pocket residues: median {df.n_pocket.median():.0f}")
    print(f"  positive rate  : median {frac.median():.3f}, "
          f"mean {frac.mean():.3f}")
    print(f"  pockets/system : median {df.n_pockets.median():.0f}, "
          f"max {df.n_pockets.max()}")
    n_el = int(df.eligible.sum())
    print(f"  eligible       : {n_el:,} of {len(df):,} "
          f"({100*n_el/len(df):.1f}%)")
    report["eligible_systems"] = n_el
    report["ineligible_systems"] = int(len(df) - n_el)
    if any(counts.values()):
        print("  skipped:")
        for k, n in counts.items():
            if n:
                print(f"    {k}: {n:,}")

    print(f"\n  Ineligible systems are KEPT in the file with eligible=False "
          f"rather than\n  dropped, so the exclusion is visible and "
          f"reversible. Their observed\n  positives remain valid, but the "
          f"positive set may be INCOMPLETE -- some\n  proper ligand lacks "
          f"pocket rows -- and the negatives are therefore not\n  fully "
          f"backed by annotation. Filter on eligible before using them as\n"
          f"  negatives.")
    print(f"\n  Coverage is reported because the targets that drop out are "
          f"not a random\n  sample: they are systems whose pocket "
          f"annotation or sequence is missing.\n  Treating them as negatives "
          f"would manufacture easy true negatives.")
    print(f"\n  The positive rate is the class balance the probe faces. "
          f"At roughly one\n  residue in fifteen, accuracy and AUROC are "
          f"both misleading, which is why\n  the protocol specifies AUPRC.")

    Path(str(out) + ".report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"Wrote {out}.report.json")

    if args.sets_out:
        sp = Path(args.sets_out)
        if sp.exists():
            print(f"NOTE: {sp} exists; sets file not written.")
        else:
            pd.DataFrame(setrows).to_parquet(sp)
            print(f"Wrote {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
