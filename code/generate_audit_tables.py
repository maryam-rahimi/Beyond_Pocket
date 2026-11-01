#!/usr/bin/env python3
"""
Generate the onset and false-positive-rate tables from the REAL-DATA
audit output of audit_onset_null.py.

Distinct from generate_onset_tables.py, which tabulates the synthetic
validation in validate_onset_rule.py. That one asks whether a rule
recovers a known onset; this one reports the onset measured on the
study's own representations and the rule's error rate on the study's
own null.

  audit_onset_observed.tex   the onset for each regime, with how many
                             layers exceeded the threshold
  audit_onset_fpr.tex        the leave-one-replicate-out false-positive
                             rate at each candidate persistence
  audit_onset_summary.md     both, for progress notes

WHY THE FPR TABLE MATTERS
-------------------------
The persistence value was fixed at five by a synthetic calibration in
which a three-stage rule reported an onset nineteen stages early. That
is a simulation result. This table asks the same question of the real
representations: how often does each rule fire on trajectories built
from this study's own null?

A rule whose measured rate sits near the nominal level is controlling
error on the data it is applied to. One whose rate is several times
nominal is not, whatever a simulation said.

REPORTING AN ONSET OF ZERO
--------------------------
An onset at the first layer is not a failed measurement, and the table
should not read as though something went wrong. It means the signal
already exceeds the threshold where the trunk begins, so "the layer at
which it first becomes decodable" has no room to vary and cannot
distinguish the regimes.

That is said plainly in the caption, derived from the table rather than
asserted, and the accumulation depth is named as what carries the depth
comparison instead.

USAGE
-----
    python3 generate_audit_tables.py \\
        --onset onset_results.json \\
        --curve-stats curve_tables_final/curve_stats.json \\
        --outdir audit_tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PERSISTENCES = (1, 2, 3, 4, 5, 6, 7)


def fpr_table(d: dict) -> str:
    audit = d["audit"]
    persist = int(d["persist"])
    alpha = float(d["alpha"])
    keys = sorted(audit)
    at_p = [audit[k]["fpr_by_persistence"][str(persist)] for k in keys]
    at_3 = [audit[k]["fpr_by_persistence"].get("3") for k in keys]
    at_3 = [x for x in at_3 if x is not None]
    lo, hi = min(at_p), max(at_p)
    three = (f"{min(at_3):.2f}--{max(at_3):.2f}" if at_3 else "--")

    cols = "l" + "r" * len(PERSISTENCES)
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Empirical false-positive rate of the persistence "
        rf"rule, measured on this study's own null. Each entry is the "
        rf"fraction of null trajectories producing an onset at that "
        rf"persistence, each trajectory tested against a threshold "
        rf"built from the other $B-1$ replicates. The pre-registered "
        rf"value is $p={persist}$, whose measured rate is "
        rf"{lo:.2f}--{hi:.2f} against a nominal $\alpha={alpha:g}$; at "
        rf"$p=3$ it is {three}. Columns other than $p={persist}$ are "
        rf"diagnostics, not alternative endpoints.}}",
        r"\label{tab:onset-fpr}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        "Curve & " + " & ".join(rf"$p={p}$" for p in PERSISTENCES)
        + r" \\", r"\midrule",
    ]
    for k in keys:
        f = audit[k]["fpr_by_persistence"]
        row = " & ".join(
            (f"{f[str(p)]:.2f}" if str(p) in f else "--")
            + (r"$^{\star}$" if p == persist else "")
            for p in PERSISTENCES)
        lines.append(f"\\texttt{{{k}}} & {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def onset_table(d: dict, stats) -> str:
    obs = d["observed"]
    persist = int(d["persist"])
    keys = sorted(obs)
    # all([]) is True, so filtering out the unresolved onsets first
    # would caption an experiment where NOTHING resolved as one where
    # everything resolved at layer zero -- the exact opposite.
    resolved = [obs[k]["onset"] for k in keys]
    all_zero = (bool(resolved)
                and all(v is not None for v in resolved)
                and all(v == 0 for v in resolved))
    all_unresolved = (bool(resolved)
                      and all(v is None for v in resolved))
    saturated = all(obs[k]["n_layers_above_tau"]
                    == obs[k]["n_layers_compared"] for k in keys)

    if all_unresolved:
        note = (rf"No regime produced {persist} adjacent layers above "
                rf"its threshold, so no onset is estimated in any of "
                rf"them. This is reported as unresolved rather than "
                rf"assigned to a layer.")
    elif all_zero and saturated:
        note = (r"Every curve exceeds its threshold at every layer, so "
                r"the onset is the first layer in all regimes and "
                r"cannot distinguish them. The accumulation depth "
                r"$\ell_{90}$, pre-declared as a secondary statistic "
                r"for exactly this case, carries the depth comparison "
                r"instead.")
    elif all_zero:
        note = (r"The onset is the first layer in all regimes and "
                r"cannot distinguish them; $\ell_{90}$ carries the "
                r"depth comparison instead.")
    else:
        note = (rf"An onset reported as unresolved means no {persist} "
                rf"adjacent layers stayed above the threshold.")

    has_l90 = stats is not None
    cols = "lrrr" + ("r" if has_l90 else "")
    header = (r"Curve & Onset & First above $\tau$ & Layers above "
              r"$\tau$" + (r" & $\ell_{90}$" if has_l90 else "") + r" \\")
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Onset of pocket decodability under the "
        rf"pre-registered rule: the first of {persist} adjacent layers "
        rf"whose median AUPRC exceeds the null-calibrated threshold at "
        rf"its own layer. {note}}}",
        r"\label{tab:onset-observed}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header,
        r"\midrule",
    ]
    for k in keys:
        o = obs[k]
        onset = ("unresolved" if o["onset"] is None else str(o["onset"]))
        first = ("--" if o["first_layer_above_tau"] is None
                 else str(o["first_layer_above_tau"]))
        extra = ""
        if has_l90:
            v = stats.get(k, {}).get("l90")
            extra = f" & {v if v is not None else '--'}"
        lines.append(f"\\texttt{{{k}}} & {onset} & {first} & "
                     f"{o['n_layers_above_tau']} of "
                     f"{o['n_layers_compared']}{extra} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onset", required=True)
    ap.add_argument("--curve-stats",
                    help="curve_stats.json, to put l90 beside the onset")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    d = json.loads(Path(args.onset).read_text())
    for k in ("audit", "observed", "persist", "alpha"):
        if k not in d:
            sys.exit(f"{Path(args.onset).name} has no '{k}'")
    if not d["audit"] or not d["observed"]:
        sys.exit("the audit or observed block is empty")
    stats = (json.loads(Path(args.curve_stats).read_text())
             if args.curve_stats else None)
    if stats is not None:
        # The caption may name l90 as the statistic carrying the depth
        # comparison. A stale stats file omitting a curve would then
        # print a dash where that statistic belongs, in a table whose
        # caption says it is the one to read.
        absent = [k for k in d["observed"]
                  if k not in stats or "l90" not in stats[k]]
        if absent:
            sys.exit(f"--curve-stats has no l90 for {absent}; refusing "
                     f"to print a dash where the reported statistic "
                     f"belongs.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "audit_onset_fpr.tex").write_text(fpr_table(d) + "\n")
    (outdir / "audit_onset_observed.tex").write_text(
        onset_table(d, stats) + "\n")

    md = ["## Onset, measured", "",
          "| curve | onset | layers above tau |", "|---|---|---|"]
    for k in sorted(d["observed"]):
        o = d["observed"][k]
        md.append(f"| {k} | "
                  f"{'unresolved' if o['onset'] is None else o['onset']}"
                  f" | {o['n_layers_above_tau']} of "
                  f"{o['n_layers_compared']} |")
    md += ["", "## Leave-one-out false-positive rate", "",
           "| curve | " + " | ".join(f"p={p}" for p in PERSISTENCES)
           + " |", "|---|" + "---|" * len(PERSISTENCES)]
    for k in sorted(d["audit"]):
        f = d["audit"][k]["fpr_by_persistence"]
        md.append(f"| {k} | " + " | ".join(
            (f"{f[str(p)]:.2f}" if str(p) in f else "--")
            for p in PERSISTENCES) + " |")
    (outdir / "audit_onset_summary.md").write_text("\n".join(md) + "\n")

    persist = int(d["persist"])
    at_p = [d["audit"][k]["fpr_by_persistence"][str(persist)]
            for k in d["audit"]]
    at_3 = [d["audit"][k]["fpr_by_persistence"].get("3")
            for k in d["audit"]]
    at_3 = [x for x in at_3 if x is not None]
    print(f"persistence {persist}: FPR {min(at_p):.2f}-{max(at_p):.2f} "
          f"against nominal alpha {d['alpha']:g}")
    if at_3:
        print(f"persistence 3: FPR {min(at_3):.2f}-{max(at_3):.2f}")
    onsets = {k: v["onset"] for k, v in d["observed"].items()}
    print(f"\nonsets: {onsets}")
    vals = list(onsets.values())
    if vals and all(v is None for v in vals):
        print("  none resolved: no regime produced "
              f"{persist} adjacent layers above\n  its threshold.")
    elif vals and all(v is not None and v == 0 for v in vals):
        print("  all zero: the signal exceeds the threshold at the "
              "first layer in every\n  regime, so the onset cannot "
              "distinguish them and l90 carries the depth\n  "
              "comparison.")
    print(f"\nWrote 3 files to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
