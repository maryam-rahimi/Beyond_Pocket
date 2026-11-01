#!/usr/bin/env python3
"""
Build paper tables from validate_onset_rule.py JSON outputs.

Reads one or more result JSONs and emits LaTeX (.tex) and markdown (.md)
tables. Each input JSON is one RULE CONFIGURATION -- a (rho, persistence,
alpha) setting -- and the tables put those configurations side by side, which
is the comparison the validation is actually making.

Tables produced (each only if the underlying data is present):

  onset_recovery   scenario-by-configuration recovered onset vs truth.
                   The core validation result.
  onset_ci         bootstrap interval width per scenario per configuration.
                   Separated from recovery because width and point accuracy
                   fail independently: a configuration can recover the exact
                   layer while its interval spans most of the model.
  delta_control    matched / self-test / magnitude / timing contrasts.
  fpr              false-positive rate of the rule on signal-free data.

USAGE
-----
    python3 generate_onset_tables.py "03 Results/onset_validation"/*.json
    python3 generate_onset_tables.py results/*.json --outdir paper_tables
    python3 generate_onset_tables.py results/*.json --label-from-filename

Configuration labels default to "rho=X, p=N" built from each JSON's own
recorded config, so a mislabelled filename cannot propagate into a table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_runs(paths, label_from_filename=False):
    """Load result JSONs, newest-sorted by (rho, persistence, alpha).

    Labels are built from the recorded config rather than the filename. An
    earlier version of the validation script could silently override an
    explicit --rho 0.0 to 0.9, so a file named *_rho00.json could contain
    rho=0.9 results; reading the label from the payload makes that class of
    mistake impossible to carry into a table.
    """
    runs = []
    for p in paths:
        path = Path(p)
        try:
            with path.open() as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping {path.name}: {e}", file=sys.stderr)
            continue

        cfg = d.get("config", {})
        rho = cfg.get("rho")
        pers = d.get("persistence")
        alpha = d.get("alpha")

        if label_from_filename:
            label = path.stem
        else:
            label = f"$\\rho$={rho}, p={pers}"
            if alpha is not None and abs(alpha - 0.05) > 1e-12:
                label += f", $\\alpha$={alpha}"

        md_label = (label.replace("$\\rho$", "rho")
                         .replace("$\\alpha$", "alpha"))

        runs.append({"path": path, "data": d, "label": label,
                     "md_label": md_label, "rho": rho,
                     "persistence": pers, "alpha": alpha,
                     "n_layers": cfg.get("n_layers"),
                     # The sampling REGIME: how much data each run saw. Two
                     # runs can share rho and persistence yet differ wildly
                     # in these, and the column label shows neither.
                     "regime": (cfg.get("n_proteins"), cfg.get("n_layers"),
                                cfg.get("dim"), d.get("n_null"))})

    runs.sort(key=lambda r: (r["rho"] if r["rho"] is not None else -1,
                             r["persistence"] or 0,
                             -(r["alpha"] or 0)))
    return runs


def check_single_regime(runs, allow_mixed=False):
    """Refuse to tabulate smoke-test and full-calibration runs together.

    A --quick run records 60 proteins, 21 layers and 5 null replicates; a
    --full run records 200, 65 and 200. Placed in adjacent columns they look
    directly comparable, and nothing in the rho/persistence column label says
    otherwise -- so an exploratory setting can silently reach a paper table.
    Keeping quick/ and full/ outputs in separate folders avoids this, but a
    single stray glob defeats that, which is why it is checked here too.
    """
    groups = {}
    for r in runs:
        groups.setdefault(r["regime"], []).append(r["path"].name)
    if len(groups) <= 1:
        return
    lines = ["", "Runs span more than one sampling regime:"]
    for (npro, nlay, dim, nnull), names in sorted(
            groups.items(), key=lambda kv: -(kv[0][0] or 0)):
        lines.append(f"  proteins={npro} layers={nlay} dim={dim} "
                     f"null_reps={nnull}")
        for n in names:
            lines.append(f"      {n}")
    lines += ["",
              "These are not comparable: columns would sit side by side with "
              "nothing in",
              "their labels indicating the difference. Tabulate each regime "
              "separately,",
              "or pass --allow-mixed if the comparison is deliberate."]
    msg = "\n".join(lines)
    if allow_mixed:
        print("WARNING (--allow-mixed):" + msg, file=sys.stderr)
    else:
        sys.exit("ERROR:" + msg)


def short_name(scenario_string):
    """'clean  (strong signal, late onset)' -> 'clean'."""
    return scenario_string.split("(")[0].strip()


def fmt_onset(v):
    return "unres." if v is None else str(v)


# ----------------------------------------------------------------------
# Table 1: onset recovery
# ----------------------------------------------------------------------

def table_onset_recovery(runs):
    scen_order, truths = [], {}
    for r in runs:
        for s in r["data"].get("scenarios", []):
            n = short_name(s["scenario"])
            if n not in scen_order:
                scen_order.append(n)
                truths[n] = s["true_onset"]
    if not scen_order:
        return None, None

    by_run = [{short_name(s["scenario"]): s
               for s in r["data"].get("scenarios", [])} for r in runs]

    cols = "l c " + " ".join("c" for _ in runs)
    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Onset-rule validation on synthetic representations with "
        r"known ground truth. Each column is one rule configuration; each row "
        r"one injected scenario. \emph{True} is the layer at which signal is "
        r"first injected; cells give the recovered onset, with the signed "
        r"error in parentheses. \texttt{unres.} denotes no onset detected, "
        r"which is the correct outcome for the null scenario and an honest "
        r"outcome elsewhere.}",
        r"\label{tab:onset_recovery}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        "Scenario & True & " + " & ".join(r["label"] for r in runs) + r" \\",
        r"\midrule",
    ]
    md = ["## Onset recovery vs known ground truth", "",
          "| Scenario | True | " + " | ".join(r["md_label"] for r in runs) + " |",
          "|---|---|" + "---|" * len(runs)]

    for name in scen_order:
        truth = truths[name]
        truth_s = "--" if truth is None else str(truth)
        tex_cells, md_cells = [], []
        for col in by_run:
            s = col.get(name)
            if s is None:
                tex_cells.append("--"); md_cells.append("--"); continue
            got = s["recovered_onset"]
            if got is None:
                tex_cells.append(r"\texttt{unres.}"); md_cells.append("unres.")
            elif truth is None:
                # Null scenario that resolved: any onset is a false positive.
                tex_cells.append(rf"\textbf{{{got}}} (FP)")
                md_cells.append(f"**{got}** (FP)")
            else:
                err = got - truth
                cell = f"{got} ({err:+d})"
                # Flag early detections: an onset reported BEFORE the signal
                # exists is categorically worse than a late one, because it
                # can invert a depth contrast rather than merely attenuate it.
                if err < 0:
                    tex_cells.append(rf"\textbf{{{cell}}}")
                    md_cells.append(f"**{cell}**")
                else:
                    tex_cells.append(cell); md_cells.append(cell)
        tex.append(f"{name} & {truth_s} & " + " & ".join(tex_cells) + r" \\")
        md.append(f"| {name} | {truth_s} | " + " | ".join(md_cells) + " |")

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md += ["", "Bold marks an onset reported *before* the signal exists "
           "(or at all, in the null scenario).", ""]
    return "\n".join(tex), "\n".join(md)


# ----------------------------------------------------------------------
# Table 2: bootstrap interval width
# ----------------------------------------------------------------------

def table_onset_ci(runs):
    scen_order = []
    for r in runs:
        for s in r["data"].get("scenarios", []):
            n = short_name(s["scenario"])
            if n not in scen_order:
                scen_order.append(n)
    by_run = [{short_name(s["scenario"]): s
               for s in r["data"].get("scenarios", [])} for r in runs]

    any_ci = any(s.get("ci_low") is not None
                 for col in by_run for s in col.values())
    if not any_ci:
        return None, None

    n_layers = next((r["n_layers"] for r in runs if r["n_layers"]), None)

    cols = "l " + " ".join("c" for _ in runs)
    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Protein-level bootstrap interval for the recovered onset, "
        r"by scenario and rule configuration. Width is reported alongside the "
        r"interval because width and point accuracy fail independently: a "
        r"configuration can recover the exact injected layer while its "
        r"interval spans most of the model, so interval width carries "
        r"information the point estimate does not."
        + (rf" Model depth is {n_layers} stages." if n_layers else ""),
        r"\label{tab:onset_ci}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        "Scenario & " + " & ".join(r["label"] for r in runs) + r" \\",
        r"\midrule",
    ]
    md = ["## Bootstrap interval for the recovered onset", "",
          "| Scenario | " + " | ".join(r["md_label"] for r in runs) + " |",
          "|---|" + "---|" * len(runs)]

    for name in scen_order:
        tex_cells, md_cells = [], []
        for col in by_run:
            s = col.get(name)
            if s is None or s.get("ci_low") is None:
                tex_cells.append("--"); md_cells.append("--"); continue
            lo, hi = s["ci_low"], s["ci_high"]
            width = hi - lo
            cell = f"[{lo:.0f}, {hi:.0f}] w={width:.0f}"
            # A lower bound at zero means the interval reaches the input
            # embedding: the bootstrap cannot exclude "present from the start".
            if lo <= 0:
                tex_cells.append(rf"\textbf{{{cell}}}")
                md_cells.append(f"**{cell}**")
            else:
                tex_cells.append(cell); md_cells.append(cell)
        tex.append(f"{name} & " + " & ".join(tex_cells) + r" \\")
        md.append(f"| {name} | " + " | ".join(md_cells) + " |")

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md += ["", "Bold marks an interval whose lower bound reaches layer 0, "
           "i.e. the bootstrap cannot rule out the signal being present from "
           "the input embedding onward.", ""]
    return "\n".join(tex), "\n".join(md)


# ----------------------------------------------------------------------
# Table 3: matched-magnitude delta control
# ----------------------------------------------------------------------

def table_delta_control(runs):
    with_delta = [r for r in runs if r["data"].get("delta_control")]
    if not with_delta:
        return None, None

    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Matched-magnitude control for the onset contrast "
        r"$\Delta_{\mathrm{onset}}$. Paired conditions share a seed, so both "
        r"arms see identical proteins, splits, noise and signal direction and "
        r"differ only in the factor under test. The magnitude row is a "
        r"sensitivity benchmark, not a correction term: it is not an estimand "
        r"and must not be subtracted from a real $\Delta_{\mathrm{onset}}$.}",
        r"\label{tab:delta_control}",
        r"\begin{tabular}{l l c c c}", r"\toprule",
        r"Config & Condition & Expected & Observed (SD) & $n$ resolved \\",
        r"\midrule",
    ]
    md = ["## Matched-magnitude control", "",
          "| Config | Condition | Expected | Observed (SD) | n resolved |",
          "|---|---|---|---|---|"]

    for r in with_delta:
        for c in r["data"]["delta_control"]:
            cond = c["condition"].split("(")[0].strip()
            exp = f"{c['expected_delta']:+d}"
            if c["mean_delta"] is None:
                obs = "all unresolved"
            else:
                obs = f"{c['mean_delta']:+.2f} ({c['sd_delta']:.2f})"
            n = f"{c['n_resolved']}"
            if c.get("n_unresolved"):
                n += f" (+{c['n_unresolved']} unres.)"
            tex.append(f"{r['label']} & {cond} & {exp} & {obs} & {n} " + r"\\")
            md.append(f"| {r['md_label']} | {cond} | {exp} | {obs} | {n} |")

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md.append("")
    return "\n".join(tex), "\n".join(md)


# ----------------------------------------------------------------------
# Table 4: false-positive rate
# ----------------------------------------------------------------------

def table_fpr(runs):
    with_fpr = [r for r in runs if r["data"].get("fpr")]
    if not with_fpr:
        return None, None

    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Held-out false-positive rate of the null-calibrated onset "
        r"rule, measured on signal-free synthetic runs. A rate materially "
        r"above $\alpha$ indicates the persistence requirement is too weak "
        r"for that $\alpha$ at the tested depth correlation.}",
        r"\label{tab:onset_fpr}",
        r"\begin{tabular}{l c c c}", r"\toprule",
        r"Config & $\alpha$ & False onsets & FPR \\", r"\midrule",
    ]
    md = ["## False-positive rate of the onset rule", "",
          "| Config | alpha | False onsets | FPR |", "|---|---|---|---|"]

    for r in with_fpr:
        f = r["data"]["fpr"]
        rate = f["fpr"]
        cell = f"{rate:.3f}"
        flag = rate > (r["alpha"] or 0.05) + 1e-9
        tex_cell = rf"\textbf{{{cell}}}" if flag else cell
        md_cell = f"**{cell}**" if flag else cell
        counts = f"{f['n_false']}/{f['n_seeds']}"
        tex.append(f"{r['label']} & {r['alpha']} & {counts} & {tex_cell} "
                   + r"\\")
        md.append(f"| {r['md_label']} | {r['alpha']} | {counts} | {md_cell} |")

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md += ["", "Bold marks a rate above the nominal alpha.", ""]
    return "\n".join(tex), "\n".join(md)


# ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="result JSON files")
    ap.add_argument("--outdir", default="onset_tables")
    ap.add_argument("--label-from-filename", action="store_true",
                    help="label columns by filename instead of by the "
                         "config recorded inside each JSON")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="permit runs from different sampling regimes "
                         "(e.g. --quick alongside --full) in one table; "
                         "warns instead of aborting")
    args = ap.parse_args(argv)

    runs = load_runs(args.inputs, args.label_from_filename)
    if not runs:
        sys.exit("No readable result JSONs.")
    check_single_regime(runs, args.allow_mixed)

    os.makedirs(args.outdir, exist_ok=True)
    npro, nlay, dim, nnull = runs[0]["regime"]
    print(f"Regime: proteins={npro} layers={nlay} dim={dim} "
          f"null_reps={nnull}")
    print(f"Loaded {len(runs)} run(s):")
    for r in runs:
        parts = []
        if r["data"].get("scenarios"):
            parts.append(f"{len(r['data']['scenarios'])} scenarios")
        if r["data"].get("delta_control"):
            parts.append("delta")
        if r["data"].get("fpr"):
            parts.append("fpr")
        print(f"  {r['path'].name:<34} {r['md_label']:<24} "
              f"[{', '.join(parts) or 'empty'}]")

    builders = [("onset_recovery", table_onset_recovery),
                ("onset_ci", table_onset_ci),
                ("delta_control", table_delta_control),
                ("fpr", table_fpr)]

    written, md_all = [], ["# Onset-rule validation tables", ""]
    for name, fn in builders:
        tex, md = fn(runs)
        if tex is None:
            print(f"  (no data for {name}, skipped)")
            continue
        tp = Path(args.outdir) / f"{name}.tex"
        mp = Path(args.outdir) / f"{name}.md"
        tp.write_text(tex + "\n")
        mp.write_text(md + "\n")
        written += [tp, mp]
        md_all += [md, ""]

    combined = Path(args.outdir) / "all_tables.md"
    combined.write_text("\n".join(md_all) + "\n")
    written.append(combined)

    print(f"\nWrote {len(written)} file(s) to {args.outdir}/:")
    for p in written:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
