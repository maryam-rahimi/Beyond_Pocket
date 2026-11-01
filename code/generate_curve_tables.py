#!/usr/bin/env python3
"""
Generate every table and figure describing a depth curve, from the
curve file itself.

No number in the manuscript or in a progress note should be typed by
hand from a terminal. A transcribed figure cannot be regenerated, cannot
be checked against its source, and silently goes stale when the run is
repeated with different data. Everything here is derived from the
parquet `run_depth_probes.py` wrote, so re-running this script after a
new probe run updates every number at once.

Outputs, into --outdir:

    curve_<feature>_<quadrant>.tex   LaTeX table, one row per layer
    curve_<feature>_<quadrant>.csv   same content, for notes and plots
    curve_summary.tex                one row per (feature set, quadrant)
    curve_summary.md                 the same, for progress notes
    curve_stats.json                 the derived quantities, named

curve_summary.md exists so a note can INCLUDE the numbers rather than
copy them. A Markdown table typed out by hand is transcription however
carefully it is done, and goes stale the moment the probes are re-run.

DERIVED QUANTITIES
------------------
Each is computed here rather than read off a plot:

  peak_layer          layer of maximum median AUPRC
  near_peak_layer     first layer reaching 95% of the peak
  l90                 first layer reaching 90% of the gain available
                      above layer 0, measured against the OBSERVED
                      maximum -- so it inherits whatever noise placed
                      that maximum, and is an accumulation depth
                      relative to the peak seen, not a latent
                      saturation point
  final_vs_peak       how far the last layer falls below the peak
  fold_over_baseline  final AUPRC divided by the random-ranking rate
  iqr_near_peak_low/high
                      the MEDIAN across layers (from near_peak_layer on)
                      of the per-layer lower and upper quartiles. Not a
                      single pooled IQR over all proteins and layers.

`near_peak_layer` is named for exactly what it computes: the first
measured layer within 5% of the maximum. It does NOT assert that the
curve stays there -- a curve going 0.34, 0.36, 0.30, 0.37 would have its
near-peak layer at the first 0.36 despite falling afterwards. Calling
that a "plateau" would claim more than the statistic supports.

l90 is the accumulation depth:

    l90 = min { l : M_l >= M_0 + 0.9 (M_max - M_0) }

It asks when the curve has acquired 90% of the improvement AVAILABLE
above its own layer-0 value, rather than 90% of its absolute height.
That distinction matters here: this pilot's FF curve starts at 2.1x the
random baseline, so a fraction-of-absolute statistic would be dominated
by what the first layer already provides, while l90 measures what depth
adds.

l90 is a pre-declared SECONDARY statistic. The primary endpoint remains
the null-calibrated five-stage onset, and both are reported whatever the
outcome. The two answer different questions -- when the signal becomes
detectable, versus when most of the depth-dependent gain has
accumulated -- and the second remains defined when a regime is above the
null threshold from layer 0 and the first cannot resolve.

USAGE
-----
    python3 generate_curve_tables.py \\
        --curves curves_ff_pilot.parquet --outdir curve_tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NEAR_PEAK_TOL = 0.05   # within 5% of the peak
L90_FRACTION = 0.90    # fraction of the gain available above layer 0


def derive(sub: pd.DataFrame) -> dict:
    """Derived quantities for one (feature set, quadrant) curve."""
    sub = sub.sort_values("layer").reset_index(drop=True)

    # The baseline is a property of the test labels, so it must be the
    # same at every layer; a change would mean the curve compares
    # different test populations at different depths. Asserted rather
    # than assumed, since every fold value below divides by it.
    bases = sub.baseline_median.to_numpy()
    if not np.allclose(bases, bases[0], rtol=0, atol=1e-12):
        raise RuntimeError("baseline changes across layers: the curve "
                           "is not scoring one fixed test set")
    if sub.n_systems.nunique() != 1:
        raise RuntimeError("n_systems changes across layers: the curve "
                           "is not scoring one fixed test set")

    peak_i = int(sub.auprc_median.idxmax())
    peak = float(sub.auprc_median.iloc[peak_i])
    near = sub[sub.auprc_median >= peak * (1.0 - NEAR_PEAK_TOL)]
    near_peak_layer = int(near.layer.iloc[0])

    # l90: 90% of the gain available ABOVE layer 0, not 90% of the
    # absolute height. With a curve that starts well above chance the
    # two differ substantially.
    m0 = float(sub.auprc_median.iloc[0])
    target = m0 + L90_FRACTION * (peak - m0)
    reached = sub[sub.auprc_median >= target]
    l90 = int(reached.layer.iloc[0]) if len(reached) else None

    base = float(bases[0])
    final = float(sub.auprc_median.iloc[-1])
    plat = sub[sub.layer >= near_peak_layer]
    return {
        "n_layers_measured": int(len(sub)),
        "n_systems": int(sub.n_systems.iloc[0]),
        "first_layer": int(sub.layer.iloc[0]),
        "first_auprc": float(sub.auprc_median.iloc[0]),
        "peak_layer": int(sub.layer.iloc[peak_i]),
        "peak_auprc": peak,
        "near_peak_layer": near_peak_layer,
        "l90": l90,
        "l90_target_auprc": round(target, 4),
        "final_layer": int(sub.layer.iloc[-1]),
        "final_auprc": final,
        "final_vs_peak": round(final - peak, 4),
        "baseline": base,
        "first_fold_over_baseline": round(
            float(sub.auprc_median.iloc[0]) / base, 2),
        "final_fold_over_baseline": round(final / base, 2),
        "iqr_near_peak_low": round(float(plat.auprc_q25.median()), 3),
        "iqr_near_peak_high": round(float(plat.auprc_q75.median()), 3),
    }


def layer_table(sub: pd.DataFrame, feature: str, quadrant: str) -> str:
    sub = sub.sort_values("layer")
    peak_layer = int(sub.loc[sub.auprc_median.idxmax(), "layer"])
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Pocket decodability by Pairformer depth, feature set "
        rf"\texttt{{{feature}}}, regime {quadrant}. AUPRC is the median "
        r"across test proteins, with the interquartile range; the "
        r"random-ranking baseline is the median positive rate. The peak "
        r"layer is marked.}",
        rf"\label{{tab:curve-{feature}-{quadrant}}}",
        r"\begin{tabular}{rccc}", r"\toprule",
        r"Layer & AUPRC & IQR & Fold over baseline \\", r"\midrule",
    ]
    base = float(sub.baseline_median.iloc[0])
    for r in sub.itertuples():
        mark = r"$^{\star}$" if int(r.layer) == peak_layer else ""
        lines.append(
            f"{int(r.layer)}{mark} & {r.auprc_median:.3f} & "
            f"{r.auprc_q25:.3f}--{r.auprc_q75:.3f} & "
            f"{r.auprc_median / base:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def summary_table(stats: dict) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Depth curve summary. The near-peak layer is the "
        r"first measured layer reaching 95\% of the maximum median "
        r"AUPRC; $\ell_{90}$ is the first reaching 90\% of the gain "
        r"available above layer 0. Fold values are relative to the "
        r"random-ranking baseline.}",
        r"\label{tab:curve-summary}",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Features & Regime & $n$ & Layer 0 fold & Near-peak & "
        r"$\ell_{90}$ & Peak AUPRC & Final fold \\", r"\midrule",
    ]
    for key in sorted(stats):
        feature, quadrant = key.split("/")
        s = stats[key]
        lines.append(
            rf"\texttt{{{feature}}} & {quadrant} & {s['n_systems']} & "
            f"{s['first_fold_over_baseline']:.2f} & "
            f"{s['near_peak_layer']} & "
            f"{s['l90'] if s['l90'] is not None else '--'} & "
            f"{s['peak_auprc']:.3f} & "
            f"{s['final_fold_over_baseline']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curves", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.curves)
    need = {"feature_set", "layer", "quadrant", "auprc_median",
            "auprc_q25", "auprc_q75", "baseline_median", "n_systems"}
    if not need.issubset(df.columns):
        sys.exit(f"curve file lacks {sorted(need - set(df.columns))}")

    stats = {}
    for (feature, quadrant), sub in df.groupby(["feature_set",
                                                "quadrant"]):
        if len(sub) < 2:
            print(f"  skipping {feature}/{quadrant}: only "
                  f"{len(sub)} layer(s)")
            continue
        key = f"{feature}/{quadrant}"
        stats[key] = derive(sub)
        stem = f"curve_{feature}_{quadrant}"
        (outdir / f"{stem}.tex").write_text(
            layer_table(sub, feature, quadrant) + "\n")
        sub.sort_values("layer")[
            ["layer", "auprc_median", "auprc_q25", "auprc_q75",
             "baseline_median", "n_systems"]
        ].to_csv(outdir / f"{stem}.csv", index=False)

    if not stats:
        sys.exit("nothing to tabulate")

    (outdir / "curve_summary.tex").write_text(summary_table(stats) + "\n")

    md = ["| quantity | " + " | ".join(sorted(stats)) + " |",
          "|---|" + "---|" * len(stats)]
    fields = [
        ("systems (test)", "n_systems", "{}"),
        ("layers measured", "n_layers_measured", "{}"),
        ("layer 0 AUPRC", "first_auprc", "{:.3f}"),
        ("layer 0, fold over baseline", "first_fold_over_baseline",
         "{:.2f}"),
        ("first near-peak layer (95% of peak)", "near_peak_layer", "{}"),
        ("l90 (90% of gain above layer 0)", "l90", "{}"),
        ("l90 target AUPRC", "l90_target_auprc", "{:.3f}"),
        ("peak layer", "peak_layer", "{}"),
        ("peak AUPRC", "peak_auprc", "{:.3f}"),
        ("final AUPRC", "final_auprc", "{:.3f}"),
        ("final minus peak", "final_vs_peak", "{:+.3f}"),
        ("final fold over baseline", "final_fold_over_baseline",
         "{:.2f}"),
        ("median Q25 across layers, near-peak on", "iqr_near_peak_low",
         "{:.3f}"),
        ("median Q75 across layers, near-peak on", "iqr_near_peak_high",
         "{:.3f}"),
    ]
    for label, field, fmt in fields:
        cells = []
        for key in sorted(stats):
            v = stats[key].get(field)
            cells.append("--" if v is None else fmt.format(v))
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    (outdir / "curve_summary.md").write_text("\n".join(md) + "\n")
    (outdir / "curve_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"{'curve':16}{'n':>5}{'L0 fold':>9}{'near-pk':>9}{'l90':>6}"
          f"{'peak':>8}{'final':>8}{'fold':>7}")
    for key in sorted(stats):
        st = stats[key]
        print(f"{key:16}{st['n_systems']:>5}"
              f"{st['first_fold_over_baseline']:>9.2f}"
              f"{st['near_peak_layer']:>9}"
              f"{(st['l90'] if st['l90'] is not None else '--'):>6}"
              f"{st['peak_auprc']:>8.3f}{st['final_auprc']:>8.3f}"
              f"{st['final_fold_over_baseline']:>7.2f}")

    print(f"\nWrote {len(stats)*2 + 3} files to {outdir}")
    print("\n  Every number is derived from the curve file. Include\n"
          "  curve_summary.md in a progress note rather than copying\n"
          "  its rows, so re-running the probes updates the note too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
