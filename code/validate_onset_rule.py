#!/usr/bin/env python3
"""
Synthetic validation of the null-calibrated pocket-information onset rule.

WHAT THIS IS FOR
----------------
The onset rule proposed for the pocket-generalization study is

    onset = min { l : M_l > tau, M_{l+1} > tau, M_{l+2} > tau }

where tau is the (1 - alpha) quantile of a label-permuted null distribution.
This is a measuring instrument, and its error properties are unknown until
measured. On real data you cannot measure them, because you do not know the
true onset. This script builds synthetic representations where the true onset
is known by construction, and asks three questions:

  1. RECOVERY   -- when a signal is injected at a known layer, does the rule
                   recover that layer?
  2. FALSE POSITIVES -- when NO signal is injected at any layer, how often does
                   the rule nevertheless declare an onset? This is the
                   held-out false-positive rate of the null-calibration
                   procedure. If it substantially exceeds alpha, the
                   persistence requirement (3 consecutive layers) is too weak
                   and should be tightened before use on real data.
  3. STABILITY  -- does the recovered onset move across random seeds, and how
                   wide is its protein-level bootstrap interval?

Nothing here touches a protein model, a GPU, or any real dataset. It validates
the STATISTIC, not the biology. Run it before committing to a full extraction.

DESIGN NOTES (these mirror the real protocol deliberately)
----------------------------------------------------------
* Train/test splits are by PROTEIN, never by residue. Residues within a
  protein are not independent replicates, and a residue-level split would let
  the probe memorize a protein and inflate every layer's score uniformly.
* The null permutes pocket labels WITHIN each protein, preserving that
  protein's label count. This keeps class balance identical to the real
  problem and destroys only the residue-level correspondence, which is the
  thing the probe is supposed to be reading.
* tau is pooled across layers and null replicates, so it is calibrated
  independently of the observed curve. It is never a function of the curve's
  own maximum or of any top-k order statistic.
* Onset confidence intervals resample TEST PROTEINS against already-fitted
  probes. Refitting per bootstrap replicate would be more thorough but costs
  ~1000x more compute for a diagnostic whose job is to catch gross problems.
* Per-layer scores use average precision (AUPRC), which is the metric the
  pocket protocol specifies for imbalanced residue-level labels.

WHAT IT RUNS
------------
  four scenarios        clean / early / weak / null signal injection
  matched-magnitude     three paired conditions isolating how much apparent
    control             onset delay comes from signal STRENGTH rather than
                        signal TIMING -- the confound that sits directly on
                        top of the study's primary Delta_onset contrast
  FPR estimate          repeated signal-free runs, counting false onsets

USAGE
-----
    python3 validate_onset_rule.py --quick         # fast smoke test
    python3 validate_onset_rule.py                 # defaults
    python3 validate_onset_rule.py --full --out onset_calibration.json
                                                   # the real calibration
    python3 validate_onset_rule.py --rho 0.9       # transformer-like depth
                                                   # correlation
    python3 validate_onset_rule.py --alpha 0.01 --persistence 4
                                                   # tighten the rule
    python3 validate_onset_rule.py --skip-delta --skip-fpr
                                                   # scenarios only

Compare --rho 0 against --rho 0.9: the persistence rule assumes nothing about
depth correlation, but correlated adjacent layers make a chance run of
exceedances more likely, so the rho > 0 false-positive rate is the honest one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


# ----------------------------------------------------------------------
# Synthetic representation generator
# ----------------------------------------------------------------------

@dataclass
class SynthConfig:
    """Parameters of the synthetic landscape."""
    n_proteins: int = 200
    min_residues: int = 40
    max_residues: int = 90
    pocket_frac: float = 0.12     # fraction of residues labelled pocket
    n_layers: int = 65            # Boltz-2 has 64 Pairformer blocks -> 0..64
    dim: int = 32                 # representation width per residue
    noise_sd: float = 1.0
    test_frac: float = 0.30
    rho: float = 0.0              # AR(1) depth correlation; 0 = independent


def make_protein_structure(cfg: SynthConfig, rng: np.random.Generator):
    """Residue counts, protein ids, and pocket labels.

    Pocket residues are placed in one contiguous run per protein, which is
    more realistic than scattering them and makes the task slightly harder
    for a probe that keys on position alone.
    """
    sizes = rng.integers(cfg.min_residues, cfg.max_residues + 1,
                         size=cfg.n_proteins)
    protein_id, labels = [], []
    for p, size in enumerate(sizes):
        n_pocket = max(2, int(round(size * cfg.pocket_frac)))
        y = np.zeros(size, dtype=int)
        start = rng.integers(0, size - n_pocket + 1)
        y[start:start + n_pocket] = 1
        protein_id.append(np.full(size, p))
        labels.append(y)
    return np.concatenate(protein_id), np.concatenate(labels)


def signal_profile(n_layers: int, true_onset: int | None, gain: float,
                   ramp: float = 4.0) -> np.ndarray:
    """Signal strength per layer.

    Exactly zero before `true_onset`, STRICTLY POSITIVE AT `true_onset`, then
    rising smoothly toward `gain`. The hard zero matters: it means any onset
    the rule reports before `true_onset` is unambiguously a false positive,
    not a detection of a faint real signal.

    The `+ 1` in the exponent is load-bearing. Without it the exponent is zero
    at l == true_onset, so 1 - exp(0) = 0 and the first nonzero signal would
    land at true_onset + 1 -- the simulator would inject a one-layer delay
    that the rule would then be blamed for. An earlier version of this file
    had precisely that bug, and it inflated the measured late bias by one
    layer in every signal-bearing scenario.

    `true_onset=None` gives an all-zero profile (the null scenario).
    """
    s = np.zeros(n_layers)
    if true_onset is None:
        return s
    for l in range(true_onset, n_layers):
        s[l] = gain * (1.0 - np.exp(-(l - true_onset + 1) / ramp))
    return s


def build_representations(cfg: SynthConfig, labels: np.ndarray,
                          profile: np.ndarray, rng: np.random.Generator):
    """Yield (layer_index, feature_matrix) one layer at a time.

    Generating lazily keeps peak memory at one layer rather than all n_layers.

    Depth correlation. With cfg.rho = 0 each layer draws independent noise,
    which is convenient but unlike a real transformer, where adjacent blocks
    are residual updates of one another and their representations are highly
    correlated. With cfg.rho > 0 the noise follows an AR(1) process in depth,

        eps_l = rho * eps_{l-1} + sqrt(1 - rho^2) * innovation_l

    which holds the marginal variance at noise_sd while making neighbouring
    layers resemble each other. This matters specifically for the
    persistence requirement: the rule asks for `persistence` CONSECUTIVE
    layers above tau, and consecutive layers being correlated makes a chance
    run of exceedances more likely than independent sampling would suggest.
    The false-positive rate under rho > 0 is therefore the one to trust.

    The signal is added on top of the noise rather than folded into the
    recursion, so the per-layer signal strength is exactly `profile` and the
    declared true onset stays exactly true. Folding it into the recursion
    would let signal accumulate geometrically across depth and the effective
    onset would no longer be the declared one.

    The signal direction is fixed for the whole run, again matching a real
    model more closely than redrawing it per layer: whatever subspace encodes
    pocket identity does not re-randomize between blocks.
    """
    n = labels.shape[0]
    y = labels.astype(float)

    direction = rng.normal(size=cfg.dim)
    direction /= np.linalg.norm(direction)

    eps = None
    for l, strength in enumerate(profile):
        innovation = rng.normal(0.0, cfg.noise_sd, size=(n, cfg.dim))
        if eps is None or cfg.rho <= 0.0:
            eps = innovation
        else:
            eps = cfg.rho * eps + np.sqrt(1.0 - cfg.rho ** 2) * innovation
        X = eps if strength <= 0 else eps + strength * np.outer(y, direction)
        yield l, X


# ----------------------------------------------------------------------
# Probe and null
# ----------------------------------------------------------------------

def protein_level_split(protein_id: np.ndarray, test_frac: float,
                        rng: np.random.Generator):
    """Split by protein. Residues of one protein never straddle the split."""
    uniq = np.unique(protein_id)
    shuffled = rng.permutation(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_proteins = set(shuffled[:n_test].tolist())
    is_test = np.array([p in test_proteins for p in protein_id])
    return ~is_test, is_test


def fit_probe_scores(X, y, train_mask, test_mask):
    """Fit a linear probe on train residues, score test residues.

    Returns None when the training labels are single-class, which cannot
    happen with the generator above but would be a silent disaster on real
    data, so it is checked rather than assumed.
    """
    y_tr = y[train_mask]
    if len(np.unique(y_tr)) < 2:
        return None
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(X[train_mask], y_tr)
    return clf.decision_function(X[test_mask])


def permute_labels_within_protein(y: np.ndarray, protein_id: np.ndarray,
                                  rng: np.random.Generator) -> np.ndarray:
    """Shuffle pocket labels inside each protein, preserving its label count.

    This is the null. It leaves the number of pocket residues per protein
    untouched -- so class balance, and therefore the achievable AUPRC floor,
    is unchanged -- and destroys only which residues those labels belong to.
    A probe reading genuine pocket structure collapses to chance here. A probe
    reading a protein-level nuisance correlate does not, which is exactly what
    this null is meant to expose.
    """
    out = y.copy()
    for p in np.unique(protein_id):
        idx = np.flatnonzero(protein_id == p)
        out[idx] = rng.permutation(y[idx])
    return out


# ----------------------------------------------------------------------
# Onset rule
# ----------------------------------------------------------------------

def find_onset(curve: np.ndarray, tau: float, persistence: int) -> int | None:
    """Earliest layer starting a run of `persistence` layers all above tau.

    Returns None when no such run exists. The pocket protocol is explicit
    that this case is reported as UNRESOLVED and never silently assigned to
    the final layer -- assigning it would turn a non-detection into a
    detection at maximum depth, which is the opposite of what happened.
    """
    n = len(curve)
    for l in range(n - persistence + 1):
        if np.all(curve[l:l + persistence] > tau):
            return l
    return None


def bootstrap_onset_ci(score_cache, y_test, protein_id_test, tau,
                       persistence, n_boot, rng):
    """Protein-level bootstrap CI for the onset.

    Probes stay fixed; only the test proteins are resampled. The resampling
    unit is the protein, matching the evaluation unit, because resampling
    residues would treat correlated residues as independent and give an
    interval that is far too narrow.
    """
    uniq = np.unique(protein_id_test)
    onsets = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(protein_id_test == p)
                              for p in drawn])
        yb = y_test[idx]
        if len(np.unique(yb)) < 2:
            continue
        curve = np.array([average_precision_score(yb, score_cache[l][idx])
                          for l in range(len(score_cache))])
        o = find_onset(curve, tau, persistence)
        if o is not None:
            onsets.append(o)
    if not onsets:
        return None, None, 0.0
    lo, hi = np.percentile(onsets, [2.5, 97.5])
    return float(lo), float(hi), len(onsets) / n_boot


# ----------------------------------------------------------------------
# One scenario, end to end
# ----------------------------------------------------------------------

def run_scenario(name, true_onset, gain, cfg, alpha, persistence,
                 n_null, n_boot, seed, verbose=True):
    rng = np.random.default_rng(seed)

    protein_id, y = make_protein_structure(cfg, rng)
    train_mask, test_mask = protein_level_split(protein_id, cfg.test_frac, rng)
    y_test = y[test_mask]
    pid_test = protein_id[test_mask]

    profile = signal_profile(cfg.n_layers, true_onset, gain)

    # Fix the generator seed so the real pass and every null pass see the
    # SAME representations. Only the labels differ between them -- otherwise
    # the null would be calibrated against different features and would not
    # be a null for this curve.
    gen_seed = int(rng.integers(0, 2**31 - 1))

    # --- real curve -------------------------------------------------------
    real_curve, score_cache = [], []
    for l, X in build_representations(cfg, y, profile,
                                      np.random.default_rng(gen_seed)):
        s = fit_probe_scores(X, y, train_mask, test_mask)
        score_cache.append(s)
        real_curve.append(average_precision_score(y_test, s))
    real_curve = np.array(real_curve)

    # --- null distribution ------------------------------------------------
    null_scores = []
    for r in range(n_null):
        y_null = permute_labels_within_protein(y, protein_id, rng)
        yn_test = y_null[test_mask]
        for l, X in build_representations(cfg, y, profile,
                                          np.random.default_rng(gen_seed)):
            s = fit_probe_scores(X, y_null, train_mask, test_mask)
            if s is not None:
                null_scores.append(average_precision_score(yn_test, s))
    null_scores = np.array(null_scores)
    tau = float(np.quantile(null_scores, 1.0 - alpha))

    onset = find_onset(real_curve, tau, persistence)

    ci_lo = ci_hi = None
    boot_resolved = 0.0
    if onset is not None and n_boot > 0:
        ci_lo, ci_hi, boot_resolved = bootstrap_onset_ci(
            score_cache, y_test, pid_test, tau, persistence, n_boot, rng)

    if verbose:
        print(f"\n--- {name} ---")
        truth = "none (null scenario)" if true_onset is None else str(true_onset)
        print(f"  true onset          : {truth}")
        print(f"  tau (alpha={alpha})     : {tau:.4f}")
        print(f"  null mean / max     : {null_scores.mean():.4f} / "
              f"{null_scores.max():.4f}")
        print(f"  curve min / max     : {real_curve.min():.4f} / "
              f"{real_curve.max():.4f}")
        if onset is None:
            print("  RECOVERED ONSET     : unresolved")
        else:
            ci = (f"[{ci_lo:.0f}, {ci_hi:.0f}]"
                  if ci_lo is not None else "n/a")
            print(f"  RECOVERED ONSET     : {onset}   95% CI {ci}"
                  f"   (resolved in {boot_resolved:.0%} of bootstraps)")
        if true_onset is None:
            print("  VERDICT             : "
                  + ("PASS -- correctly unresolved" if onset is None
                     else f"FAIL -- false onset at layer {onset}"))
        elif onset is None:
            print("  VERDICT             : MISS -- signal present, none found")
        else:
            err = onset - true_onset
            tag = ("PASS" if abs(err) <= 2 else
                   "EARLY (false positive)" if err < 0 else "LATE")
            print(f"  VERDICT             : {tag} (error {err:+d} layers)")

    return {
        "scenario": name,
        "true_onset": true_onset,
        "recovered_onset": onset,
        "tau": tau,
        "null_mean": float(null_scores.mean()),
        "null_max": float(null_scores.max()),
        "curve_min": float(real_curve.min()),
        "curve_max": float(real_curve.max()),
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "bootstrap_resolved_frac": boot_resolved,
        "curve": real_curve.tolist(),
    }


# ----------------------------------------------------------------------
# Matched-magnitude control for the Delta_onset contrast
# ----------------------------------------------------------------------

def run_delta_control(cfg, alpha, persistence, n_null, n_pairs, base_seed):
    """Measure how much apparent onset delay comes from magnitude alone.

    The primary depth contrast of the pocket study is

        Delta_onset = onset(NN) - onset(FF)

    and it is read as a TIMING claim: pocket information becomes accessible
    later under simultaneous novelty. But a weaker signal crosses a fixed
    threshold later even when it begins at exactly the same layer, so a
    positive Delta_onset is also exactly what a pure MAGNITUDE difference
    produces. Since the NN regime is expected to be weaker, the confound
    points the same way as the hypothesis, which is the worst case.

    Three paired conditions separate the two:

      matched    both arms identical (same onset, same gain, different seed)
                 -> Delta should be ~0. This is the noise floor of the
                    contrast: whatever spread appears here is irreducible,
                    and a real Delta must clear it to mean anything.

      magnitude  same true onset, weaker gain in arm B
                 -> Delta should be 0 if the rule measured timing. Whatever
                    it actually reports is FAKE DELAY, purchased entirely
                    with signal strength.

                    This is a SENSITIVITY BENCHMARK, not a correction term.
                    It is not an estimand and must not be subtracted from a
                    real Delta_onset. Its size depends on the gain ratio,
                    ramp shape, SNR and depth injected HERE, none of which
                    are calibrated against real representations, so the
                    number has no transportable scale. What it supports is a
                    comparison: a real Delta_onset that does not exceed the
                    delay a pure magnitude difference produces under a
                    matched synthetic control has not been shown to be a
                    timing effect. Reporting the benchmark alongside the
                    real contrast is the honest use; arithmetic on it is not.

      timing     same gain, genuinely later onset in arm B
                 -> Delta should recover the injected shift. Confirms the
                    rule can see a real delay when there is one.

    Each arm is calibrated against its OWN tau, matching the real protocol
    where every familiarity regime q gets its own tau_q.

    PAIRING. The magnitude and timing conditions are run PAIRED: both arms
    receive the same seed, so they share the same proteins, the same
    train/test split, the same noise realization and the same signal
    direction, and differ ONLY in the factor under test. This works because
    every RNG draw in run_scenario() that precedes the profile -- protein
    construction, the split, and gen_seed itself -- is profile-independent,
    and build_representations() does not branch its RNG calls on signal
    strength. Unpaired arms would differ in proteins, split, noise and
    direction as well as in the manipulated variable, so the resulting Delta
    would mix the causal effect with ordinary between-run variability.

    The matched condition is deliberately left UNPAIRED: its job is to
    measure exactly that between-run variability, and paired it would return
    zero by construction. The paired version of the same condition is run
    anyway, as a self-test -- it must return exactly 0.00 in every pair, and
    anything else means the pairing is leaking and the other two conditions
    cannot be trusted.
    """
    print(f"\n=== Matched-magnitude control ({n_pairs} pairs per condition) ===")

    shift = max(2, int(round(cfg.n_layers * 0.12)))
    base_onset = int(round(cfg.n_layers * 0.31))
    conditions = [
        ("matched   (same onset+gain, UNPAIRED)",
         (base_onset, 1.2), (base_onset, 1.2), 0, False),
        ("selftest  (same onset+gain, paired)",
         (base_onset, 1.2), (base_onset, 1.2), 0, True),
        ("magnitude (same onset, weaker arm B)",
         (base_onset, 1.2), (base_onset, 0.5), 0, True),
        ("timing    (same gain, later arm B)",
         (base_onset, 1.2), (base_onset + shift, 1.2), shift, True),
    ]

    out = []
    for name, (on_a, gain_a), (on_b, gain_b), expected, paired in conditions:
        deltas, unresolved = [], 0
        for p in range(n_pairs):
            seed_a = base_seed + 5000 + p * 2
            # Paired: arm B reuses arm A's seed, so only the injected
            # profile differs. Unpaired: arm B gets its own.
            seed_b = seed_a if paired else seed_a + 1
            ra = run_scenario("A", on_a, gain_a, cfg, alpha, persistence,
                              n_null, 0, seed_a, verbose=False)
            rb = run_scenario("B", on_b, gain_b, cfg, alpha, persistence,
                              n_null, 0, seed_b, verbose=False)
            if ra["recovered_onset"] is None or rb["recovered_onset"] is None:
                unresolved += 1
                continue
            deltas.append(rb["recovered_onset"] - ra["recovered_onset"])
            print(f"  {name[:9]} pair {p + 1}/{n_pairs}", end="\r", flush=True)
        print(" " * 60, end="\r")

        if deltas:
            arr = np.array(deltas, dtype=float)
            mean, sd = float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            lo, hi = (float(np.percentile(arr, 2.5)),
                      float(np.percentile(arr, 97.5))) if len(arr) > 1 else (mean, mean)
            print(f"  {name:<38} expected {expected:+d}   "
                  f"observed {mean:+.2f} (sd {sd:.2f}, range [{lo:+.0f}, {hi:+.0f}])"
                  + (f"   [{unresolved} unresolved]" if unresolved else ""))
        else:
            mean = sd = lo = hi = None
            print(f"  {name:<38} expected {expected:+d}   "
                  f"observed: all {n_pairs} pairs unresolved")

        out.append({"condition": name, "expected_delta": expected,
                    "paired": paired,
                    "mean_delta": mean, "sd_delta": sd,
                    "ci_low": lo, "ci_high": hi,
                    "n_resolved": len(deltas), "n_unresolved": unresolved,
                    "deltas": deltas})

    st = next((c for c in out if c["condition"].startswith("selftest")), None)
    if st is not None:
        ok = st["deltas"] and all(d == 0 for d in st["deltas"])
        if not ok:
            print("\n  *** PAIRING SELF-TEST FAILED ***")
            print("  Two arms given identical seeds AND identical profiles "
                  "returned a\n  nonzero delta. The arms are not actually "
                  "paired, so the magnitude\n  and timing numbers below mix "
                  "the manipulated factor with between-run\n  variability. "
                  "Fix before using either.")
        else:
            print("\n  pairing self-test: PASS "
                  "(identical arms return delta exactly 0)")

    mag = next((c for c in out if c["condition"].startswith("magnitude")), None)
    if mag and mag["mean_delta"] is not None:
        print(f"\n  FAKE DELAY FROM MAGNITUDE ALONE: {mag['mean_delta']:+.2f} layers")
        print("  Sensitivity benchmark, not a correction term -- do NOT "
              "subtract it from a\n  real Delta_onset. Read a real contrast "
              "as a timing effect only if it\n  exceeds the delay a pure "
              "magnitude difference produces under a matched\n  synthetic "
              "control, and report both figures together.")
    return out


# ----------------------------------------------------------------------
# False-positive rate of the rule itself
# ----------------------------------------------------------------------

def estimate_fpr(cfg, alpha, persistence, n_null, n_seeds, base_seed):
    """Fraction of signal-free runs in which the rule declares an onset.

    This is the number the pocket protocol promises to report as the
    "held-out false-positive rate of the null-calibration rule". Compare it
    against alpha. Materially above alpha means the persistence requirement
    is too weak for this alpha: tighten one or the other before using the
    rule on real representations.
    """
    print(f"\n=== False-positive rate over {n_seeds} signal-free runs ===")
    false_onsets = []
    for i in range(n_seeds):
        r = run_scenario(f"fpr_{i}", None, 0.0, cfg, alpha, persistence,
                         n_null, n_boot=0, seed=base_seed + 1000 + i,
                         verbose=False)
        false_onsets.append(r["recovered_onset"])
        print(f"  run {i + 1:>3}/{n_seeds}: "
              + ("clean" if r["recovered_onset"] is None
                 else f"FALSE ONSET at layer {r['recovered_onset']}"),
              end="\r", flush=True)
    n_false = sum(o is not None for o in false_onsets)
    fpr = n_false / n_seeds
    print(" " * 60, end="\r")
    print(f"  false onsets declared : {n_false}/{n_seeds}")
    print(f"  FALSE-POSITIVE RATE   : {fpr:.3f}   (alpha = {alpha})")
    if fpr <= alpha + 1e-9:
        print("  VERDICT               : PASS -- rule controls false onsets")
    elif fpr <= 2 * alpha:
        print("  VERDICT               : MARGINAL -- above alpha, "
              "consider persistence+1 or smaller alpha")
    else:
        print("  VERDICT               : FAIL -- rule does NOT control "
              "false onsets at this setting")
    return {"fpr": fpr, "n_false": n_false, "n_seeds": n_seeds,
            "false_onset_layers": [o for o in false_onsets if o is not None]}


# ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="null quantile for tau (default 0.05)")
    ap.add_argument("--persistence", type=int, default=3,
                    help="consecutive layers required above tau (default 3)")
    ap.add_argument("--n-null", type=int, default=5,
                    help="label-permutation replicates (default 5)")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="protein-level bootstrap replicates (default 200)")
    ap.add_argument("--fpr-seeds", type=int, default=20,
                    help="signal-free runs for the FPR estimate (default 20)")
    ap.add_argument("--rho", type=float, default=None,
                    help="AR(1) depth correlation of the noise, 0-0.99 "
                         "(default 0 = independent layers, or 0.9 under "
                         "--full; try 0.9 for a transformer-like test of the "
                         "persistence rule). An explicit --rho 0.0 is "
                         "honoured even under --full.")
    ap.add_argument("--delta-pairs", type=int, default=8,
                    help="pairs per matched-magnitude condition (default 8)")
    ap.add_argument("--delta-n-null", type=int, default=None,
                    help="null replicates inside the delta control only "
                         "(default: 20 under --full, else --n-null). The "
                         "delta control dominates total cost -- 4 conditions "
                         "x pairs x 2 arms, each arm paying n_null x n_layers "
                         "probe fits -- and needs far fewer replicates than "
                         "the scenarios do: paired arms share a seed, so they "
                         "draw identical null permutations and their taus are "
                         "near-identical, leaving much of the tau noise to "
                         "cancel in the difference. The scenarios and the FPR "
                         "estimate keep the full --n-null, where tau stands "
                         "alone and its noise does not cancel.")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--quick", action="store_true",
                    help="small/fast configuration for a smoke test")
    ap.add_argument("--full", action="store_true",
                    help="serious calibration preset: 200 null replicates, "
                         "100 FPR seeds, 25 delta pairs, rho=0.9")
    ap.add_argument("--skip-fpr", action="store_true")
    ap.add_argument("--skip-delta", action="store_true")
    ap.add_argument("--out", type=str, default=None,
                    help="write results to this JSON path")
    args = ap.parse_args(argv)

    if args.quick and args.full:
        ap.error("--quick and --full are mutually exclusive")

    # Resolve rho from a None sentinel rather than from "is it positive".
    # An earlier version used `args.rho if args.rho > 0 else 0.9` under
    # --full, which silently promoted an EXPLICIT --rho 0.0 to 0.9: the run
    # then reported correlated-depth results under a filename the user had
    # chosen to mean independent layers. Unset and explicitly-zero must stay
    # distinguishable.
    rho_explicit = args.rho is not None
    args.rho = args.rho if rho_explicit else (0.9 if args.full else 0.0)

    cfg = SynthConfig(rho=args.rho)
    if args.quick:
        cfg = SynthConfig(n_proteins=60, min_residues=25, max_residues=45,
                          n_layers=21, dim=16, rho=args.rho)
        args.n_boot = min(args.n_boot, 50)
        args.fpr_seeds = min(args.fpr_seeds, 5)
        args.delta_pairs = min(args.delta_pairs, 3)
    if args.full:
        # Five null replicates is enough to see whether the machinery runs,
        # but nowhere near enough to lock a 95th-percentile threshold: the
        # quantile itself is then estimated from a handful of draws per
        # layer and tau inherits that noise. These are the settings for the
        # calibration you would actually cite.
        cfg = SynthConfig(rho=args.rho)
        args.n_null = max(args.n_null, 200)
        args.fpr_seeds = max(args.fpr_seeds, 100)
        args.delta_pairs = max(args.delta_pairs, 25)

    # Scale injected onsets to the configured depth so --quick stays meaningful.
    late = int(round(cfg.n_layers * 0.31))
    early = int(round(cfg.n_layers * 0.08))

    print("=" * 68)
    print("ONSET-RULE VALIDATION ON SYNTHETIC REPRESENTATIONS")
    print("=" * 68)
    print(f"proteins={cfg.n_proteins}  layers={cfg.n_layers}  dim={cfg.dim}  "
          f"pocket_frac={cfg.pocket_frac}  rho={cfg.rho}")
    print(f"alpha={args.alpha}  persistence={args.persistence}  "
          f"null_reps={args.n_null}  bootstrap={args.n_boot}")
    if args.n_null < 50:
        print(f"\n  NOTE: --n-null {args.n_null} is a smoke-test setting. The "
              f"{1 - args.alpha:.0%} quantile\n  defining tau is estimated "
              "from these replicates, so tau is itself noisy here.\n  Use "
              "--full (or --n-null 200 --fpr-seeds 100) for a calibration "
              "you would cite.")

    # Cost preview. Every probe fit is one logistic regression; the totals
    # below are what actually determines wall time, and the delta control
    # dominates unless its null count is reduced. Printing this up front is
    # cheaper than discovering it two pairs into a multi-day run.
    _dnn = args.delta_n_null if args.delta_n_null is not None else (
        20 if args.full else args.n_null)
    per_arm = cfg.n_layers * (1 + args.n_null)
    per_delta_arm = cfg.n_layers * (1 + _dnn)
    c_scen = 4 * per_arm
    c_delta = 0 if args.skip_delta else 4 * args.delta_pairs * 2 * per_delta_arm
    c_fpr = 0 if args.skip_fpr else args.fpr_seeds * per_arm
    total = c_scen + c_delta + c_fpr
    print(f"\n  cost preview (probe fits): scenarios {c_scen:,}  "
          f"delta {c_delta:,}  fpr {c_fpr:,}")
    print(f"  TOTAL {total:,} fits. At ~50 ms/fit that is roughly "
          f"{total * 0.05 / 3600:.1f} h single-core.")
    print("  Time one section first (--skip-delta --skip-fpr) if that "
          "estimate matters.")

    scenarios = [
        ("clean  (strong signal, late onset)",  late,  1.2),
        ("early  (strong signal, early onset)", early, 1.2),
        ("weak   (low-gain signal, late onset)", late, 0.35),
        ("null   (no signal at any layer)",     None,  0.0),
    ]

    results = []
    for i, (name, onset, gain) in enumerate(scenarios):
        results.append(run_scenario(name, onset, gain, cfg, args.alpha,
                                    args.persistence, args.n_null,
                                    args.n_boot, args.seed + i))

    delta_n_null = args.delta_n_null
    if delta_n_null is None:
        delta_n_null = 20 if args.full else args.n_null

    delta = None
    if not args.skip_delta:
        print(f"\n(delta control uses {delta_n_null} null replicates per arm; "
              f"scenarios and FPR use {args.n_null})")
        delta = run_delta_control(cfg, args.alpha, args.persistence,
                                  delta_n_null, args.delta_pairs, args.seed)

    fpr = None
    if not args.skip_fpr:
        fpr = estimate_fpr(cfg, args.alpha, args.persistence, args.n_null,
                           args.fpr_seeds, args.seed)

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    for r in results:
        truth = "none" if r["true_onset"] is None else r["true_onset"]
        got = "unresolved" if r["recovered_onset"] is None else r["recovered_onset"]
        print(f"  {r['scenario']:<38} true={truth!s:>5}  recovered={got}")
    if delta is not None:
        for c in delta:
            obs = ("all unresolved" if c["mean_delta"] is None
                   else f"{c['mean_delta']:+.2f}")
            print(f"  {c['condition']:<38} "
                  f"expected={c['expected_delta']:+d}  observed={obs}")
    if fpr is not None:
        print(f"  {'false-positive rate of the rule':<38} "
              f"{fpr['fpr']:.3f} vs alpha {args.alpha}")
    print("\nIf the null scenario resolved, or the FPR sits materially above "
          "alpha,\ntighten --persistence or --alpha BEFORE running this rule "
          "on real data.")
    print("The magnitude condition is a sensitivity benchmark, not a "
          "correction term.\nInterpret a real Delta_onset as a timing effect "
          "only if it exceeds the\nmagnitude-induced delay seen under a "
          "matched synthetic control; report both.")

    if args.out:
        payload = {"config": asdict(cfg),
                   "alpha": args.alpha,
                   "persistence": args.persistence,
                   "n_null": args.n_null,
                   "delta_n_null": delta_n_null,
                   "n_boot": args.n_boot,
                   "seed": args.seed,
                   "scenarios": results,
                   "delta_control": delta,
                   "fpr": fpr}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
