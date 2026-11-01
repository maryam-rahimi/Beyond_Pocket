#!/usr/bin/env python3
"""
Verify the null calibration's permutation machinery against cases whose
answers are known in advance.

The null is what every onset claim rests on. If the permutations are
wrong the threshold is wrong, and a wrong threshold is invisible: the
curves still look like curves and the onsets still look like layers.
These checks are therefore run before the expensive calibration, not
after it.

Each check asserts an outcome that can be derived without running the
pipeline, so a failure means the implementation disagrees with
arithmetic rather than with an expectation.

WHAT IS CHECKED

1. Within-protein permutation preserves each protein's positive rate
   exactly, and across-protein permutation does not. This is the
   difference between a null that is as hard as the real task and one
   that is easier for reasons unrelated to the probe.

2. Replicate b produces the same shuffle at every layer and for every
   feature set. Without this the saved values are 48 unrelated draws
   rather than a trajectory, and the persistence rule cannot be audited
   at all.

3. Different replicates produce different shuffles. A seeding mistake
   that made them identical would collapse the null to a point and give
   a threshold with no variability.

4. The order statistic returns a value the null actually produced, is
   conservative at the stated level, and is at least as large as the
   interpolated quantile it replaces.

5. The persistence rule finds the FIRST qualifying run, requires the
   run to be consecutive, and returns None rather than a layer when no
   run qualifies.

6. On trajectories built with known autocorrelation, the measured
   false-positive rate rises with correlation and falls with
   persistence, and the leave-one-out estimate exceeds the in-sample
   one. The last is the reason the audit is done leave-one-out.

USAGE
    python3 verify_permutations.py
"""

from __future__ import annotations

import sys

import numpy as np

ALPHA = 0.05
PASS, FAIL = "ok", "FAILED"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))
    mark = PASS if condition else FAIL
    print(f"  [{mark:>6}] {name}" + (f"  {detail}" if detail else ""))


# --- the implementations under test, copied verbatim -------------------

def perm_labels(ys, b, seed=0):
    r = np.random.default_rng(seed * 1_000_003 + b)
    return [r.permutation(y) for y in ys]


def order_tau(vals, alpha=ALPHA):
    v = np.sort(np.asarray(vals))
    k = min(int(np.ceil((1 - alpha) * (v.size + 1))), v.size)
    return float(v[k - 1])


def first_run(exceeds, layers, persist):
    run = 0
    for i, e in enumerate(exceeds):
        run = run + 1 if e else 0
        if run >= persist:
            return int(layers[i - persist + 1])
    return None


def main():
    print("1. within-protein permutation preserves per-protein "
          "positive rates")
    ys = [np.array([1] * 20 + [0] * 180), np.array([1] * 2 + [0] * 98),
          np.array([1] * 50 + [0] * 50)]
    before = [float(y.mean()) for y in ys]
    after = [float(y.mean()) for y in perm_labels(ys, 0)]
    check("rates unchanged", np.allclose(before, after),
          f"{[round(x, 3) for x in before]}")

    rng = np.random.default_rng(1)
    flat = np.concatenate(ys)
    shuffled = rng.permutation(flat)
    i, across = 0, []
    for y in ys:
        across.append(float(shuffled[i:i + len(y)].mean()))
        i += len(y)
    check("across-protein CHANGES them, as it must",
          not np.allclose(before, across),
          f"-> {[round(x, 3) for x in across]}")
    check("total positives conserved either way",
          sum(y.sum() for y in ys)
          == sum(y.sum() for y in perm_labels(ys, 0)))

    print("\n2. replicate b is identical across layers and feature sets")
    a = perm_labels(ys, 17)
    b = perm_labels(ys, 17)
    check("same replicate, two calls -> identical",
          all(np.array_equal(x, y) for x, y in zip(a, b)))
    check("holds for several replicates",
          all(all(np.array_equal(x, y) for x, y in
                  zip(perm_labels(ys, k), perm_labels(ys, k)))
              for k in (0, 1, 5, 99)))

    print("\n3. different replicates differ")
    distinct = len({perm_labels(ys, k)[0].tobytes() for k in range(50)})
    check("50 replicates give 50 distinct shuffles", distinct == 50,
          f"{distinct} distinct")

    print("\n4. the threshold is a conservative order statistic")
    v = np.sort(np.random.default_rng(2).normal(0.1, 0.02, 100))
    t = order_tau(v)
    check("tau is a value the null produced", t in set(v.tolist()),
          f"tau={t:.5f}")
    k = int(np.ceil((1 - ALPHA) * (v.size + 1)))
    check("k = ceil((1-a)(B+1)) = 96 for B=100", k == 96, f"k={k}")
    check("exceedance probability <= alpha",
          (v.size + 1 - k) / (v.size + 1) <= ALPHA,
          f"{(v.size + 1 - k) / (v.size + 1):.4f}")
    check("at least the interpolated quantile",
          t >= float(np.quantile(v, 1 - ALPHA)),
          f"vs {float(np.quantile(v, 1 - ALPHA)):.5f}")
    check("B=20 gives the maximum", order_tau(v[:20]) == v[:20].max())

    print("\n5. the persistence rule")
    lay = np.arange(10)
    check("finds the first qualifying run",
          first_run(np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 1], bool),
                    lay, 5) == 1)
    check("requires consecutiveness",
          first_run(np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1], bool),
                    lay, 5) is None)
    check("None when no run qualifies, not a default layer",
          first_run(np.zeros(10, bool), lay, 5) is None)
    check("a run at the very end still counts",
          first_run(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], bool),
                    lay, 5) == 5)
    check("exactly `persist` long is enough",
          first_run(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], bool),
                    lay, 5) == 0)

    print("\n6. false-positive rate on trajectories of known "
          "autocorrelation")
    rng = np.random.default_rng(3)
    B, L = 100, 48
    layers = np.arange(L)

    def traj(rho):
        M = np.zeros((B, L))
        for i in range(B):
            e = rng.normal(size=L)
            for j in range(1, L):
                M[i, j] = rho * M[i, j - 1] + np.sqrt(1 - rho ** 2) * e[j]
        return M

    def fpr(M, persist, loo):
        fired = 0
        for i in range(M.shape[0]):
            src = np.delete(M, i, axis=0) if loo else M
            t = np.array([order_tau(src[:, j]) for j in range(L)])
            if first_run(M[i] > t, layers, persist) is not None:
                fired += 1
        return fired / M.shape[0]

    M0, M9 = traj(0.0), traj(0.9)
    f0 = fpr(M0, 5, True)
    f9 = fpr(M9, 5, True)
    check("correlation raises the error rate", f9 > f0,
          f"rho=0: {f0:.2f}  rho=0.9: {f9:.2f}")
    r3, r5, r7 = (fpr(M9, p, True) for p in (3, 5, 7))
    check("persistence lowers it", r3 >= r5 >= r7,
          f"p=3: {r3:.2f}  p=5: {r5:.2f}  p=7: {r7:.2f}")
    ins = fpr(M9, 5, False)
    check("leave-one-out exceeds in-sample, so in-sample understates",
          f9 >= ins, f"LOO {f9:.2f} vs in-sample {ins:.2f}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        print("\nFailed:")
        for name, ok, _ in results:
            if not ok:
                print(f"  {name}")
        print("\n  Do not run the calibration until these pass: a wrong\n"
              "  null produces curves and onsets that look entirely\n"
              "  normal and are not.")
        return 1
    print("\n  The permutation machinery behaves as the calibration\n"
          "  assumes. Note that check 6 uses synthetic trajectories, so\n"
          "  it validates the RULE's behaviour under correlation, not\n"
          "  the correlation present in the real representations --\n"
          "  that is what the leave-one-out audit measures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
