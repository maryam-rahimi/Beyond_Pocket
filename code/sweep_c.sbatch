#!/bin/bash
#SBATCH --job-name=bp_sweep
#SBATCH --account=nyousefi
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Select the frozen regularisation strength C.
#
# THE RULE, PRE-REGISTERED
#   C is the largest value on the grid below that converges at every
#   one of the 48 layers and for all three feature sets, fitted on the
#   training split only. No held-out score is computed during this
#   selection, and no depth curve is inspected.
#
# WHY NOT CHOOSE BY PERFORMANCE
#   The same C must apply at every layer, or a rise in the depth curve
#   could mean the probe gained freedom rather than that the signal
#   appeared. Choosing by which value produced the most favourable
#   curves would select on the outcome; convergence is a numerical
#   requirement that carries no information about which layers look
#   good, which is what makes it a legitimate basis for a fixed choice.
#
# WHY THE WHOLE GRID AT ONCE
#   run_depth_probes.py refuses a sweep over a subset of layers or
#   feature sets, because "converges everywhere" over seven layers of
#   one feature set is not the rule. The job therefore does up to 1008
#   fits (7 C values x 3 feature sets x 48 layers), fewer in practice
#   because each C stops at its first failure.
#
# COST
#   Each fit is roughly 400,000 residues x 384 or 768 features. No GPU
#   is used; this is CPU and memory bound.
#
# RESUME
#   --sweep-state records every individual (C, feature set, layer) fit
#   as it completes. A job killed by the walltime loses only the fit
#   then in flight: resubmitting this script continues from there. The
#   state binds the index and label files by content hash, so verdicts
#   cannot be reused against different data.
#
# CPU AFFINITY
#   Thread counts are pinned to the allocation so that BLAS does not
#   spawn more threads than the job has cores, which on a shared node
#   slows everything down rather than speeding this up.

set -u

RESULTS="$HOME/Beyond Pocket/03 Results"
CODE="$HOME/Beyond Pocket/02 Code"
PY="$HOME/.conda/envs/boltz/bin/python"

# Without this, print() to a file is block-buffered and the sweep's
# verdicts sit in memory for hours, so a long run looks identical to a
# hung one.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=================================================================="
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-none}"
echo "started : $(date -Is)"
echo "threads : ${SLURM_CPUS_PER_TASK:-8}"
echo "=================================================================="

cd "$CODE" || exit 1

"$PY" run_depth_probes.py \
    --index  "$RESULTS/probe_index.parquet" \
    --labels "$RESULTS/pocket_labels_boltz1.parquet" \
    --features s zpool both \
    --sweep-state "$RESULTS/sweep_c_state.json" \
    --sweep-c 10 3 1 0.3 0.1 0.03 0.01
STATUS=$?

echo "=================================================================="
echo "finished: $(date -Is)   exit status: $STATUS"
echo "=================================================================="
echo
echo "Record the selected C. Every subsequent run -- the depth curves"
echo "and the null calibration -- uses that one value, and the null"
echo "checks it against the curves' metadata."
exit $STATUS
