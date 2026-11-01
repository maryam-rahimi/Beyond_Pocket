#!/bin/bash
#SBATCH --job-name=bp_curves
#SBATCH --account=nyousefi
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Depth curves: AUPRC against Pairformer depth, per familiarity regime.
#
# C IS FROZEN AT 10
#   Selected by the pre-registered admissibility rule in job 1016892:
#   the largest value on the grid 10, 3, 1, 0.3, 0.1, 0.03, 0.01 that
#   converged at all 48 layers for all three feature sets, fitted on
#   training data only. C=10 passed all 144 fits, so the probe is as
#   weakly constrained as the rule permits and the onset cannot be an
#   artefact of an over-regularised readout.
#
#   This value is not revisited. Changing it after seeing a curve would
#   be selecting on the outcome, which is what fixing it in advance
#   exists to prevent.
#
# RESUME
#   Each (feature set, layer) is appended to the output as it
#   completes. If the walltime kills this job, resubmit with the
#   partial file as --resume-from and it continues rather than
#   restarting. The sweep needed 12 hours for 144 fits alone; this does
#   the same fits plus scoring 534 test proteins at each, so a single
#   20-hour window may not be enough.
#
# RESUBMITTING IS ALL THAT IS NEEDED
#   The command below adds --resume-from automatically when a partial
#   output exists, so the same `sbatch curves.sbatch` both starts and
#   continues the run. Nothing is edited between attempts, and there is
#   no way to forget the flag and silently restart from zero.

set -u

RESULTS="$HOME/Beyond Pocket/03 Results"
CODE="$HOME/Beyond Pocket/02 Code"
PY="$HOME/.conda/envs/boltz/bin/python"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=================================================================="
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-none}"
echo "started : $(date -Is)"
echo "C       : 10 (frozen by the admissibility sweep, job 1016892)"
echo "=================================================================="

cd "$CODE" || exit 1

CURVES="$RESULTS/depth_curves.parquet"
RESUME_ARGS=()
if [ -f "$CURVES" ]; then
    echo "partial curves found; resuming from $CURVES"
    RESUME_ARGS=(--resume-from "$CURVES")
fi

"$PY" run_depth_probes.py \
    --index    "$RESULTS/probe_index.parquet" \
    --labels   "$RESULTS/pocket_labels_boltz1.parquet" \
    --out      "$CURVES" \
    --features s zpool both \
    --c 10 \
    "${RESUME_ARGS[@]}"
STATUS=$?

echo "=================================================================="
echo "finished: $(date -Is)   exit status: $STATUS"
echo "=================================================================="
if [ $STATUS -ne 0 ]; then
  echo
  echo "If this was a walltime kill, the completed layers are already"
  echo "in depth_curves.parquet. Just run sbatch curves.sbatch again;"
  echo "it resumes automatically from the last completed layer."
fi
exit $STATUS
