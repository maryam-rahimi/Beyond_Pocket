#!/bin/bash
#SBATCH --job-name=bp_null
#SBATCH --account=nyousefi
#SBATCH --partition=normal
#SBATCH --array=0-47
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=null_logs/%x_%A_%a.out
#SBATCH --error=null_logs/%x_%A_%a.err

# Null calibration, one array task per Pairformer layer.
#
# WHY AN ARRAY
#   100 permutations at each of 48 layers is 4,800 probe fits, and a
#   fit plus scoring takes 8 to 10 minutes on this data. Serially that
#   is roughly 700 hours. Split by layer, each task does 100 fits at
#   one depth -- about 14 hours -- and the 48 run concurrently, so the
#   calibration finishes overnight instead of in a month.
#
#   Layers are independent: the null at layer 7 does not depend on the
#   null at layer 6. What is NOT independent is the permutation
#   identity, and that is preserved because calibrate_null.py derives
#   replicate b's shuffle from a per-replicate seed rather than from a
#   running generator. Task 7 and task 8 therefore use the SAME
#   shuffle for replicate b, which is what makes the 48 per-layer
#   files combine into genuine null trajectories.
#
# SCOPE
#   This calibrates the primary feature set only. zpool and both would
#   triple the compute, and the pre-registered primary endpoint is
#   defined on s. Run them afterwards by changing --features and the
#   output directory.
#
# AFTER IT FINISHES
#   merge_null.py combines the per-layer outputs. It refuses if any
#   layer is missing, because the onset rule is defined on the whole
#   trunk and a gap would silently shorten a run of exceedances.

set -u

RESULTS="$HOME/Beyond Pocket/03 Results"
CODE="$HOME/Beyond Pocket/02 Code"
PY="$HOME/.conda/envs/boltz/bin/python"
NULLDIR="$RESULTS/null_s"
LAYER=${SLURM_ARRAY_TASK_ID}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

mkdir -p "$NULLDIR" null_logs

OUT="$NULLDIR/null_layer_$(printf '%02d' "$LAYER").parquet"

echo "=================================================================="
echo "host    : $(hostname)"
echo "job     : ${SLURM_ARRAY_JOB_ID:-none} task ${LAYER}"
echo "layer   : ${LAYER}"
echo "started : $(date -Is)"
echo "=================================================================="

# A finished layer is not recomputed, so the array can be resubmitted
# after a partial failure and only the missing layers run.
#
# Completion means all three products, not just the threshold file.
# calibrate_null.py writes the thresholds, then the replicates, then
# the metadata; a task killed between those leaves a parquet that
# looks finished. Testing only "$OUT" would then make that layer
# permanently unrepairable: the merge refuses it for the missing
# replicate file, and every resubmission skips it.
REP="${OUT%.parquet}_replicates.parquet"
META="${OUT}.meta.json"

if [ -f "$OUT" ] && [ -f "$REP" ] && [ -f "$META" ]; then
    echo "layer ${LAYER} already complete"
    exit 0
fi

if [ -e "$OUT" ] || [ -e "$REP" ] || [ -e "$META" ]; then
    echo "layer ${LAYER} has incomplete prior output; rebuilding"
    rm -f "$OUT" "$REP" "$META"
fi

cd "$CODE" || exit 1

"$PY" calibrate_null.py \
    --index       "$RESULTS/probe_index.parquet" \
    --labels      "$RESULTS/pocket_labels_boltz1.parquet" \
    --curves-meta "$RESULTS/depth_curves.parquet.meta.json" \
    --features    s \
    --layers      "$LAYER" \
    --n-perm      100 \
    --permute     both \
    --c           10 \
    --out         "$OUT"
STATUS=$?

echo "=================================================================="
echo "finished: $(date -Is)   exit status: $STATUS"
echo "=================================================================="
exit $STATUS
