#!/bin/bash
#SBATCH --job-name=bp_extract
#SBATCH --account=nyousefi
#SBATCH --partition=normal
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --exclude=evc11,evc26
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Full Boltz-1 trunk extraction for the 1,918-system manifest.
#
# WALLTIME: 20h. The earlier inline-MSA run measured 0.92 h of work in
# 13.6 h elapsed -- 93% of the GPU reservation spent waiting on the
# public alignment server. With alignments pre-fetched to --msa-cache,
# the projection is the work figure alone: roughly 17 h for 1,913
# systems at the measured 31.8 s each. The margin is
# not padding: the stress test found one 799-residue system taking 118s
# where two others of the same length took 46s, so per-system cost has a
# long tail that the average hides. If the job is killed anyway, the
# extractor is resumable -- resubmit this same script and it skips what
# is already valid on disk.
#
# NODE EXCLUSIONS: evc11 and evc26 report an allocated GPU that CUDA
# cannot initialise (cuInit returns CUDA_ERROR_NO_DEVICE despite
# /dev/nvidia* being present). Landing on one wastes the whole
# allocation, so they are excluded rather than discovered at runtime.
#
# GPU CHOICE: a 32 GB V100 peaked at 22.5 GB on the longest system in
# the manifest (800 residues), so the length cap is certified for this
# card. A 16 GB V100 is NOT sufficient and the type is pinned for that
# reason. The torch build must be CUDA 12.x: the default CUDA 13 wheel
# has no kernels for compute capability 7.0.

set -u

RESULTS="$HOME/Beyond Pocket/03 Results"
CODE="$HOME/Beyond Pocket/02 Code"
PY="$HOME/.conda/envs/boltz/bin/python"

echo "=================================================================="
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "started   : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================================="

cd "$CODE" || exit 1

# Fail early if the environment is not the certified one, rather than
# discovering it after the queue wait.
"$PY" - <<'PYEOF' || exit 1
import sys
import torch, numpy
if not torch.cuda.is_available():
    sys.exit("CUDA unavailable on this node")
cc = torch.cuda.get_device_capability(0)
print(f"device   : {torch.cuda.get_device_name(0)} (cc {cc[0]}.{cc[1]})")
print(f"torch    : {torch.__version__} | numpy {numpy.__version__}")
if torch.version.cuda and torch.version.cuda.startswith("13"):
    sys.exit("torch is built for CUDA 13; a V100 (cc 7.0) has no kernels "
             "in that build. Install the cu126 wheel.")
# A real kernel launch, not just a capability query: the cu130 build
# reports a device and then fails here.
x = torch.randn(512, 512, device="cuda")
float((x @ x).sum())
print("cuda check: ok")
PYEOF

# Log GPU memory alongside the run, so a later OOM has evidence rather
# than only a traceback.
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 30 \
    > "gpumem_${SLURM_JOB_ID:-local}.log" 2>&1 &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

# NOTE: --allow-msa-server is deliberately NOT passed. With it, a system
# missing from the cache would be fetched live, and a retry could receive
# a different alignment while the run signature stayed the same -- which
# is exactly the frozen-input guarantee this redesign exists to provide.
# A missing cache entry should stop the run, not be filled in silently.
"$PY" extract_representations.py \
    --manifest  "$RESULTS/extraction_manifest_cached.parquet" \
    --sequences "$RESULTS/sequences.json" \
    --msa-cache "$RESULTS/msa_cache" \
    --outdir    "$RESULTS/reps_v3" \
    --no-final-z \
    --save-msa  "$RESULTS/msa"
STATUS=$?

# The extractor is deliberately tolerant of per-system failures: it logs
# them and carries on, so it can exit 0 having extracted 1,915 of 1,918.
# Slurm would then mark the job COMPLETED while the collection is not.
# "the process exited cleanly" and "the extraction finished" are
# different claims; provenance.json records the second.
if [ $STATUS -eq 0 ]; then
    "$PY" "$CODE/check_complete.py" "$RESULTS/reps_v3" || STATUS=2
fi

kill $WATCH 2>/dev/null
echo "=================================================================="
echo "finished  : $(date -Is)   exit status: $STATUS"
if [ -f "gpumem_${SLURM_JOB_ID:-local}.log" ]; then
    echo "peak GPU  : $(sort -n "gpumem_${SLURM_JOB_ID:-local}.log" | tail -1)"
fi
# From provenance rather than a glob: the extractor validates each file
# against the manifest using the same contract resume uses, which a file
# count does not.
"$PY" "$CODE/check_complete.py" "$RESULTS/reps_v3" --quiet
echo "size      : $(du -sh "$RESULTS/reps_v3" 2>/dev/null | cut -f1)"
echo "=================================================================="

# Resubmit guidance rather than automatic resubmission: a job that died
# from a code fault would otherwise resubmit itself in a loop.
if [ $STATUS -eq 2 ]; then
    echo "The extractor exited cleanly but the collection is INCOMPLETE."
    echo "See failures.csv for systems without a valid file, then"
    echo "resubmit this script to retry only those."
elif [ $STATUS -ne 0 ]; then
    echo "Non-zero exit. Check the .err file, then resubmit this same"
    echo "script to resume -- completed systems are skipped."
fi
exit $STATUS
