#!/bin/bash
#SBATCH --job-name=bp_msa
#SBATCH --account=nyousefi
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
R="$HOME/Beyond Pocket/03 Results"
cd "$HOME/Beyond Pocket/02 Code"
~/.conda/envs/boltz/bin/python prefetch_msas.py \
  --sequences "$R/sequences.json" --outdir "$R/msa_cache"
