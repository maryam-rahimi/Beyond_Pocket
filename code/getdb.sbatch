#!/bin/bash
#SBATCH --job-name=bp_db
#SBATCH --account=nyousefi
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
DB="$HOME/msa_db"
mkdir -p "$DB" && cd "$DB"
echo "start $(date -Is)"
curl -L -C - -o uniref30_2302.tar.gz \
  https://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2302.tar.gz
echo "downloaded $(date -Is)"
ls -la
tar xzf uniref30_2302.tar.gz
echo "extracted $(date -Is)"
ls -la
