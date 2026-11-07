#!/bin/bash
#SBATCH --job-name=plan_naive
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/plan_naive_%j.out
#SBATCH --error=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/plan_naive_%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
set -euo pipefail
source /gpfs/radev/home/ntl5/miniconda3/etc/profile.d/conda.sh
conda activate segneuron_cpu
cd /gpfs/radev/project/kuan/ntl5/cell_A_full

python -u build_tile_plan_naive.py \
  --skel-nml "${SKEL_NML:-cellB_skel.nml}" \
  --seg-precomputed "${SEG_PC:-precomputed_cell201_merged}" \
  --seg-target-id "${SEG_ID:-1}" \
  --seg-dilation-nm "${DIL:-3000}" \
  --slab-z-tiles "${SLAB_Z:-8}" \
  --dst "${DST:-magneton_inference/cellB_tile_plan_3um_naive.json}"
