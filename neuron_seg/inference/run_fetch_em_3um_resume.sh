#!/bin/bash
#SBATCH --job-name=cellB_fetch_3um_r
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/fetch_em_3um_r_%A_%a.out
#SBATCH --error=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/fetch_em_3um_r_%A_%a.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=2,5,8,9

# Resume-capable fetch — retries tasks 2, 5, 9 which failed under 6hr limit.
set -euo pipefail
source /gpfs/radev/home/ntl5/miniconda3/etc/profile.d/conda.sh
conda activate segneuron_cpu
cd /gpfs/radev/project/kuan/ntl5/cell_A_full

python -u fetch_em_tile_plan_resume.py \
  --tile-plan magneton_inference/cellB_tile_plan_3um.json \
  --src s3://bossdb-open-data/morgan2020/lgn/em \
  --dst magneton_inference/cellB_em_sparse_3um_256_halo64 \
  --pad-vox 128,128,8 \
  --task-idx ${SLURM_ARRAY_TASK_ID} \
  --n-tasks 10
