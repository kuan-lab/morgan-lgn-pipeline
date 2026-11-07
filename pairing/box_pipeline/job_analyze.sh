#!/bin/bash
#SBATCH --job-name=box_analyze
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline/logs/%x_%j.out
#SBATCH --partition=devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:30:00
# Aggregate all finished boxes of the current RUN (config.py) into analysis/<RUN>/.
set -euo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
cd /gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline
python -u analyze_boxes.py
