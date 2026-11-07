#!/bin/bash
#SBATCH --job-name=box_cpu
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline/logs/%x_%j.err
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=2:00:00
# Step 3 for one box: waterz seg, synapse instances, pre/post pairing, boutons. Needs BOX env var.
set -euo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
# private cython cache: parallel waterz compiles deadlock on a shared ~/.cython
export HOME=/tmp/boxcpu_$SLURM_JOB_ID; mkdir -p "$HOME"
cd /gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline
python -u step3_seg_and_pair.py --box "$BOX"
