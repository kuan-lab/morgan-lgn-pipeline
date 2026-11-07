#!/bin/bash
#SBATCH --job-name=cellB_syn_v6ft_3um
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/syn_v6ft_3um_%A_%a.out
#SBATCH --error=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/syn_v6ft_3um_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --gpus=1
#SBATCH --constraint="a100|h100|h200|l40s"
#SBATCH --time=8:00:00
#SBATCH --array=0-15
#SBATCH --requeue

# Whole cellB synapse re-inference at 3 um dilation (matches cellA's biggest
# bouton; original was 500 nm and missed partner boutons further from skel).
set -euo pipefail
module load StdEnv
export SLURM_EXPORT_ENV=ALL
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
export PYTHONPATH=/gpfs/radev/project/kuan/ntl5/magneton_morgan:$PYTHONPATH
cd /gpfs/radev/project/kuan/ntl5/cell_A_full

python -u blob_infer_from_plan.py \
  --tile-plan magneton_inference/cellB_tile_plan_3um.json \
  --em-precomputed magneton_inference/cellB_em_sparse_3um_256_halo64 \
  --out-precomputed /gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_pred_v6ft_3um_sparse \
  --cfg /gpfs/radev/project/kuan/jj837/synapse_model/training/checkpoints_v6ft/config.yaml \
  --checkpoint /gpfs/radev/project/kuan/jj837/synapse_model/training/checkpoints_v6ft/checkpoint_25000.pth.tar \
  --input-size 32,256,256 --pad-size 8,64,64 \
  --task-idx ${SLURM_ARRAY_TASK_ID} --n-tasks 16
