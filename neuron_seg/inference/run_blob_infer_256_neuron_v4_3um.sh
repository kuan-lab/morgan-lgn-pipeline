#!/bin/bash
#SBATCH --job-name=cellB_blob_neuron_v4_3um
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/blob_neuron_v4_3um_%A_%a.out
#SBATCH --error=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/blob_neuron_v4_3um_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --gpus=1
#SBATCH --constraint="a100|h100|h200|l40s"
#SBATCH --time=8:00:00
#SBATCH --array=0-15
#SBATCH --requeue

# Whole cellB neuron aff, v4 model (myelin -> background), 2 um dilation plan.
set -euo pipefail
module load StdEnv
export SLURM_EXPORT_ENV=ALL
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
export PYTHONPATH=/gpfs/radev/project/kuan/ntl5/magneton_morgan:$PYTHONPATH
cd /gpfs/radev/project/kuan/ntl5/cell_A_full

python -u blob_infer_from_plan_3ch.py \
  --tile-plan magneton_inference/cellB_tile_plan_3um.json \
  --em-precomputed magneton_inference/cellB_em_sparse_3um_256_halo64 \
  --out-precomputed /gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_neuron_v4_3um_sparse \
  --cfg /gpfs/radev/project/kuan/ntl5/cell_A_full/finetune_morgan_chunk1_v4_no_myelin/config.yaml \
  --checkpoint /gpfs/radev/project/kuan/ntl5/cell_A_full/finetune_morgan_chunk1_v4_no_myelin/checkpoint_08000.pth.tar \
  --input-size 32,256,256 --pad-size 8,64,64 \
  --task-idx ${SLURM_ARRAY_TASK_ID} --n-tasks 16
