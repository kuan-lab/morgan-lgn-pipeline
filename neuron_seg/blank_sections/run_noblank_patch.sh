#!/bin/bash
#SBATCH --job-name=noblank_patch
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/noblank_patch_%A_%a.out
#SBATCH --partition=gpu_devel,gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=14G
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --array=0-7
#SBATCH --requeue
set -euo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
export PYTHONPATH=/gpfs/radev/home/yf354/magneton/pytorch_connectomics:${PYTHONPATH:-}
cd /gpfs/radev/project/kuan/ntl5/cell_A_full
python -u cellB_connectivity/infer_cellB_noblank.py --tile-plan magneton_inference/cellB_tile_plan_3um.json   --em-precomputed magneton_inference/cellB_em_sparse_3um_256_halo64 --out-neuron /gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_neuron_v4_3um_noblank_patch   --neuron-cfg finetune_morgan_chunk1_v4_no_myelin/config.yaml --neuron-ckpt finetune_morgan_chunk1_v4_no_myelin/checkpoint_08000.pth.tar   --tiles-from cellB_connectivity/blank_blocks.json --task-idx $SLURM_ARRAY_TASK_ID --n-tasks 8
