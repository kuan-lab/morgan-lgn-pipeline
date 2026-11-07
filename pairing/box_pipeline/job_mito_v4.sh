#!/bin/bash
#SBATCH --job-name=box_mito
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline/logs/%x_%j.err
#SBATCH --partition=gpu_devel,gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=14G
#SBATCH --gpus=1
#SBATCH --time=0:30:00
# Mito v4 inference only (comparison), for a box whose EM / neuron / synapse outputs already exist. Needs BOX env var.
set -euo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
export PYTHONPATH=/gpfs/radev/home/yf354/magneton/pytorch_connectomics:${PYTHONPATH:-}
cd /gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline
cfg() { python -c "import config; print(config.$1)"; }
D=$(cfg RUN)/$BOX
python -u infer_synapse.py \
  --em-tif "$D/em_infer.tif" --cfg /gpfs/radev/project/kuan/ntl5/cell_A_full/models/mito_v4_jj837/config.yaml --checkpoint /gpfs/radev/project/kuan/ntl5/cell_A_full/models/mito_v4_jj837/checkpoint_80000.pth.tar \
  --out-tif "$D/mito_prob_v4_infer.tif" --tile 128,128,16 --pad 64,64,8
