#!/bin/bash
#SBATCH --job-name=box_gpu
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline/logs/%x_%j.err
#SBATCH --partition=gpu_devel,gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=14G
#SBATCH --gpus=1
#SBATCH --time=1:50:00
# Steps 1-2 for one box: fetch EM, neuron affinity (v4), synapse prob (v6ft), mito prob (v3). Needs BOX env var.
set -euo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
export PYTHONPATH=/gpfs/radev/home/yf354/magneton/pytorch_connectomics:${PYTHONPATH:-}
cd /gpfs/radev/project/kuan/ntl5/cell_A_full/box_pipeline
cfg() { python -c "import config; print(config.$1)"; }
D=$(cfg RUN)/$BOX

python -u step1_fetch_em.py --box "$BOX"

python -u infer_neuron_aff.py \
  --config-base "$(cfg NEURON_CFG)" --checkpoint "$(cfg NEURON_CKPT)" \
  --em "$D/em_infer.tif" --out "$D/aff.tif"

python -u infer_synapse.py \
  --em-tif "$D/em_infer.tif" --cfg "$(cfg SYN_CFG)" --checkpoint "$(cfg SYN_CKPT)" \
  --out-tif "$D/syn_prob_infer.tif" --tile 128,128,16 --pad 64,64,8

python -u infer_synapse.py \
  --em-tif "$D/em_infer.tif" --cfg "$(cfg MITO_CFG)" --checkpoint "$(cfg MITO_CKPT)" \
  --out-tif "$D/mito_prob_infer.tif" --tile 128,128,16 --pad 64,64,8
echo "GPU steps done for $BOX"
