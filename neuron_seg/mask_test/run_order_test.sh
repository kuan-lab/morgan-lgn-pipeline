#!/bin/bash
#SBATCH -J wzorder
#SBATCH -p devel
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB/masktest/order_test_%j.out
set -eo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
D=/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB
export CLOUDFILES_MONITORING=false PYTHONPATH=$D/pylib HOME=/tmp/wzo_$SLURM_JOB_ID; mkdir -p $HOME
cd $D; python -u precompile_waterz.py 2>&1 | tail -1
for R in 100 250; do python -u masktest/order_test.py $R 2>&1 | grep -E "^A_nomask|^D_open|Error|Traceback"; done
rm -rf $HOME
