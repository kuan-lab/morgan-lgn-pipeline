#!/bin/bash
#SBATCH -J masktest
#SBATCH -p bigmem
#SBATCH -c 16
#SBATCH --mem=300G
#SBATCH -t 6:00:00
#SBATCH -o /gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB/masktest/run_%j.out
# Myelin mask test on the 4 cores around wk 88557, 88889, 1411: A no mask (v1), B per-voxel <= 0.4 (v2/v3), C/D opening r4/r6.
set -eo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
D=/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB; M=/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full
export NO_PATCH=1 CLOUDFILES_MONITORING=false HOME=/tmp/masktest_$SLURM_JOB_ID PYTHONPATH=$D/pylib; mkdir -p $HOME
cd $D
REG=18914,19414,11136,11636,1247,1307          # local x0,x1,y0,y1,z0,z1 (wk 88057-89057, 88389-89389, 1381-1441)
python -u precompile_waterz.py 2>&1 | tail -1
for k in A_nomask B_voxel04 C_open_r4 D_open_r6; do
  export GRAPH_CFG=$D/masktest/config_$k.yaml; mkdir -p masktest/$k/lut; echo "== $k"
  python -u run_cellB_graph.py fragments 4 --region $REG 2>&1 | grep -E "core .* done|done in|Error|Traceback" || true
  python -u run_cellB_graph.py edges 4 --region $REG 2>&1 | grep -E "done in|Error|Traceback" || true
  python -u merge_graph_fast.py masktest/$k/edges masktest/$k/ckpt_fragments masktest/$k/lut/t02 0.2 0 8 2>&1 | tail -1
  python -u relabel_fast.py masktest/$k/lut/t02 file://$M/graph_masktest/$k/seg_$k 4 --region $REG 2>&1 | grep -vE "^  [0-9]+/" | tail -1
done
python -u masktest/eval_box.py 2>&1 | grep -vE "Warning|warn\(|google|Python version|upgrade|best-effort|with any other|^\s*$"
rm -rf $HOME; echo ALL DONE
