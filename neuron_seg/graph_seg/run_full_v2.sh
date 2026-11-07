#!/bin/bash
#SBATCH -J cellBgraphv2
#SBATCH -p bigmem
#SBATCH -c 56
#SBATCH --mem=1450G
#SBATCH -t 1-00:00:00
#SBATCH -o /gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB/v2/full_%j.out
# Whole cell B, v2 = Tim's graph segmentation + myelin / low-affinity voxels as background (mean affinity <= 0.4, as our
# legacy waterz did). Resumable per core. Then merge (merge_graph_fast.py), relabel into the webKnossos frame, shard, verify.
set -eo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
export CLOUDFILES_MONITORING=false HOME=/tmp/cellBgraphv2_$SLURM_JOB_ID PYTHONPATH=/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB/pylib GRAPH_CFG=/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB/config_cellB_v2.yaml; mkdir -p $HOME
cd /gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB
keep() { grep -E "INFO\] (fragments|edges):|DONE|done in|Error|Traceback|RuntimeError|merged into|edges read|relabel" || true; }
python -u run_cellB_graph.py fragments 48 2>&1 | keep
python -u precompile_waterz.py 2>&1 | tail -1
python -u run_cellB_graph.py edges 48 2>&1 | keep
mkdir -p v2/lut; [ -f v2/lut/full_t02_roots.npy ] || python -u merge_graph_fast.py v2/edges v2/ckpt_fragments v2/lut/full_t02 0.2 0 52 2>&1 | keep
python -u relabel_fast.py v2/lut/full_t02 file:///gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/graph_cellB_v2/seg_wk_v2 44 2>&1 | grep -vE "^  [0-9]+/" | keep
conda deactivate; source /gpfs/radev/home/ntl5/miniconda3/etc/profile.d/conda.sh; conda activate segneuron_cpu
export PYTHONUSERBASE=/gpfs/radev/home/ntl5/.local; mkdir -p $HOME/.cloudfiles/locks      # locks on node scratch, not the near-full home
python -u /gpfs/radev/project/kuan/ntl5/cell_A_full/make_sharded_pyramid.py --src /gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/graph_cellB_v2/seg_wk_v2 --dst /gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/graph_cellB_v2_seg_sharded --num-mips 3 --workers 52 2>&1 | grep -vE "it/s\]|%\|" | tail -3
python3 -u verify_full_v2.py
rm -rf $HOME
echo ALL DONE
