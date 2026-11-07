#!/bin/bash
#SBATCH -J cellBgraphv4
#SBATCH -p bigmem
#SBATCH -c 52
#SBATCH --mem=1450G
#SBATCH -t 1-00:00:00
#SBATCH -o /gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB/v4/full_v4_%j.out
# Whole cell B, v4 = v3 (blank-section / no-data patch, rebuilt 2026-10-28 by job 3217180) + myelin by opening (r6) instead of the
# per-voxel 0.4 mask. Supervoxels change everywhere, so every stage runs from scratch (resumable per core). v1-v3 outputs are not touched.
#   run_full_v4.sh percore   fragments + edges (per core; ~20 GB per worker, fits next to other jobs with fewer workers: NW)
#   run_full_v4.sh global    merge, relabel, shard, verify (v2 peaked at 1.23 TiB here)
set -eo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
D=/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB; M=/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full
export CLOUDFILES_MONITORING=false HOME=/tmp/cellBgraphv4_$SLURM_JOB_ID PYTHONPATH=$D/pylib GRAPH_CFG=$D/config_cellB_v4.yaml; mkdir -p $HOME
cd $D; STAGE=${1:-all}; NW=${NW:-48}
keep() { grep -E "INFO\] (fragments|edges):|DONE|done in|Error|Traceback|RuntimeError|merged into|edges read|relabel" || true; }
if [ $STAGE != global ]; then
python -u run_cellB_graph.py fragments $NW 2>&1 | keep
python -u precompile_waterz.py 2>&1 | tail -1
python -u run_cellB_graph.py edges $NW 2>&1 | keep
[ $STAGE = percore ] && { rm -rf $HOME; echo PERCORE DONE; exit 0; }
fi
mkdir -p v4/lut; [ -f v4/lut/full_t02_roots.npy ] || python -u merge_graph_fast.py v4/edges v4/ckpt_fragments v4/lut/full_t02 0.2 0 50 2>&1 | keep
python -u relabel_fast.py v4/lut/full_t02 file://$M/graph_cellB_v4/seg_wk_v4 44 2>&1 | grep -vE "^  [0-9]+/" | keep
conda deactivate; source /gpfs/radev/home/ntl5/miniconda3/etc/profile.d/conda.sh; conda activate segneuron_cpu
export PYTHONUSERBASE=/gpfs/radev/home/ntl5/.local; mkdir -p $HOME/.cloudfiles/locks      # locks on node scratch, not the near-full home
python -u /gpfs/radev/project/kuan/ntl5/cell_A_full/make_sharded_pyramid.py --src $M/graph_cellB_v4/seg_wk_v4 --dst $M/graph_cellB_v4_seg_sharded --num-mips 3 --workers 50 2>&1 | grep -vE "it/s\]|%\|" | tail -3
python3 -u verify_full_v4.py
rm -rf $HOME
echo ALL DONE
