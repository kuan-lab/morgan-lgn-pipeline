#!/bin/bash
#SBATCH -J pairgraphv3
#SBATCH -p bigmem
#SBATCH -c 60
#SBATCH --mem=600G
#SBATCH -t 1-00:00:00
#SBATCH -o /gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/pairing_graph_v3/job_%j.out
# Synapse pairing for cell B on the whole-cell graph segmentation v3 (v2 + no-data start fix) (SEG_SOURCE=graph), then collection, analysis and
# the ground-truth checks. Results in pairing_graph_v3/ (the legacy-stitch results stay in pairing/).
cd /gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity
export SEG_SOURCE=graph PAIR_RUN=pairing_graph_v3 GRAPH_SEG=file:///gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/graph_cellB_v3_seg_sharded
./pair_chunks.sh pairing_graph_v3 local 56 | awk '/FAILED|finished/'
module load StdEnv
export PYTHONPATH=/gpfs/radev/project/kuan/ntl5/cell_A_full/magneton_local
P=/gpfs/radev/home/yf354/.conda/envs/magneton/bin/python
F="Warning|warn\(|google|Python version|upgrade|best-effort|with any other"
$P -u collect_cellB.py pairing_graph_v3 56 2>&1 | grep -vE "$F"
$P -u analyze_cellB.py pairing_graph_v3 2>&1 | grep -vE "$F" | head -45
$P -u check_pairs.py 2>&1 | grep -vE "$F"
$P -u check_triad_rgc_lin.py 2>&1 | grep -vE "$F" | head -24
echo ALL DONE
