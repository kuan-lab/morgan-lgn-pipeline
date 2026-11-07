#!/bin/bash
#SBATCH -J cellBgraphv3
#SBATCH -p bigmem
#SBATCH -c 56
#SBATCH --mem=1450G
#SBATCH -t 1-00:00:00
#SBATCH -o /gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB/v2/full_v3_%j.out
# v3 = v2 + no-data fix: wk z 134-174 (before the EM volume starts) outside the data, and the tiles next to it (and next to the
# empty sections wk 193, 8959) re-inferred without the empty sections (nodatafix.json, patch volume). Only the affected cores are
# redone: fragments for core layers 0 and 34, edges for layers 0, 1, 33, 34 (fragment ids are per core, so neighbours' links go stale).
# Old checkpoints / edges are moved to v2/fixbackup_v3 (not deleted). v2 outputs (graph_cellB_v2_seg_sharded) are not touched.
set -eo pipefail
module load StdEnv
source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
conda activate /gpfs/radev/home/yf354/.conda/envs/magneton
D=/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity/graph_cellB; M=/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full
export CLOUDFILES_MONITORING=false HOME=/tmp/cellBgraphv3_$SLURM_JOB_ID PYTHONPATH=$D/pylib GRAPH_CFG=$D/config_cellB_v2.yaml; mkdir -p $HOME
cd $D
keep() { grep -E "INFO\] (fragments|edges):|DONE|done in|Error|Traceback|RuntimeError|merged into|edges read|relabel" || true; }
python - <<'PY'
import json, os, shutil, run_cellB_graph as R
blocks, _ = R.cores(); sel = json.load(open('cores.json'))['cores']
z1 = {i: blocks[i]['core'][0] for i in sel}
frag = [i for i in sel if z1[i] in (0, 8704)]; edge = [i for i in sel if z1[i] in (0, 256, 8448, 8704)]
B = 'v2/fixbackup_v3'; os.makedirs(B, exist_ok=True); n = 0
for i in frag:
    for ext in ('done', 'json'):
        f = f'v2/ckpt_fragments/block_{i:04d}.{ext}'
        if os.path.exists(f): shutil.move(f, os.path.join(B, os.path.basename(f))); n += 1
for i in edge:
    f = f'v2/edges/edges_{i:04d}.npz'
    if os.path.exists(f): shutil.move(f, os.path.join(B, os.path.basename(f))); n += 1
json.dump(dict(fragments=frag, edges=edge), open('v2/v3_redo_cores.json', 'w'))
print(f'v3 redo: fragments {len(frag)} cores, edges {len(edge)} cores; moved {n} files to {B}', flush=True)
PY
python -u run_cellB_graph.py fragments 48 2>&1 | keep
python -u precompile_waterz.py 2>&1 | tail -1
python -u run_cellB_graph.py edges 48 2>&1 | keep
python -u merge_graph_fast.py v2/edges v2/ckpt_fragments v2/lut/v3_t02 0.2 0 52 2>&1 | keep
python -u relabel_fast.py v2/lut/v3_t02 file://$M/graph_cellB_v3/seg_wk_v3 44 2>&1 | grep -vE "^  [0-9]+/" | keep
conda deactivate; source /gpfs/radev/home/ntl5/miniconda3/etc/profile.d/conda.sh; conda activate segneuron_cpu
export PYTHONUSERBASE=/gpfs/radev/home/ntl5/.local; mkdir -p $HOME/.cloudfiles/locks
python -u /gpfs/radev/project/kuan/ntl5/cell_A_full/make_sharded_pyramid.py --src $M/graph_cellB_v3/seg_wk_v3 --dst $M/graph_cellB_v3_seg_sharded --num-mips 3 --workers 52 2>&1 | grep -vE "it/s\]|%\|" | tail -3
python3 -u verify_full_v3.py
python3 - <<'PY'
import numpy as np
from cloudvolume import CloudVolume
for nm, p in (('v2', 'graph_cellB_v2_seg_sharded'), ('v3', 'graph_cellB_v3_seg_sharded')):
    cv = CloudVolume(f'file:///gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/{p}', mip=0, fill_missing=True, progress=False); x, y = 99104 // 2, 79698 // 2
    for z in (176, 182, 194, 230):
        w = np.asarray(cv[x - 125:x + 125, y - 125:y + 125, z:z + 1]).ravel(); u, c = np.unique(w[w > 0], return_counts=True)
        print(f'{nm} wk 99104, 79698, {z}: largest object {c.max() / w.size if len(c) else 0:.2f} of 2x2 um window, {len(u)} objects')
PY
rm -rf $HOME
echo ALL DONE
