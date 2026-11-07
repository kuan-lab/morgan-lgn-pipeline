#!/bin/bash
#SBATCH --job-name=shard_syn_v6ft_3um
#SBATCH --output=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/shard_syn_v6ft_3um_%j.out
#SBATCH --error=/gpfs/radev/project/kuan/ntl5/cell_A_full/logs/shard_syn_v6ft_3um_%j.err
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
source /gpfs/radev/home/ntl5/miniconda3/etc/profile.d/conda.sh
conda activate segneuron_cpu
rm -rf ~/.cloudfiles/locks && mkdir -p ~/.cloudfiles/locks
cd /gpfs/radev/project/kuan/ntl5/cell_A_full

python -u make_sharded_pyramid.py \
  --src /gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_pred_v6ft_3um_sparse \
  --dst /gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_pred_v6ft_3um_sharded \
  --num-mips 3 --workers 4 --memory-target 2000000000

python3 <<'PY'
import json
BASE = [25114, 33058, 134]
p = '/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_pred_v6ft_3um_sharded/info'
info = json.load(open(p))
for s in info['scales']:
    sx = s['resolution'][0] // 8; sy = s['resolution'][1] // 8
    s['voxel_offset'] = [BASE[0]//sx, BASE[1]//sy, BASE[2]]
    print(f"{s['key']}: vo={s['voxel_offset']}")
json.dump(info, open(p, 'w'), indent=2)
PY

python3 <<'PY'
import os
os.environ["CLOUDFILES_MONITORING"] = "false"
from cloudvolume import CloudVolume
import numpy as np
p = 'file:///gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_pred_v6ft_3um_sharded'
for mip in [0, 1, 2, 3]:
    cv = CloudVolume(p, mip=mip, fill_missing=True, progress=False)
    b = cv.bounds
    z_mid = (int(b.minpt[2])+int(b.maxpt[2]))//2
    sx = min(2048, int(b.maxpt[0])-int(b.minpt[0]))
    arr = cv[int(b.minpt[0]):int(b.minpt[0])+sx, int(b.minpt[1]):int(b.maxpt[1]), z_mid:z_mid+1]
    nz = int((np.asarray(arr) != 0).sum())
    print(f'[VERIFY] mip{mip}: nz={nz}')
    assert nz > 0
print('[OK] all mips OK')
PY
echo "DONE"
