"""Is the r6 merge of cell 1 (v1 859799784283) and cell 2 (v1 859799123552) a waterz merge-ORDER effect? Same supervoxels (no-mask
run) and same affinity on a crop around the first cross merge (wk 88690, 89365, z 1395/1396); agglomerate to 0.6 as in the
edges stage, with the supervoxel ids randomly renumbered (waterz breaks ties in a queue bin by insertion order).
Reports per permutation whether a region holds >= 20% of both cells' pixels."""
import numpy as np, sys
from cloudvolume import CloudVolume
from waterz import agglomerate
from magneton.instance_segmentation.waterz_block import getScoreFunc
M = '/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full'; OFF = np.array([25114, 33058, 134])
cx, cy = 88690 // 2, 89365 // 2; R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
a0 = np.array([cx - R, cy - R, 1380]); a1 = np.array([cx + R, cy + R, 1404]); l0, l1 = a0 - OFF, a1 - OFF
rd = lambda p, lo, hi: np.asarray(CloudVolume(f'file://{M}/{p}', mip=0, fill_missing=True, progress=False)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
v1 = np.transpose(rd('graph_cellB_seg_sharded', a0, a1)[..., 0], (2, 1, 0)); c1, c2 = v1 == 859799784283, v1 == 859799123552
A = np.ascontiguousarray(np.transpose(rd('magneton_inference/cellB_neuron_v4_3um_sharded', l0, l1), (3, 2, 1, 0)).astype(np.float32) / 255)
for run in ('A_nomask', 'D_open_r6'):
    F = np.transpose(rd(f'graph_masktest/{run}/fragments', l0, l1)[..., 0], (2, 1, 0)).astype(np.uint64)
    ids = np.unique(F); L = np.searchsorted(ids, F).astype(np.uint64); n = len(ids)
    res = []
    for seed in range(12):
        perm = np.arange(n, dtype=np.uint64)
        if seed: rng = np.random.default_rng(seed); perm[1:] = rng.permutation(n - 1).astype(np.uint64) + 1   # keep 0 = background
        seg = next(agglomerate(A, [0.6], fragments=np.ascontiguousarray(perm[L]), scoring_function=getScoreFunc('aff50_his256'), discretize_queue=256))
        s1 = np.unique(seg[c1], return_counts=True); s2 = np.unique(seg[c2], return_counts=True)
        big1 = {int(k) for k, c in zip(*s1) if k and c >= 0.2 * c1.sum()}; big2 = {int(k) for k, c in zip(*s2) if k and c >= 0.2 * c2.sum()}
        res.append('MERGED' if big1 & big2 else 'separate')
    print(f'{run}: crop {2 * R}x{2 * R}x24 px, {n} supervoxels; original order + 11 random orders: {res}', flush=True)
