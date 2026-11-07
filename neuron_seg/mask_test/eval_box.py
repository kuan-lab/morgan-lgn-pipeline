"""Mask test at wk 88557, 88889, 1411 (+-500 wk xy, +-30 z): per version, labelled share, objects, merges against v1
(objects covering >= 2 big v1 objects), myelin-like voxels labelled, thin low-affinity voxels kept. Writes masktest/eval_box.csv."""
import numpy as np, pandas as pd
from cloudvolume import CloudVolume
from scipy.ndimage import binary_opening
M = '/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full'; OFF = np.array([25114, 33058, 134])
cx, cy, cz = 88557 // 2, 88889 // 2, 1411; s = np.s_[cx - 250:cx + 250, cy - 250:cy + 250, cz - 30:cz + 31]
rd = lambda p: np.asarray(CloudVolume(p, mip=0, fill_missing=True, progress=False)[s])[..., 0]
v1 = rd(f'file://{M}/graph_cellB_seg_sharded')
aff = np.asarray(CloudVolume(f'file://{M}/magneton_inference/cellB_neuron_v4_3um_sharded', mip=0, fill_missing=True, progress=False)[
    cx - 250 - OFF[0]:cx + 250 - OFF[0], cy - 250 - OFF[1]:cy + 250 - OFF[1], cz - 30 - OFF[2]:cz + 31 - OFF[2]]).astype(np.float32).mean(-1) / 255
low = aff <= 0.4; yy, xx = np.mgrid[-4:5, -4:5]; disk = yy ** 2 + xx ** 2 <= 16
thick = np.stack([binary_opening(low[:, :, z], structure=disk) for z in range(low.shape[2])], -1)   # myelin-like: thick low bands
thin = low & ~thick                                                                                  # thin low: membranes / faint processes
u1, c1 = np.unique(v1[v1 > 0], return_counts=True); big = set(u1[c1 >= 20000])
rows = []
for k, p in [('v1 (reference)', f'file://{M}/graph_cellB_seg_sharded'), ('v3 (current)', f'file://{M}/graph_cellB_v3_seg_sharded')] + \
            [(k, f'file://{M}/graph_masktest/{k}/seg_{k}') for k in ('A_nomask', 'B_voxel04', 'C_open_r4', 'D_open_r6', 'E_open_r8', 'E_open_r10', 'E_open_r12')]:
    S = v1 if k.startswith('v1') else rd(p)
    o, c = np.unique(S[S > 0], return_counts=True); merges = []
    for obj in o[c >= 20000]:
        w = S == obj; uu, cc = np.unique(v1[w], return_counts=True)
        hit = [int(a) for a, b in zip(uu, cc) if a in big and b >= 0.2 * (v1 == a).sum()]
        if len(hit) > 1: merges.append((int(obj), hit))
    rows.append(dict(version=k, labelled=round(float((S > 0).mean()), 3), objects=len(o), objects_joining_2plus_v1=len(merges),
                     v1_objects_joined=sum(len(h) for _, h in merges), myelin_like_labelled=round(float((S[thick] > 0).mean()), 3),
                     thin_low_labelled=round(float((S[thin] > 0).mean()), 3)))
    if merges: print(k, 'joins (obj: v1 objects):', merges[:5], flush=True)
T = pd.DataFrame(rows); T.to_csv('masktest/eval_box.csv', index=False)
print(f'\nbox: {(thick).mean():.3f} myelin-like (thick low), {(thin).mean():.3f} thin low, {len(big)} big v1 objects')
print(T.to_string(index=False))
