"""Find blank (black) EM sections across cell B's z-range, sampled at several xy spots along the skeleton.
Writes cellB_connectivity/blank_sections_cellB.csv (one row per z: fraction of sampled spots that are blank)."""
import os, re
os.environ['CLOUDFILES_MONITORING'] = 'false'
import numpy as np, pandas as pd
from cloudvolume import CloudVolume
OUT = '/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity'
txt = open('/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_skel.nml').read()
P = np.array([(int(x), int(y), int(z)) for x, y, z in re.findall(r'<node [^>]*? x="(-?\d+)" y="(-?\d+)" z="(-?\d+)"', txt)])
z0, z1 = P[:, 2].min(), P[:, 2].max()
print('skeleton wk z range', z0, z1, 'nodes', len(P), flush=True)
cv = CloudVolume('s3://bossdb-open-data/morgan2020/lgn/em', mip=0, progress=False, fill_missing=True, use_https=True)
for i, s in enumerate(cv.info['scales']): print(i, s['key'], s['resolution'], s['size'], s['voxel_offset'], flush=True)
mip = max(i for i, s in enumerate(cv.info['scales']) if s['resolution'][0] <= 64)
cv.mip = mip; r = cv.info['scales'][mip]['resolution']; print('using mip', mip, r, flush=True)
rng = np.random.default_rng(0)
spots = P[rng.choice(len(P), 6, replace=False)]
res = {}
for k, (x, y, _) in enumerate(spots):
    cx, cy = int(x * 4 / r[0]), int(y * 4 / r[1])          # wk 4nm -> this mip
    a = np.asarray(cv[cx - 32:cx + 32, cy - 32:cy + 32, z0:z1 + 1])[..., 0]   # x,y,z
    res[k] = ((a == 0).mean(axis=(0, 1)) > 0.5) | (a.mean(axis=(0, 1)) < 40)
    print(f'spot {k} wk ({x},{y}): blank sections {int(res[k].sum())}', flush=True)
B = np.array(list(res.values()))
df = pd.DataFrame({'z': np.arange(z0, z1 + 1), 'frac_spots_blank': B.mean(0)})
df.to_csv(f'{OUT}/blank_sections_cellB.csv', index=False)
full = df.frac_spots_blank == 1
runs, cur = [], None
for z, b in zip(df.z, full):
    if b and cur is None: cur = [z, z]
    elif b: cur[1] = z
    elif cur is not None: runs.append(cur); cur = None
if cur: runs.append(cur)
lens = np.array([b - a + 1 for a, b in runs])
print(f'sections in range: {len(df)}; blank at all spots: {int(full.sum())}; blank at some spots only: {int(((df.frac_spots_blank > 0) & ~full).sum())}')
print(f'runs of fully blank sections: {len(runs)}; lengths: single {int((lens == 1).sum())}, 2: {int((lens == 2).sum())}, >=3: {int((lens >= 3).sum())}, longest {lens.max() if len(lens) else 0}')
print('long gaps (>=3):', [r for r in runs if r[1] - r[0] >= 2])
