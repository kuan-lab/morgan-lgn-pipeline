"""Find EM sections with no image content across cell B's z-range: black, saturated white, grey noise or washed out.

blank_sections_cellB.py only caught black frames (zeros / mean < 40). wk z 2355-2357 are white / grey-noise frames
right after the black run 2349-2354, and the models hallucinate on them just the same.
Sampled at N_SPOTS skeleton positions, 64 x 64 px at 64 nm (4 x 4 um). A real section keeps contrast at that scale
(std ~40-65); noise averages out (std 3-13). A spot is empty in a section if it is black, or its std is below
LOW_ABS, or below LOW_REL x the median std of that spot over the +-15 neighbouring sections.
Writes empty_sections_cellB.csv (one row per z).
"""
import os, re
os.environ['CLOUDFILES_MONITORING'] = 'false'
import numpy as np, pandas as pd
from cloudvolume import CloudVolume
from scipy.ndimage import median_filter
OUT = '/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_connectivity'
N_SPOTS, LOW_ABS, LOW_REL = 16, 15.0, 0.5
txt = open('/gpfs/radev/project/kuan/ntl5/cell_A_full/cellB_skel.nml').read()
P = np.array([(int(x), int(y), int(z)) for x, y, z in re.findall(r'<node [^>]*? x="(-?\d+)" y="(-?\d+)" z="(-?\d+)"', txt)])
z0, z1 = P[:, 2].min(), P[:, 2].max()
cv = CloudVolume('s3://bossdb-open-data/morgan2020/lgn/em', mip=0, progress=False, fill_missing=True, use_https=True)
mip = max(i for i, s in enumerate(cv.info['scales']) if s['resolution'][0] <= 64)
cv.mip = mip; r = cv.info['scales'][mip]['resolution']; print('using mip', mip, r, 'z', z0, z1, flush=True)
rng = np.random.default_rng(1)
spots = P[rng.choice(len(P), N_SPOTS, replace=False)]
black, low, stds, means = [], [], [], []
for k, (x, y, _) in enumerate(spots):
    cx, cy = int(x * 4 / r[0]), int(y * 4 / r[1])
    a = np.asarray(cv[cx - 32:cx + 32, cy - 32:cy + 32, z0:z1 + 1])[..., 0].astype(np.float32)
    sd = a.std(axis=(0, 1)); mn = a.mean(axis=(0, 1))
    bl = ((a == 0).mean(axis=(0, 1)) > 0.5) | (mn < 40)
    ref = median_filter(sd, size=31, mode='nearest')
    lo = ~bl & ((sd < LOW_ABS) | (sd < LOW_REL * ref))
    black.append(bl); low.append(lo); stds.append(sd); means.append(mn)
    print(f'spot {k} wk ({x},{y}): black {int(bl.sum())}, no-content but not black {int(lo.sum())}; median std {np.median(sd):.0f}', flush=True)
black, low, stds, means = map(np.array, (black, low, stds, means))
df = pd.DataFrame({'z': np.arange(z0, z1 + 1), 'frac_spots_black': black.mean(0), 'frac_spots_low_contrast': low.mean(0),
                   'frac_spots_empty': (black | low).mean(0), 'median_std': np.median(stds, 0).round(1), 'median_mean': np.median(means, 0).round(0)})
df.to_csv(f'{OUT}/empty_sections_cellB.csv', index=False)
new = df[(df.frac_spots_empty >= 0.5) & (df.frac_spots_black < 0.5)]
print(f'\nsections {len(df)}; black at >= half the spots: {int((df.frac_spots_black >= .5).sum())}; empty but NOT black at >= half the spots: {len(new)}')
print(new.to_string(index=False))
part = df[(df.frac_spots_empty > 0.1) & (df.frac_spots_empty < 0.5)]
print(f'\nempty at some spots only (10-50 %): {len(part)}'); print(part.head(60).to_string(index=False))
