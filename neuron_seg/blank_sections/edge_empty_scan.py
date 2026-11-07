"""Empty (zero) EM sections in plan tiles at the two z ends of the cell B tile plan, which the skeleton-range blank scan
(wk 201-8925) never covered. Per tile and section: fraction of zero pixels (stride 4). Writes edge_empty_tiles.csv."""
import json, numpy as np, pandas as pd
from multiprocessing import Pool
from cloudvolume import CloudVolume
P = json.load(open('/gpfs/radev/project/kuan/ntl5/cell_A_full/magneton_inference/cellB_tile_plan_3um.json')); T = np.array(P['tiles'])
EM = 'file:///gpfs/radev/project/kuan/ntl5/cell_A_full/magneton_inference/cellB_em_sparse_3um_256_halo64'
LO, HI = 201 - 134, 8925 - 134                      # local z covered by the old scan
sel = np.where((T[:, 2] < LO + 32) | (T[:, 5] > HI - 32))[0]
def one(i):
    cv = CloudVolume(EM, mip=0, fill_missing=True, progress=False); x0, y0, z0, x1, y1, z1 = T[i]
    a = np.asarray(cv[x0:x1:4, y0:y1:4, z0:z1])[..., 0]; zf = (a == 0).mean((0, 1))
    return [(int(i), int(z0 + k + 134), float(zf[k])) for k in range(len(zf))]
if __name__ == '__main__':
    with Pool(10) as p: R = [r for rr in p.map(one, sel, chunksize=8) for r in rr]
    D = pd.DataFrame(R, columns=['tile', 'wk_z', 'zero_frac']); D.to_csv('edge_empty_tiles.csv', index=False)
    print('edge tiles scanned', len(sel), '| sections x tiles', len(D))
    e = D[D.zero_frac > 0.95]; p = D[(D.zero_frac > 0.2) & (D.zero_frac <= 0.95)]
    print('fully empty (>95% zero):', len(e), 'tile-sections in', e.tile.nunique(), 'tiles; wk z', e.wk_z.min() if len(e) else '-', '-', e.wk_z.max() if len(e) else '-')
    print('partly empty (20-95%):', len(p), 'tile-sections in', p.tile.nunique(), 'tiles')
    print(D.groupby('wk_z').zero_frac.apply(lambda s: (s > 0.95).mean()).round(2).loc[lambda s: s > 0].to_string())
