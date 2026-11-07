"""Compare segmentation of the same boxes between two runs (default: blank sections handled vs not).
Usage: python compare_runs.py d050_a d050_b d050_c
Writes analysis/<NEW>/compare_vs_<OLD>.csv"""
import json, os, sys
import numpy as np, pandas as pd, tifffile
from scipy.sparse import coo_matrix
import config as C
OLD, NEW = 'boxes_z300', C.RUN
VOX = 8 * 8 * 30 / 1e9
rows = []
for name in sys.argv[1:]:
    o = tifffile.imread(f'{C.ROOT}/{OLD}/{name}/seg.tif'); n = tifffile.imread(f'{C.ROOT}/{NEW}/{name}/seg.tif')
    box = json.load(open(f'{C.ROOT}/{NEW}/{name}/box.json'))
    blank = np.array(box['blank_z'], dtype=int)
    good = np.setdiff1d(np.arange(o.shape[0]), blank)
    def stats(s):
        c = np.bincount(s.ravel()); c[0] = 0
        return int((c >= 1000).sum()), c.max() * VOX, int((c * VOX > 10).sum())
    so, sn = stats(o), stats(n)
    # variation of information on non-blank sections (sub-sampled): split = old object broken up in new, merge = the reverse
    a, b = o[good][:, ::3, ::3].ravel(), n[good][:, ::3, ::3].ravel()
    k = (a > 0) & (b > 0); a, b = a[k], b[k]
    M = coo_matrix((np.ones(len(a)), (a, b))).tocsr(); N = len(a)
    pij = M.data / N; pa = np.asarray(M.sum(1)).ravel() / N; pb = np.asarray(M.sum(0)).ravel() / N
    H = lambda p: -(p[p > 0] * np.log2(p[p > 0])).sum()
    Hab = H(pij)
    r = dict(box=name, blank_sections=len(blank), long_gaps=len(box['cuts']),
             old_segs_ge1000vox=so[0], new_segs_ge1000vox=sn[0], old_biggest_um3=round(so[1], 1), new_biggest_um3=round(sn[1], 1),
             old_segs_gt10um3=so[2], new_segs_gt10um3=sn[2],
             old_split_in_new_bits=round(Hab - H(pa), 3), new_split_in_old_bits=round(Hab - H(pb), 3))
    for tag, run in (('old', OLD), ('new', NEW)):
        d = f'{C.ROOT}/{run}/{name}'
        s = pd.read_csv(f'{d}/synapses.csv'); bt = pd.read_csv(f'{d}/boutons.csv')
        r[f'{tag}_synapses'] = len(s); r[f'{tag}_syn_ok'] = int((s.quality == 'ok').sum()); r[f'{tag}_boutons'] = len(bt)
    rows.append(r)
    print(json.dumps(r), flush=True)
    if name == 'd050_a':   # the myelin + axon merge the user found: old segment 2996
        for sid in (2996, 3781, 4729, 2692):
            m = o == sid
            ids, cnt = np.unique(n[m], return_counts=True); order = np.argsort(cnt)[::-1][:5]
            zs = np.nonzero(m.any(axis=(1, 2)))[0]
            print(f'  old seg {sid} ({m.sum() * VOX:.2f} um3, z {zs.min()}-{zs.max()}) -> new segs (id: share, new total um3): '
                  + ', '.join(f'{ids[i]}: {cnt[i] / m.sum():.2f}, {(n == ids[i]).sum() * VOX:.2f}' for i in order), flush=True)
        for (x, y, z) in ((621, 323, 162), (618, 321, 162), (670, 357, 162)):
            print(f'  at ({x},{y},{z}): old seg {o[z, y, x]}  new seg {n[z, y, x]}')
    del o, n
os.makedirs(C.ANALYSIS_DIR, exist_ok=True)
pd.DataFrame(rows).to_csv(f'{C.ANALYSIS_DIR}/compare_vs_{OLD}.csv', index=False)
print(pd.DataFrame(rows).T.to_string())
