"""Compare mito v3 vs v4 on one box: per-segment mito fraction, RGC candidates at the 4.5 % rule, the user's labels.
Usage: python compare_mito_v4.py d050_a   (needs <RUN>/<box>/mito_prob_v4_infer.tif)"""
import json, os, sys
import numpy as np, pandas as pd, tifffile
import config as C
box = sys.argv[1]; d = os.path.join(C.BOX_DIR, box)
src = np.array(json.load(open(f'{d}/box.json'))['src'])
m4 = tifffile.imread(f'{d}/mito_prob_v4_infer.tif')[src]
tifffile.imwrite(f'{d}/mito_prob_v4.tif', m4, compression='zlib')
seg = tifffile.imread(f'{d}/seg.tif'); m3 = tifffile.imread(f'{d}/mito_prob.tif')
n = int(seg.max()) + 1
vol = np.bincount(seg.ravel(), minlength=n)
f3 = np.bincount(seg[m3 > 127], minlength=n) / np.maximum(vol, 1)
f4 = np.bincount(seg[m4 > 127], minlength=n) / np.maximum(vol, 1)
print(f'mito voxels (prob>0.5): v3 {(m3 > 127).mean():.4f} of box, v4 {(m4 > 127).mean():.4f}')
syn = pd.read_csv(f'{d}/synapses.csv')
npre = np.bincount(syn.pre_id, minlength=n)[:n]; npost = np.bincount(syn.post_id, minlength=n)[:n]
um3 = vol * 1920 / 1e9
base = (npre >= 1) & (um3 >= 0.1) & (npost <= np.maximum(npre, 2)); base[0] = False
c3, c4 = base & (f3 >= 0.045), base & (f4 >= 0.045)
print(f'candidates at 4.5 %: v3 {c3.sum()}  v4 {c4.sum()}  both {(c3 & c4).sum()}  only v3 {(c3 & ~c4).sum()}  only v4 {(c4 & ~c3).sum()}')
lab = {1321509: 'right', 1334622: 'right', 1091696: 'right', 753971: 'right', 1572211: 'wrong', 1333471: 'maybe wrong', 857170: 'wrong (hard)'}
print(pd.DataFrame([dict(seg=s, label=l, vol_um3=round(um3[s], 2), pre=int(npre[s]), post=int(npost[s]),
                         mito_v3=round(f3[s], 4), mito_v4=round(f4[s], 4)) for s, l in lab.items()]).to_string(index=False))
ids = np.nonzero(base & ((f3 > 0) | (f4 > 0)))[0]
t = pd.DataFrame(dict(seg=ids, vol_um3=um3[ids].round(3), pre=npre[ids], post=npost[ids], mito_v3=f3[ids].round(4), mito_v4=f4[ids].round(4)))
t.to_csv(f'{C.ANALYSIS_DIR}/mito_v3_vs_v4_{box}.csv', index=False)
print(f'\nsegments passing non-mito criteria with any mito: {len(t)}; median v4/v3 mito ratio among v3>=4.5%: '
      f'{np.median(t[t.mito_v3 >= 0.045].mito_v4 / t[t.mito_v3 >= 0.045].mito_v3):.2f}')
h = lambda v: np.histogram(v[v > 0], bins=[0, .01, .02, .045, .1, .2, .3, 1])[0]
print('mito fraction histogram (bins 0-1-2-4.5-10-20-30-100 %): v3', h(t.mito_v3.values), ' v4', h(t.mito_v4.values))
out = np.where(c4[seg], seg, 0).astype(np.uint32)
tifffile.imwrite(f'{d}/rgc_candidates_v2_mito045_v4.tif', out, compression='zlib')
print(f'wrote {d}/rgc_candidates_v2_mito045_v4.tif')
