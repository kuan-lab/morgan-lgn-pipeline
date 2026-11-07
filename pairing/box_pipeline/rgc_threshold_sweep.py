"""How the RGC-candidate count depends on the mito-fraction cutoff (other criteria fixed).
Writes analysis/<RUN>/rgc_presyn_segments.csv (every segment passing the non-mito criteria) and rgc_mito_sweep.csv."""
import os
import numpy as np, pandas as pd, tifffile
import config as C
rows = []
for name, row in C.load_boxes().items():
    d = os.path.join(C.BOX_DIR, name)
    seg = tifffile.imread(f'{d}/seg.tif'); mito = tifffile.imread(f'{d}/mito_prob.tif') > 127
    n = int(seg.max()) + 1
    vol = np.bincount(seg.ravel(), minlength=n); mv = np.bincount(seg[mito], minlength=n)
    del seg, mito
    syn = pd.read_csv(f'{d}/synapses.csv')
    n_pre = np.bincount(syn.pre_id, minlength=n)[:n]; n_post = np.bincount(syn.post_id, minlength=n)[:n]
    um3 = vol * 1920 / 1e9
    ok = (n_pre >= 1) & (um3 >= 0.1) & (n_post <= np.maximum(n_pre, 2)); ok[0] = False
    for i in np.nonzero(ok)[0]:
        rows.append(dict(box=name, depth_um=int(row['depth_um']), seg_id=i, vol_um3=round(um3[i], 3),
                         mito_frac=round(mv[i] / vol[i], 4), n_pre=int(n_pre[i]), n_post=int(n_post[i])))
t = pd.DataFrame(rows)
t.to_csv(f'{C.ANALYSIS_DIR}/rgc_presyn_segments.csv', index=False)
cuts = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
sw = pd.DataFrame([dict(mito_cutoff=c, n_segments=int((t.mito_frac >= c).sum()),
                        n_ge_0p5um3=int(((t.mito_frac >= c) & (t.vol_um3 >= 0.5)).sum()),
                        n_multisyn=int(((t.mito_frac >= c) & (t.n_pre >= 2)).sum()),
                        median_um3=round(float(t[t.mito_frac >= c].vol_um3.median()), 2)) for c in cuts])
sw.to_csv(f'{C.ANALYSIS_DIR}/rgc_mito_sweep.csv', index=False)
print(f'segments passing non-mito criteria: {len(t)}; with zero mito: {(t.mito_frac == 0).sum()}')
print(sw.to_string(index=False))
print('mito_frac quantiles among segments with any mito:', t[t.mito_frac > 0].mito_frac.quantile([.1, .25, .5, .75, .9]).round(3).tolist())
print('histogram of mito_frac (segments with any mito):')
h, e = np.histogram(t[t.mito_frac > 0].mito_frac, bins=[0, .01, .02, .05, .1, .15, .2, .3, .5, 1])
for a, b, c in zip(e[:-1], e[1:], h): print(f'  {a:.2f}-{b:.2f}: {c}')
b = t[(t.mito_frac >= 0.02) & (t.mito_frac < 0.10)]
print(f'segments between 2% and 10% mito (dropped by the 10% cutoff): {len(b)}, median {b.vol_um3.median():.2f} um3, >=2 pre: {(b.n_pre >= 2).sum()}')
