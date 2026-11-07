"""Does the synapse model fire at interneuron (post ffi) sites as it does at RGC (post rgc) sites?
Raw v6ft probability around each annotated tip: max within ~300 nm in x,y and +-3 sections; voxels > 0.9."""
import os
os.environ['CLOUDFILES_MONITORING'] = 'false'
import numpy as np, pandas as pd
from cloudvolume import CloudVolume
pred = CloudVolume('file:///gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full/magneton_inference/cellB_pred_v6ft_3um_sharded', mip=0, fill_missing=True, bounded=False, progress=False)
P = pd.read_csv('pairing_graph_v2/analysis/pair_check.csv')
EXTRA = [('Nghia list', '88768, 83438, 2905'), ('Nghia list', '85451, 89699, 3976'), ('Nghia list', '79570, 90733, 4565'), ('Nghia list', '78216, 90549, 4782'), ('Nghia list', '79465, 90570, 4536')]
def sig(w):
    x, y, z = (int(float(v)) for v in w.split(',')); a = np.asarray(pred[x // 2 - 38:x // 2 + 38, y // 2 - 38:y // 2 + 38, z - 3:z + 4])[..., 0]
    return a.max() / 255, int((a > 229).sum())
rows = []
for r in P[P.status == 'ok'].itertuples():
    m, n = sig(r.tip_wk); rows.append(dict(who=r.who, kind=r.kind.replace(' ?', ''), tip_wk=r.tip_wk, max_prob=m, vox_gt09=n, captured=r.captured))
for w, t in EXTRA:
    m, n = sig(t); rows.append(dict(who=w, kind='your coordinate', tip_wk=t, max_prob=m, vox_gt09=n, captured=np.nan))
T = pd.DataFrame(rows); T.to_csv('pairing_graph_v2/analysis/ffi_signal.csv', index=False)
g = T[T.who != 'Nghia list'].groupby('kind').agg(tips=('kind', 'size'), median_max=('max_prob', 'median'), no_signal_lt05=('max_prob', lambda x: int((x < 0.5).sum())),
                                                 weak_05_09=('max_prob', lambda x: int(((x >= 0.5) & (x < 0.9)).sum())), strong_ge09=('max_prob', lambda x: int((x >= 0.9).sum())),
                                                 median_vox_gt09=('vox_gt09', 'median'))
print(g.to_string()); print()
print(T[T.who == 'Nghia list'][['tip_wk', 'max_prob', 'vox_gt09']].round(2).to_string(index=False))
f = T[(T.kind == 'post ffi') & (T.max_prob < 0.9)]
print(f'\npost ffi tips with no strong signal (< 0.9) within 300 nm: {len(f)}'); print(f[['who', 'tip_wk', 'max_prob', 'vox_gt09']].round(2).to_string(index=False))
