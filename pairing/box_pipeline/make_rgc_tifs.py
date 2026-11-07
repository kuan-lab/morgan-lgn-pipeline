"""RGC-candidate masks for every box of the current RUN, for checking in webKnossos.  (v2, mito-first)

v1 (bouton >= 0.5 um3 with >= 1 mito voxel) let through huge dendrites with a trace of mito.
v2 works on whole waterz segments:
    presynaptic to >= 1 synapse         (kept on purpose: without it there are too many false positives)
    mito fraction >= MIN_MITO_FRAC      (mito prob > 0.5)
    volume >= MIN_UM3, no upper limit   (1 voxel = 8*8*30 nm = 1920 nm3; 0.1 um3 = 52,083 voxels;
                                         a 10 um3 cap wrongly dropped big mito-rich axons, e.g. d050_a segs 4729, 2692)
    not mostly postsynaptic             (n_post <= max(n_pre, 2))
Per box:   <RUN>/<box>/rgc_candidates_v2.tif  (Z,Y,X uint32 at 8x8x30 nm; value = segment id, 0 elsewhere)
           <RUN>/<box>/rgc_candidates_v2.csv
Combined:  analysis/<RUN>/rgc_candidates_v2_wk.nml, rgc_boxes_v2_wk.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import config as C

ap = argparse.ArgumentParser()
ap.add_argument('--mito', type=float, default=0.10, help='minimum mito fraction of the segment')
ap.add_argument('--tag', default='', help="suffix for output names, e.g. _mito02")
ap.add_argument('--also-below', type=float, default=None,
                help='also write <...>_only_below<TAG>.tif with just the segments whose mito fraction is under this value')
args = ap.parse_args()

MIN_MITO_FRAC, MIN_UM3 = args.mito, 0.1
T = args.tag
DARK = 70   # EM value below this counts as dark (myelin check)
VOX_UM3 = 8 * 8 * 30 / 1e9
os.makedirs(C.ANALYSIS_DIR, exist_ok=True)
boxes, things, bbs = [], [], []
nid = 1
for k, name in enumerate(C.load_boxes()):
    d = os.path.join(C.BOX_DIR, name)
    if not os.path.exists(f'{d}/seg.tif'):
        continue
    box = json.load(open(f'{d}/box.json'))
    (ox, oy, oz), (sx, sy, sz) = box['wk_origin'], box['wk_size']
    seg = tifffile.imread(f'{d}/seg.tif')
    mito = tifffile.imread(f'{d}/mito_prob.tif') > 127
    n = int(seg.max()) + 1
    vol = np.bincount(seg.ravel(), minlength=n)
    mv = np.bincount(seg[mito], minlength=n)
    del mito
    dark = np.bincount(seg[tifffile.imread(f'{d}/em.tif') < DARK], minlength=n) / np.maximum(vol, 1)
    syn = pd.read_csv(f'{d}/synapses.csv')
    n_pre = np.bincount(syn.pre_id, minlength=n)[:n]
    n_post = np.bincount(syn.post_id, minlength=n)[:n]
    frac = mv / np.maximum(vol, 1)
    um3 = vol * VOX_UM3
    ok = (n_pre >= 1) & (frac >= MIN_MITO_FRAC) & (um3 >= MIN_UM3) & (n_post <= np.maximum(n_pre, 2))
    ok[0] = False
    ids = np.nonzero(ok)[0]
    out = np.where(ok[seg], seg, 0).astype(np.uint32)
    del seg
    tifffile.imwrite(f'{d}/rgc_candidates_v2{T}.tif', out, compression='zlib')
    if args.also_below is not None:
        low = ok & (frac < args.also_below)
        tifffile.imwrite(f'{d}/rgc_candidates_v2{T}_only_below.tif', np.where(low[out], out, 0).astype(np.uint32), compression='zlib')
    sl = ndi.find_objects(out)
    rows = []
    for i in ids:
        s = sl[i - 1]
        sub = out[s] == i
        cz, cy, cx = (np.array(ndi.center_of_mass(sub)) + [q.start for q in s]).astype(int)
        cut = any(q.start == 0 for q in s) or any(q.stop == m for q, m in zip(s, out.shape))
        rows.append(dict(seg_id=i, vol_um3=round(um3[i], 3), vol_vox=int(vol[i]), mito_frac=round(frac[i], 3), dark_frac=round(dark[i], 3),
                         n_syn_as_pre=int(n_pre[i]), n_syn_as_post=int(n_post[i]), truncated_at_box=int(cut),
                         cz=cz, cy=cy, cx=cx, wk_x=ox + 2 * cx, wk_y=oy + 2 * cy, wk_z=oz + cz))
    del out
    c = pd.DataFrame(rows)
    c.to_csv(f'{d}/rgc_candidates_v2{T}.csv', index=False)
    boxes.append(dict(box=name, depth_um=box['depth_um'], n_candidates=len(c),
                      n_complete=int((c.truncated_at_box == 0).sum()), n_with_synapse=int((c.n_syn_as_pre > 0).sum()),
                      median_um3=round(float(c.vol_um3.median()), 2), total_um3=round(float(c.vol_um3.sum()), 1),
                      center_wk=f"{box['wk_x']}, {box['wk_y']}, {box['wk_z']}", topleft_wk=f'{ox}, {oy}, {oz}',
                      topleft_8nm=f'{ox // 2}, {oy // 2}, {oz}'))
    bbs.append(f'<userBoundingBox topLeftX="{ox}" topLeftY="{oy}" topLeftZ="{oz}" width="{sx}" height="{sy}" '
               f'depth="{sz}" color.r="1" color.g="0.8" color.b="0" color.a="1" id="{k + 1}" name="{name}" isVisible="true" />')
    for r in c.itertuples():
        things.append(f'<thing id="{nid}" color.r="1" color.g="0.2" color.b="0.2" color.a="1" '
                      f'name="{name}_seg{r.seg_id}_{r.vol_um3:.2f}um3_mito{r.mito_frac:.2f}_pre{r.n_syn_as_pre}_{"cut" if r.truncated_at_box else "full"}">'
                      f'<nodes><node id="{nid}" radius="120" x="{r.wk_x}" y="{r.wk_y}" z="{r.wk_z}" /></nodes><edges /></thing>')
        nid += 1
    print(f'{name}: {len(c)} candidate segments; segs >=0.1um3 by mito frac: '
          + '  '.join(f'>={t}: {int(((frac >= t) & (um3 >= MIN_UM3)).sum())}' for t in (0.02, 0.05, 0.1, 0.15, 0.2)), flush=True)

with open(f'{C.ANALYSIS_DIR}/rgc_candidates_v2{T}_wk.nml', 'w') as f:
    f.write('<things>\n<parameters>\n<experiment name="Morgan" />\n<scale x="4" y="4" z="30" />\n'
            + '\n'.join(bbs) + '\n</parameters>\n' + '\n'.join(things) + '\n</things>\n')
pd.DataFrame(boxes).to_csv(f'{C.ANALYSIS_DIR}/rgc_boxes_v2{T}_wk.csv', index=False)
print(pd.DataFrame(boxes).to_string(index=False))
