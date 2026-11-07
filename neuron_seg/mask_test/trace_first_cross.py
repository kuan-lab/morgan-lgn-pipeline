"""Replay the r6 mask-test waterz agglomeration of core 3203 and find the FIRST merge that joins a region holding cell 1 (v1 859799784283)
with a region holding cell 2 (v1 859799123552; Nghia: separate cells). Every supervoxel is labelled by the v1 cell holding >= 50% of
its voxels. Lists the cell-1 / cell-2 supervoxel pairs on that boundary and any supervoxel split between the two cells.
-> masktest/first_cross_r6.csv
Run (run_trace_waterz.sh): GRAPH_CFG=masktest/config_D_open_r6.yaml NO_PATCH=1 PYTHONPATH=pylib python masktest/trace_first_cross.py"""
import csv, os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_cellB_graph as R
from waterz import agglomerate
from cloudvolume import CloudVolume
from magneton.instance_segmentation.utils.graph_utils import contact_counts
CORE, C1, C2 = 3203, 859799784283, 859799123552
OFF = np.array([25114, 33058, 134]); M = '/gpfs/radev/.marilyn/pi/kuan/ntl5/cell_A_full'
t0 = time.time()
blocks, _ = R.FS.build_grid(R.cfg, R.CV(R.INPUT, mip=0, bounded=False, progress=False)); blk = [b for b in blocks if int(b['index']) == CORE][0]
z1, z2, y1, y2, x1, x2 = blk['read']; print('core', CORE, 'read', blk['read'], flush=True)
frag, aff = R.ES._read_block(R.FRAG, R.INPUT, blk['read'], 0)
ids = np.unique(frag); local = np.searchsorted(ids, frag).astype(np.uint64); back = ids.astype(np.uint64)
if ids[0] != 0: local += np.uint64(1); back = np.concatenate([[np.uint64(0)], back])
del frag
local0 = local.copy(); contacts = contact_counts(local)
v1 = CloudVolume(f'file://{M}/graph_cellB_seg_sharded', mip=0, fill_missing=True, progress=False)
V = np.ascontiguousarray(np.transpose(np.asarray(v1[x1 + OFF[0]:x2 + OFF[0], y1 + OFF[1]:y2 + OFF[1], z1 + OFF[2]:z2 + OFF[2]])[..., 0], (2, 1, 0)))
lv = local0.ravel(); vv = V.ravel(); nl = len(back)
size = np.bincount(lv.astype(np.int64), minlength=nl); n1 = np.bincount(lv[vv == C1].astype(np.int64), minlength=nl); n2 = np.bincount(lv[vv == C2].astype(np.int64), minlength=nl)
del V, vv, lv
isA = (n1 >= 0.5 * size) & (size > 0); isB = (n2 >= 0.5 * size) & (size > 0); isA[0] = isB[0] = False
mixed = np.nonzero((np.minimum(n1, n2) >= 20))[0]
print(f'supervoxels: cell 1 {int(isA.sum())}, cell 2 {int(isB.sum())}, split between them (>= 20 vox of each) {len(mixed)}', flush=True)
st = R.MAIN.get_stage_config(R.cfg, 'edges'); thr = float(st.get('score_threshold', 0.45)); at = st.get('aff_thresholds', [0.00001, 0.99999])
gen = agglomerate(aff, [0, thr], fragments=np.ascontiguousarray(local), aff_threshold_low=at[0], aff_threshold_high=at[1],
                  scoring_function=R.ES.getScoreFunc(st.get('merge_function', 'aff50_his256')), discretize_queue=256,
                  return_merge_history=True, return_region_graph=True)
_, _, rag = next(gen); _, hist, _ = next(gen)
for _ in gen: pass
print(f'waterz done: {len(rag)} rag edges, {len(hist)} merges, {time.time() - t0:.0f}s', flush=True)
par = np.arange(nl); hasA = isA.copy(); hasB = isB.copy()
def find(x):
    r = x
    while par[r] != r: r = par[r]
    while par[x] != r: par[x], x = r, par[x]
    return r
step = None
for k, m in enumerate(hist):
    ra, rb = find(int(m['a'])), find(int(m['b']))
    if ra == rb: continue
    if (hasA[ra] and hasB[rb]) or (hasB[ra] and hasA[rb]): step = (k, m, ra, rb); break
    par[ra] = rb; hasA[rb] |= hasA[ra]; hasB[rb] |= hasB[ra]
assert step is not None, 'cell 1 and cell 2 never merge in this core'
k, m, ra, rb = step
roots = np.array([find(i) for i in range(nl)]); RA = roots == ra; RB = roots == rb
print(f'first cross-cell merge: step {k} of {len(hist)}, score {m["score"]:.4f}; region a {int(RA.sum())} svs (cell1 {int((RA & isA).sum())}, cell2 {int((RA & isB).sum())}), '
      f'region b {int(RB.sum())} svs (cell1 {int((RB & isA).sum())}, cell2 {int((RB & isB).sum())})', flush=True)
def where(l):
    p = np.argwhere(local0 == l); c = p[len(p) // 2]
    return f'{int(2 * (x1 + c[2] + OFF[0]))},{int(2 * (y1 + c[1] + OFF[1]))},{int(z1 + c[0] + OFF[2])}'
def tag(l): return 'cell1' if isA[l] else 'cell2' if isB[l] else 'other'
rows = []
for e in rag:
    u, v = int(e['u']), int(e['v'])
    if not ((RA[u] and RB[v]) or (RB[u] and RA[v])): continue
    if {tag(u), tag(v)} != {'cell1', 'cell2'}: continue
    a, b = (u, v) if isA[u] else (v, u)
    rows.append(dict(sv_cell1=int(back[a]), wk_cell1=where(a), size_cell1=int(size[a]), sv_cell2=int(back[b]), wk_cell2=where(b), size_cell2=int(size[b]),
                     own_score=round(float(e['score']), 4), contact_vox=contacts.get((min(u, v), max(u, v)), 0)))
rows.sort(key=lambda r: r['own_score'])
print(f'{len(rows)} cell-1 / cell-2 supervoxel pairs on that boundary (sorted by own score, lowest = most confident join):')
for r in rows[:20]: print(r)
for l in mixed[:20]: print(f'split supervoxel {int(back[l])} at {where(l)}: {int(n1[l])} vox cell 1, {int(n2[l])} vox cell 2, size {int(size[l])}, in region a {bool(RA[l])} b {bool(RB[l])}')
if rows:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'first_cross_r6.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'done {time.time() - t0:.0f}s')
