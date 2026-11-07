"""Collect the cell B pairing chunks into one synapse table, with manual (VAST) identities where traced.

  python collect_cellB.py [pairing_dir] [workers]
- keeps each synapse once (the chunk whose core contains its centroid)
- manual labels along the line pre -> synapse -> post (precomputed_full_sharded, 64 nm, one read per chunk):
  at the synapse VAST often has a synapse piece "<target> part <rgc>" (gives both partners); 150-450 nm past the
  synapse is the postsynaptic cell; at / behind the ray landing point is the presynaptic cell.
Writes <pairing_dir>/: cellB_all_synapses.csv, cellB_skel_segs.csv (global segments under the cell B skeleton).
"""
import json, os, re, sys
os.environ['CLOUDFILES_MONITORING'] = 'false'
from multiprocessing import Pool
import numpy as np, pandas as pd
from cloudvolume import CloudVolume

ROOT = '/gpfs/radev/project/kuan/ntl5/cell_A_full'
R = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else f'{ROOT}/cellB_connectivity/pairing'
NW = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
NM = np.array([4., 4., 30.])
g = pd.read_csv(f'{ROOT}/segment_groups.csv').set_index('seg_id')
PART = re.compile(r'^\*?(\d+)[A-Za-z]?\s+part\s+(\d+)')
NAME = g['name'].to_dict(); ROLE = g.role.to_dict(); CELL = g.cell_num.to_dict()
TYPE = g[g.role == 'cell'].dropna(subset=['cell_num']).drop_duplicates('cell_num').set_index('cell_num').cell_type.to_dict()
ch = pd.read_csv(f'{R}/chunks.csv').set_index('name')
_cv = None


def vast_block(c):
    """manual segmentation around one chunk + a lookup for wk points (N,3) -> segment ids (0 outside)."""
    global _cv
    if _cv is None:
        _cv = CloudVolume(f'file://{ROOT}/precomputed_full_sharded', mip=0, fill_missing=True, progress=False)
    b = _cv.bounds
    lo = np.array([c.wk_ox // 16 - 40, c.wk_oy // 16 - 40, c.wk_oz // 2 - 20]); hi = np.array([(c.wk_ox + c.wk_sx) // 16 + 41, (c.wk_oy + c.wk_sy) // 16 + 41, (c.wk_oz + c.wk_sz) // 2 + 21])
    lo = np.maximum(lo, np.array(b.minpt)); hi = np.minimum(hi, np.array(b.maxpt))
    if (hi <= lo).any():
        return lambda p: np.zeros(len(p), np.int64)
    m = np.asarray(_cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0]

    def look(p):
        i = np.rint(np.asarray(p, float) / [16, 16, 2]).astype(int) - lo
        ok = np.all((i >= 0) & (i < m.shape), axis=1)
        out = np.zeros(len(p), np.int64); out[ok] = m[i[ok, 0], i[ok, 1], i[ok, 2]]
        return out
    return look


def cell_of(ids):
    """cell number for non-'part' manual segments, nan otherwise"""
    return np.array([CELL.get(int(i), np.nan) if i and ROLE.get(int(i)) != 'part' else np.nan for i in ids], float)


def one(name):
    c = ch.loc[name]; d = f'{R}/chunks/{name}'
    if not os.path.exists(f'{d}/.done'):
        return None
    k = None
    if os.path.exists(f'{d}/skel_segs.csv'):
        k = pd.read_csv(f'{d}/skel_segs.csv'); k['chunk'] = name
    try:
        s = pd.read_csv(f'{d}/synapses.csv')
    except Exception:
        return (None, k)
    if 'wk_x' not in s or not len(s):
        return (None, k)
    core = (s.wk_x.between(c.core_lo_x, c.core_hi_x - 1) & s.wk_y.between(c.core_lo_y, c.core_hi_y - 1) & s.wk_z.between(c.core_lo_z, c.core_hi_z - 1))
    s = s[core].copy()
    if not len(s):
        return (None, k)
    s['chunk'] = name
    nn = dict(zip(k.seg_id, k.n_skeleton_nodes)) if k is not None else {}       # cell B skeleton nodes inside each local segment
    s['post_skel_nodes'] = s.post_id.map(nn).fillna(0).astype(int); s['pre_skel_nodes'] = s.pre_id.map(nn).fillna(0).astype(int)
    look = vast_block(c)
    P = s[['wk_x', 'wk_y', 'wk_z']].to_numpy(float)
    L = np.stack([c.wk_ox + 2 * s.pre_lx, c.wk_oy + 2 * s.pre_ly, c.wk_oz + s.pre_lz], 1).astype(float)      # ray landing point (pre side), wk
    s['pre_wk_x'], s['pre_wk_y'], s['pre_wk_z'] = L[:, 0].astype(int), L[:, 1].astype(int), L[:, 2].astype(int)
    v = (P - L) * NM; n = np.linalg.norm(v, axis=1); has = (s.pre_id.to_numpy() > 0) & (n > 0)
    v = np.where(has[:, None], v / np.maximum(n, 1e-9)[:, None], 0)
    at = look(P); s['vast_at_syn'] = at; s['vast_name_at_syn'] = [NAME.get(int(i), '') if i else '' for i in at]
    mm = [PART.match(str(x)) for x in s.vast_name_at_syn]
    s['ann_target'] = [float(m.group(1)) if m and int(m.group(1)) else np.nan for m in mm]
    s['ann_pre'] = [float(m.group(2)) if m else np.nan for m in mm]
    post = np.full(len(s), np.nan); pre = np.full(len(s), np.nan)
    for t in (150, 300, 450):
        q = cell_of(look(P + v * t / NM)); fill = np.isnan(post) & has & ~np.isnan(q); post[fill] = q[fill]
    for t in (0, 150, 300):
        q = cell_of(look(L - v * t / NM)); fill = np.isnan(pre) & has & ~np.isnan(q); pre[fill] = q[fill]
    q = cell_of(at)                      # synapse instance sits mostly in the postsynaptic cell
    fill = np.isnan(post) & ~np.isnan(q); post[fill] = q[fill]
    s['post_cell'] = np.where(np.isnan(post), s.ann_target, post); s['pre_cell'] = np.where(np.isnan(pre), s.ann_pre, pre)
    s.loc[s.pre_cell == s.post_cell, 'pre_cell'] = s.ann_pre[s.pre_cell == s.post_cell]      # ray landed inside the same cell
    s['post_type'] = s.post_cell.map(TYPE); s['pre_type'] = s.pre_cell.map(TYPE)
    return (s, k)


if __name__ == '__main__':
    with Pool(NW) as p:
        res = [r for r in p.imap_unordered(one, list(ch.index), chunksize=8) if r is not None]
    S = pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True)
    K = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
    S.insert(0, 'uid', np.arange(1, len(S) + 1))
    S.to_csv(f'{R}/cellB_all_synapses.csv', index=False)
    K = K[K.seg_gid > 0].groupby('seg_gid').agg(n_nodes_sum=('n_skeleton_nodes', 'sum'), n_nodes_max=('n_skeleton_nodes', 'max'),
                                                n_chunks=('chunk', 'nunique'), vox_max=('seg_vox', 'max')).reset_index().sort_values('n_nodes_sum', ascending=False)
    K.to_csv(f'{R}/cellB_skel_segs.csv', index=False)
    print(f'chunks done {len(res)}/{len(ch)}; synapses (cores) {len(S)}; quality {S.quality.value_counts().head(6).to_dict()}')
    print(f'global segments under the cell B skeleton: {len(K)}; top 5 hold {int(K.n_nodes_sum.head(5).sum())} of {int(K.n_nodes_sum.sum())} node counts')
    print('post cell known for', int(S.post_cell.notna().sum()), '; pre cell known for', int(S.pre_cell.notna().sum()))
    print('onto cell 201 by manual label:', int((S.post_cell == 201).sum()), '; of those pre type:', S[S.post_cell == 201].pre_type.fillna('untraced').value_counts().to_dict())
